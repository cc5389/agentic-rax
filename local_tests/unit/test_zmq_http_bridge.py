"""The robopil HTTP-over-ZMQ bridge against a simulated NUC bridge: nothing here touches hardware."""
import threading
import time

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from hardware.zmq_http_bridge import Bridge, MockNuc, NucError, Refused, serve
from services.franka.bridges import ArmBridge, GripperBridge, TargetRejected
from services.franka.kinematics import flange_pose

START_Q = np.array([-0.0169, -0.6314, -0.0054, -2.3977, -0.0217, 1.7756, 0.7776])


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def make(**kwargs):
    clock = Clock()
    nuc = MockNuc(START_Q, clock=clock, object_width_m=kwargs.pop('object_width_m', None))
    return Bridge(nuc, clock=clock, **kwargs), nuc, clock


def run(bridge, clock, seconds, *, feed=None):
    for _ in range(int(round(seconds / bridge.dt))):
        clock.t += bridge.dt
        if feed:
            feed()
        bridge.tick()


def flange(q):
    F = flange_pose(q)
    return F[:3, 3], Rotation.from_matrix(F[:3, :3]).as_quat()


def test_pose_target_is_rate_limited_and_reached():
    bridge, nuc, clock = make()
    position, quaternion = flange(START_Q)
    goal = position + np.array([0.03, -0.02, -0.04])
    bridge.set_target(goal, quaternion)
    previous = position
    for _ in range(90):  # 3 s, refreshed like the backend's 20 Hz servo
        bridge.set_target(goal, quaternion)
        run(bridge, clock, bridge.dt)
        now, _ = flange(bridge.q_cmd)
        assert np.linalg.norm(now - previous) <= 0.05 * bridge.dt + 1e-6
        previous = now
    assert np.linalg.norm(flange(nuc.q)[0] - goal) < 1e-3
    assert nuc.streaming


def test_watchdog_holds_without_new_targets():
    bridge, nuc, clock = make()
    position, quaternion = flange(START_Q)
    bridge.set_target(position + [0.0, 0.0, -0.1], quaternion)
    run(bridge, clock, 0.4)
    assert bridge.mode == 'target'
    run(bridge, clock, 0.2)
    assert bridge.mode == 'hold'
    held = bridge.q_cmd.copy()
    run(bridge, clock, 0.5)
    assert np.allclose(bridge.q_cmd, held)
    assert 'watchdog' in bridge.events[-1][1]


def test_refusals_are_409():
    bridge, _, _ = make()
    position, quaternion = flange(START_Q)
    with pytest.raises(Refused, match='floor'):
        bridge.set_target(position + [0.0, 0.0, -0.8], quaternion)
    with pytest.raises(Refused, match='IK'):
        bridge.set_target([1.5, 0.0, 0.3], quaternion)
    bad = START_Q.copy()
    bad[5] = 0.2  # J6 below the FR3 lab limit 0.5745 (the Panda limit would allow it)
    with pytest.raises(Refused, match='J6'):
        bridge.move_joints(bad, 2.0)


def test_move_joints_follows_a_timed_path():
    bridge, nuc, clock = make()
    goal = START_Q + np.array([0.2, -0.1, 0.0, 0.1, 0.0, 0.1, -0.2])
    result = {}
    worker = threading.Thread(target=lambda: result.update(bridge.move_joints(goal, 1.0)))
    worker.start()
    while bridge.mode != 'joints':
        time.sleep(0.001)
    run(bridge, clock, 1.2)
    worker.join(5)
    assert result['reached'] and np.allclose(nuc.q, goal, atol=1e-6)
    assert bridge.mode == 'hold'


def test_hold_interrupts_a_joint_move():
    bridge, _, clock = make()
    errors = []

    def move():
        try:
            bridge.move_joints(START_Q + 0.3, 2.0)
        except NucError as exc:
            errors.append(exc)
    worker = threading.Thread(target=move)
    worker.start()
    while bridge.mode != 'joints':
        time.sleep(0.001)
    run(bridge, clock, 0.3)
    bridge.hold()
    worker.join(5)
    assert errors and bridge.mode == 'hold'


def test_tracking_lag_holds():
    bridge, nuc, clock = make(tracking_limit=0.05)
    position, quaternion = flange(START_Q)
    bridge.set_target(position + [0.0, 0.0, -0.05], quaternion)
    nuc.step = lambda q, width=None: nuc.get_state()  # the arm stops following
    run(bridge, clock, 2.0, feed=lambda: bridge.set_target(position + [0.0, 0.0, -0.05], quaternion)
        if bridge.mode == 'target' else None)
    assert bridge.mode == 'hold' and any('tracking' in m for _, m in bridge.events)


def test_gripper_grasp_and_release():
    bridge, nuc, clock = make(object_width_m=0.03)
    bridge.gripper(0.03, 'grasp')
    run(bridge, clock, 1.0)
    assert nuc.steps[0][1] == 0.0  # every grasp is the hand contract's width-0 grasp
    state = bridge.gripper_state()
    assert state['is_grasped'] and not state['is_moving'] and abs(state['width'] - 0.03) < 1e-6
    bridge.gripper(0.0, 'move')  # a move is raised to the smallest move width, never a grasp
    assert bridge.width_cmd == pytest.approx(0.005)
    assert not bridge.gripper_state()['is_grasped']
    run(bridge, clock, 1.0)
    assert not bridge.gripper_state()['is_grasped'] and bridge.gripper_state()['width'] == pytest.approx(0.005)


def test_http_contract_with_rax_clients():
    nuc = MockNuc(START_Q)
    bridge = Bridge(nuc)
    servers = serve(bridge, arm_port=0, gripper_port=0)
    stop = threading.Event()
    loop = threading.Thread(target=bridge.run, args=(stop,), daemon=True)
    loop.start()
    try:
        arm = ArmBridge(f'http://127.0.0.1:{servers[0].server_port}')
        gripper = GripperBridge(f'http://127.0.0.1:{servers[1].server_port}')
        state = arm.state()
        assert np.allclose(state['arm_joint_position'], START_Q)
        assert np.allclose(state['arm_pos'], flange_pose(START_Q)[:3, 3])
        with pytest.raises(TargetRejected):
            arm.set_target([1.5, 0.0, 0.3], state['arm_quat'])
        goal = state['arm_pos'] + [0.0, 0.01, 0.0]
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and np.linalg.norm(arm.state()['arm_pos'] - goal) > 1e-3:
            arm.set_target(goal, state['arm_quat'])
            time.sleep(0.05)
        assert np.linalg.norm(arm.state()['arm_pos'] - goal) < 1e-3
        arm.hold()
        arm.move_joints(START_Q, 0.5)
        assert np.allclose(arm.state()['arm_joint_position'], START_Q, atol=1e-4)
        value = gripper.move(0.04, 0.1)
        assert value['width'] == pytest.approx(0.04) and not value['stale']
        assert set(gripper.state()) >= {'width', 'max_width', 'is_grasped', 'is_moving', 'stale'}
    finally:
        stop.set()
        loop.join(2)
        for server in servers:
            server.shutdown()
