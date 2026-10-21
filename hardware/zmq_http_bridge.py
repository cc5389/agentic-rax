"""RAX's arm (:50054) and gripper (:50053) HTTP bridges over the robopil NUC's ZMQ bridge.

The robopil FR3 is driven by wmrl's `franka_bridge_polymetis.py` on the NUC: a ZMQ REQ/REP server that takes
30 Hz absolute joint targets (`start_stream`, then `step{q, width}`) and runs Polymetis joint impedance at 1 kHz.
RAX's backend instead speaks the HTTP contract in README.md ("Bridge contract"): a flange (link8) pose target that
the bridge rate-limits and IK-checks, a hold, a timed joint move, and a 0.5 s command watchdog. This process is the
translation, and the NUC bridge's only client while it runs:

- one 30 Hz loop streams a commanded joint vector `q_cmd`: held, stepped toward the pose target, or interpolated
  along a joint move;
- a pose target moves the commanded flange at most `max_speed` m/s and `max_rotation` rad/s per tick, solved with
  RAX's own DLS IK (`services/franka/ik.py`) from `q_cmd`, so FK and IK are the model the backend validates with;
- a target with no IK solution, outside the lab joint limits, or with the TCP below `min_tcp_z` answers 409;
- no new target for `watchdog` seconds, a NUC error, or the measured joints falling `tracking_limit` rad behind:
  the arm holds;
- the hand is the NUC bridge's width command (gripper backend `droid_default`: below 4 mm the force-holding grasp,
  fixed at 5 N by the hand contract; any other width a move). A requested force is recorded, not applied.

Run from the repository root:  $RAX_PYTHON -m hardware.zmq_http_bridge [--mock] [--nuc tcp://host:port]
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import math
import os
import signal
import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from services.franka import ik
from services.franka.kinematics import flange_pose, link8_to_tcp, matrix

LOG = logging.getLogger('zmq_http_bridge')
# FR3 joint limits of the robopil site (Agentics-Robotics cfg/site.yaml), already inside the hardware limits.
LAB_LOWER = np.array([-2.62, -1.65, -2.77, -2.92, -2.67, 0.5745, -2.87])
LAB_UPPER = np.array([2.62, 1.65, 2.77, -0.19, 2.67, 4.37, 2.87])
MAX_WIDTH_M = 0.08
DROID_GRASP_BELOW_M = 0.004   # droid_default: a smaller width is the grasp
LEGACY_GRASP_BELOW_M = 0.045  # legacy (wmrl GripperWorker)
EMPTY_WIDTH_M = 0.002         # fingers closed beyond this hold nothing
HELD_STATE = {'droid_default': 'grasp', 'legacy': 'closed'}
MOVE_FLOOR_M = {'droid_default': DROID_GRASP_BELOW_M + 0.001, 'legacy': LEGACY_GRASP_BELOW_M}
HAND_LATENCY_S = 0.5          # command-to-motion onset of the hand (wmrl, measured 2026-10-17)


class Refused(Exception):
    """A target the bridge will not follow: HTTP 409."""


class NucError(RuntimeError):
    pass


class NucClient:
    """ZMQ REQ client of the NUC bridge (after Agentics-Robotics `robot/bridge.py` BridgeClient)."""

    def __init__(self, address, timeout_s=2.0):
        import zmq
        self.zmq, self.address, self.timeout_ms = zmq, address, int(timeout_s * 1000)
        self.context = zmq.Context()
        self.lock = threading.Lock()
        self._open()

    def _open(self):
        self.socket = self.context.socket(self.zmq.REQ)
        for option in ('RCVTIMEO', 'SNDTIMEO'):
            self.socket.setsockopt(getattr(self.zmq, option), self.timeout_ms)
        self.socket.setsockopt(self.zmq.LINGER, 0)
        self.socket.connect(self.address)

    def call(self, **request):
        with self.lock:
            try:
                self.socket.send(json.dumps(request).encode())
                reply = json.loads(self.socket.recv().decode())
            except self.zmq.ZMQError as exc:
                self.socket.close(linger=0)  # a timed-out REQ socket cannot be reused
                self._open()
                raise NucError(f'NUC bridge {request.get("cmd")} failed: {exc}') from exc
        if not reply.get('ok', True):
            raise NucError(f'NUC bridge {request.get("cmd")} refused: {reply.get("error")}')
        return reply

    def get_state(self):
        return self.call(cmd='get_state')

    def start_stream(self):
        return self.call(cmd='start_stream')

    def step(self, q, width=None):
        return self.call(cmd='step', q=[float(x) for x in q], width=None if width is None else float(width))

    def stop_stream(self):
        return self.call(cmd='stop_stream')

    def close(self):
        self.socket.close(linger=0)
        self.context.term()


class MockNuc:
    """The NUC bridge without a robot: joints follow each step within 0.10 rad (its clamp), droid_default hand."""

    def __init__(self, q=(0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398), *, object_width_m=None,
                 finger_speed_mps=0.1, clock=time.monotonic):
        self.q, self.width = np.asarray(q, float), MAX_WIDTH_M
        self.object_width_m, self.finger_speed, self.clock = object_width_m, finger_speed_mps, clock
        self.streaming, self.gripper_state = False, 'unknown'
        self._finger = None  # (target, kind, t)
        self.steps = []

    def _fingers(self):
        if self._finger is None:
            return
        target, kind, t = self._finger
        now = self.clock()
        travel, delta = self.finger_speed * (now - t), target - self.width
        if abs(delta) <= travel:
            self.width, self._finger, self.gripper_state = target, None, kind
        else:
            self.width += math.copysign(travel, delta)
            self._finger = (target, kind, now)

    def get_state(self):
        self._fingers()
        return {'ok': True, 'q': self.q.tolist(), 'width': self.width, 'streaming': self.streaming,
                'gripper_backend': 'droid_default', 'gripper_state': self.gripper_state,
                'gripper_busy': self._finger is not None, 'mock': True}

    def start_stream(self):
        self.streaming = True
        return {'ok': True}

    def step(self, q, width=None):
        if not self.streaming:
            raise NucError('NUC bridge step refused: not streaming')
        self.q = self.q + np.clip(np.asarray(q, float) - self.q, -0.10, 0.10)
        if width is not None:
            grasp = width < DROID_GRASP_BELOW_M
            target = (self.object_width_m or 0.0) if grasp else min(width, MAX_WIDTH_M)
            self._finger = (target, 'grasp' if grasp else 'width', self.clock())
        self.steps.append((np.asarray(q, float), width))
        return self.get_state()

    def stop_stream(self):
        self.streaming = False
        return {'ok': True}

    def close(self):
        pass


def _limit(vector, norm):
    size = float(np.linalg.norm(vector))
    return vector if size <= norm else vector * (norm / size)


def _smooth(s):
    """Minimum-jerk profile on [0, 1]."""
    s = min(max(s, 0.0), 1.0)
    return s ** 3 * (10 - 15 * s + 6 * s * s)


class Bridge:
    """The streaming state machine; `tick()` is one 30 Hz cycle (the loop thread or a test calls it)."""

    def __init__(self, nuc, *, hz=30.0, max_speed=0.05, max_rotation=0.5, max_joint_speed=1.0,
                 move_joint_speed=0.5, watchdog=0.5, min_tcp_z=-0.205, tracking_limit=0.25,
                 clock=time.monotonic):
        self.nuc, self.clock = nuc, clock
        self.dt = 1.0 / float(hz)
        self.max_speed, self.max_rotation = float(max_speed), float(max_rotation)
        self.max_joint_speed, self.move_joint_speed = float(max_joint_speed), float(move_joint_speed)
        self.watchdog, self.min_tcp_z, self.tracking_limit = float(watchdog), float(min_tcp_z), float(tracking_limit)
        self.lock = threading.RLock()
        self.raw = nuc.get_state()
        self.backend = self.raw.get('gripper_backend') or (
            'legacy' if self.raw.get('gripper_state') in ('open', 'closed') else 'droid_default')
        if self.backend not in HELD_STATE:
            raise NucError(f'unknown NUC gripper backend {self.backend!r}')
        self.streaming = bool(self.raw.get('streaming'))
        self.q_cmd = np.asarray(self.raw['q'], float)
        self.mode = 'hold'           # 'hold' | 'target' | 'joints'
        self.target = None           # link8 4x4 goal of the pose target
        self.target_at = 0.0
        self.path = None             # (q0, q1, t0, duration, done_event)
        self.width_cmd = None
        self.released = False
        self.moving_until = 0.0
        self.last_error, self.last_error_at, self.failures = None, -1e9, 0
        self.ticks = 0
        self.events = []             # (time, message): holds and faults, for the operator log

    # --- requests -----------------------------------------------------------------------------------------------
    def _note(self, message):
        self.events.append((self.clock(), message))
        del self.events[:-200]
        LOG.warning(message)

    def _ensure_stream(self):
        if not self.streaming:
            self.nuc.start_stream()  # joint impedance at the current pose
            self.raw = self.nuc.get_state()
            self.q_cmd = np.asarray(self.raw['q'], float)
            self.streaming = True

    def _check_q(self, q, what):
        q = np.asarray(q, float)
        if q.shape != (7,) or not np.all(np.isfinite(q)):
            raise Refused(f'{what}: expected 7 finite joint positions')
        low, high = q < LAB_LOWER, q > LAB_UPPER
        if low.any() or high.any():
            j = int(np.argmax(low | high))
            raise Refused(f'{what}: J{j + 1} = {q[j]:.3f} rad is outside the lab limit '
                          f'[{LAB_LOWER[j]:.3f}, {LAB_UPPER[j]:.3f}]')
        return q

    def set_target(self, position, quaternion):
        goal = matrix(np.asarray(position, float), np.asarray(quaternion, float))
        tcp_p, tcp_q = link8_to_tcp(goal[:3, 3], Rotation.from_matrix(goal[:3, :3]).as_quat())
        if tcp_p[2] < self.min_tcp_z:
            raise Refused(f'target TCP z {tcp_p[2]:.3f} m is below the bridge floor {self.min_tcp_z:.3f} m')
        with self.lock:
            if self.mode == 'joints':
                raise Refused('a joint move is running')
            q, converged = ik.solve(self.q_cmd, matrix(tcp_p, tcp_q))
            if not converged:
                raise Refused('no IK solution for the target from the current joints')
            self._check_q(q, 'IK solution')
            self._ensure_stream()
            self.target, self.target_at, self.mode = goal, self.clock(), 'target'
        return {'accepted': True}

    def hold(self):
        with self.lock:
            if self.mode == 'joints' and self.path is not None:
                self.path[4].set()
            if self.streaming and self.raw is not None:
                self.q_cmd = np.asarray(self.raw['q'], float)  # where the arm is, not where it was going
            self.mode, self.target, self.path = 'hold', None, None
        return {'held': True}

    def move_joints(self, positions, time_to_go):
        with self.lock:
            goal = self._check_q(positions, 'joint goal')
            self._ensure_stream()
            start = self.q_cmd.copy()
            duration = max(float(time_to_go), float(np.max(np.abs(goal - start))) / self.move_joint_speed, self.dt)
            done = threading.Event()
            self.path = (start, goal, self.clock(), duration, done)
            self.mode, self.target = 'joints', None
        if not done.wait(duration + 10.0):
            self.hold()
            raise NucError('joint move did not finish')
        if not getattr(done, 'completed', False):
            raise NucError(f'joint move interrupted: {self.events[-1][1] if self.events else "held"}')
        return {'reached': True, 'duration': round(duration, 3)}

    def gripper(self, width, kind):
        with self.lock:
            self._ensure_stream()
            if kind == 'grasp':
                self.width_cmd = 0.0
            else:
                self.width_cmd = max(min(float(width), MAX_WIDTH_M), MOVE_FLOOR_M[self.backend])
                self.released = True
            self.moving_until = self.clock() + HAND_LATENCY_S

    # --- the loop -----------------------------------------------------------------------------------------------
    def _next_q(self, now):
        if self.mode == 'joints':
            start, goal, t0, duration, done = self.path
            s = (now - t0) / duration
            q = start + _smooth(s) * (goal - start)
            if s >= 1.0:
                self.mode, self.path = 'hold', None
                done.completed = True
                done.set()
            return q
        if self.mode == 'target':
            if now - self.target_at > self.watchdog:
                self._note(f'watchdog: no target for {self.watchdog:.2f} s; holding')
                self.mode, self.target = 'hold', None
                return self.q_cmd
            current = flange_pose(self.q_cmd)
            step = np.eye(4)
            step[:3, 3] = current[:3, 3] + _limit(self.target[:3, 3] - current[:3, 3], self.max_speed * self.dt)
            turn = (Rotation.from_matrix(self.target[:3, :3]) * Rotation.from_matrix(current[:3, :3]).inv()).as_rotvec()
            step[:3, :3] = (Rotation.from_rotvec(_limit(turn, self.max_rotation * self.dt))
                            * Rotation.from_matrix(current[:3, :3])).as_matrix()
            tcp_p, tcp_q = link8_to_tcp(step[:3, 3], Rotation.from_matrix(step[:3, :3]).as_quat())
            q, converged = ik.solve(self.q_cmd, matrix(tcp_p, tcp_q), iterations=30)
            try:
                if not converged:
                    raise Refused('IK lost the path')
                self._check_q(q, 'IK step')
            except Refused as exc:
                self._note(f'{exc}; holding')
                self.mode, self.target = 'hold', None
                return self.q_cmd
            return self.q_cmd + np.clip(q - self.q_cmd, -self.max_joint_speed * self.dt, self.max_joint_speed * self.dt)
        return self.q_cmd

    def tick(self):
        now = self.clock()
        with self.lock:
            if not self.streaming:
                return
            self.q_cmd = self._next_q(now)
            width, self.width_cmd = self.width_cmd, None
            q = self.q_cmd.copy()
        try:
            reply = self.nuc.step(q, width)
        except Exception as exc:
            with self.lock:
                self.failures += 1
                self.last_error, self.last_error_at = f'{type(exc).__name__}: {exc}', now
                if self.mode != 'hold':
                    self._note(f'NUC step failed ({exc}); holding')
                    self.hold()
                if width is not None:
                    self.width_cmd = width  # resend with the next step
            return
        with self.lock:
            self.raw, self.failures, self.ticks = reply, 0, self.ticks + 1
            if reply.get('gripper_state') != HELD_STATE[self.backend]:
                self.released = False
            lag = float(np.max(np.abs(np.asarray(reply['q'], float) - self.q_cmd)))
            if lag > self.tracking_limit and self.mode != 'hold':
                self._note(f'tracking: joints {lag:.3f} rad behind the command; holding')
                self.hold()

    def run(self, stop):
        next_at = self.clock()
        while not stop.is_set():
            self.tick()
            next_at += self.dt
            delay = next_at - self.clock()
            if delay > 0:
                stop.wait(delay)
            else:
                next_at = self.clock()

    def refresh(self):
        """The NUC state: the stream's last reply while it streams, else a fresh request."""
        if self.streaming:
            return self.raw
        try:
            raw = self.nuc.get_state()
        except Exception as exc:
            self.last_error, self.last_error_at = f'{type(exc).__name__}: {exc}', self.clock()
            raise
        with self.lock:
            self.raw = raw
        return raw

    # --- answers ------------------------------------------------------------------------------------------------
    def arm_state(self):
        raw = self.refresh()
        q = np.asarray(raw['q'], float)
        F = flange_pose(q)
        quat = Rotation.from_matrix(F[:3, :3]).as_quat()
        with self.lock:
            return {'arm_pos': F[:3, 3].tolist(), 'arm_quat': quat.tolist(), 'arm_joint_position': q.tolist(),
                    'policy_running': bool(self.streaming), 'mode': self.mode,
                    'last_event': self.events[-1][1] if self.events else None}

    def gripper_state(self):
        stale = self.clock() - self.last_error_at < 2.0
        try:
            raw = self.refresh()
        except Exception:
            raw, stale = self.raw or {}, True
        width = float(raw.get('width', 0.0))
        with self.lock:
            held = (raw.get('gripper_state') == HELD_STATE[self.backend] and width > EMPTY_WIDTH_M
                    and not self.released)
            moving = bool(raw.get('gripper_busy')) or self.clock() < self.moving_until or self.width_cmd is not None
        return {'width': width, 'max_width': MAX_WIDTH_M, 'is_grasped': bool(held), 'is_moving': moving,
                'stale': bool(stale), 'gripper_state': raw.get('gripper_state'), 'backend': self.backend}

    def wait_gripper(self, timeout=10.0):
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            state = self.gripper_state()
            if not state['is_moving']:
                return state
            time.sleep(0.03)
        return self.gripper_state()

    def shutdown(self):
        with self.lock:
            self.hold()
            streaming, self.streaming = self.streaming, False
        if streaming:
            try:
                self.nuc.stop_stream()
            except Exception as exc:
                LOG.warning('stop_stream failed: %s', exc)


def handler(bridge, routes):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            LOG.debug(fmt, *args)

        def _answer(self, code, value):
            body = json.dumps(value).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _dispatch(self, method):
            route = routes.get((method, self.path.split('?')[0]))
            if route is None:
                return self._answer(404, {'error': f'no route {method} {self.path}'})
            try:
                length = int(self.headers.get('Content-Length') or 0)
                body = json.loads(self.rfile.read(length) or b'{}') if method == 'POST' else {}
                self._answer(200, route(body))
            except Refused as exc:
                self._answer(409, {'refused': str(exc)})
            except Exception as exc:
                LOG.exception('%s %s failed', method, self.path)
                self._answer(500, {'error': f'{type(exc).__name__}: {exc}'})

        def do_GET(self):
            self._dispatch('GET')

        def do_POST(self):
            self._dispatch('POST')
    return Handler


def gripper_command(bridge, kind):
    def command(body):
        bridge.gripper(body.get('width', 0.0), kind)
        if kind == 'grasp' or body.get('force') is not None:
            LOG.info('%s width=%s speed=%s force=%s (the hand contract fixes the grasp force)', kind,
                     body.get('width'), body.get('speed'), body.get('force'))
        return bridge.wait_gripper() if body.get('blocking', True) else bridge.gripper_state()
    return command


def serve(bridge, host='127.0.0.1', arm_port=50054, gripper_port=50053):
    arm_routes = {('GET', '/state'): lambda body: bridge.arm_state(),
                  ('POST', '/target'): lambda body: bridge.set_target(body['position'], body['quaternion']),
                  ('POST', '/hold'): lambda body: bridge.hold(),
                  ('POST', '/move_joints'): lambda body: bridge.move_joints(body['positions'], body['time_to_go'])}
    gripper_routes = {('GET', '/state'): lambda body: bridge.gripper_state(),
                      ('POST', '/move'): gripper_command(bridge, 'move'),
                      ('POST', '/grasp'): gripper_command(bridge, 'grasp')}
    servers = [ThreadingHTTPServer((host, arm_port), handler(bridge, arm_routes)),
               ThreadingHTTPServer((host, gripper_port), handler(bridge, gripper_routes))]
    for server in servers:
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name=f'http-{server.server_port}', daemon=True).start()
    return servers


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--nuc', default=os.environ.get('RAX_NUC_BRIDGE'),
                        help='the NUC ZMQ bridge address, tcp://host:port (default $RAX_NUC_BRIDGE)')
    parser.add_argument('--mock', action='store_true', help='simulate the NUC bridge (no robot)')
    parser.add_argument('--arm-port', type=int, default=50054)
    parser.add_argument('--gripper-port', type=int, default=50053)
    parser.add_argument('--hz', type=float, default=30.0)
    parser.add_argument('--max-speed', type=float, default=0.05, help='flange speed limit, m/s (RAX contract 0.05)')
    parser.add_argument('--max-rotation', type=float, default=0.5, help='flange rotation limit, rad/s')
    parser.add_argument('--max-joint-speed', type=float, default=1.0, help='joint speed cap while tracking, rad/s')
    parser.add_argument('--move-joint-speed', type=float, default=0.5, help='joint speed cap of /move_joints, rad/s')
    parser.add_argument('--watchdog', type=float, default=0.5, help='hold after this long without a target, s')
    parser.add_argument('--min-tcp-z', type=float, default=-0.205, help='lowest TCP z of a target, m (base frame)')
    parser.add_argument('--tracking-limit', type=float, default=0.25, help='hold when joints lag this much, rad')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args(argv)
    if not args.mock and not args.nuc:
        parser.error('give --nuc tcp://host:port or set RAX_NUC_BRIDGE (or use --mock)')
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')
    nuc = MockNuc() if args.mock else NucClient(args.nuc)
    bridge = Bridge(nuc, hz=args.hz, max_speed=args.max_speed, max_rotation=args.max_rotation,
                    max_joint_speed=args.max_joint_speed, move_joint_speed=args.move_joint_speed,
                    watchdog=args.watchdog, min_tcp_z=args.min_tcp_z, tracking_limit=args.tracking_limit)
    servers = serve(bridge, arm_port=args.arm_port, gripper_port=args.gripper_port)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    LOG.info('bridging %s (gripper backend %s) on 127.0.0.1:%d (arm) and :%d (gripper); speed %.3f m/s',
             'a mock NUC' if args.mock else args.nuc, bridge.backend, args.arm_port, args.gripper_port, args.max_speed)
    try:
        bridge.run(stop)
    finally:
        bridge.shutdown()
        for server in servers:
            server.shutdown()
        nuc.close()


if __name__ == '__main__':
    main()
