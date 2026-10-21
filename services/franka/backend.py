"""MCP backend for the real Franka FR3 (Polymetis bridges + ZED cameras).

Safety layering, outermost first:
  1. The physical emergency stop and the operator beside the robot.
  2. Operator arming (services.franka.arming): motion tools refuse unless armed, and a
     running motion holds within one servo period once the operator disarms.
  3. This backend: a TCP workspace box (checked on the goal and on every servo tick),
     joint moves checked by FK along the whole interpolated path, step limits.
  4. The Polymetis bridge: rate limits (50 mm/s, 0.5 rad/s), IK and joint-limit checks,
     a 0.5 s command watchdog that holds the arm.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import uuid

import cv2 as cv
import numpy as np
from scipy.spatial.transform import Rotation

from . import arming
from .bridges import ArmBridge, GripperBridge, TargetRejected
from .calibration import CameraModel, backproject, camera_specs, project as project_to_pixel
from .ik import check_line
from .kinematics import LOWER, UPPER, joint_frames, link8_to_tcp, matrix
from .servo import MotionStopped, PoseServo, ServoSettings

PROJECT = Path(__file__).resolve().parents[2]
WORKSPACE_NAME = 'agent-workspace'
HOME_Q = (0.0, -0.785398, 0.0, -2.356194, 0.0, 1.570796, 0.785398)  # libfranka "ready"
DEFAULTS = {
    'arm_url': 'http://127.0.0.1:50054',
    'gripper_url': 'http://127.0.0.1:50053',
    # TCP workspace box in the base frame (metres). The table top is at z ~= -0.21 (robopil rig).
    'workspace_min': [0.25, -0.45, 0.10],
    'workspace_max': [0.80, 0.45, 0.70],
    'workspace_guard_margin_m': 0.02,    # measured TCP may overshoot the box this much
    'link_min_z': 0.12,                  # joints 4..7 and the flange stay above this
    'max_translation_m': 0.40,           # per franka_move_pose call
    'max_joint_step_rad': 1.5,           # per franka_move_joints call, any joint
    'joint_speed_radps': 0.4,
    'joint_limit_margin_rad': 0.2,
    # Extra joint-limit margin for the IK pre-check: the bridge checks its own IK solution
    # of goal + servo bias, which differs from ours near the limits.
    'ik_extra_margin_rad': 0.05,
    'move_timeout_s': 30.0,
    'gripper_speed_mps': 0.05,
    'gripper_max_force_n': 60.0,
    'cameras': None,                     # required: the ZED cameras (services.franka.calibration)
    'camera_cpus': '0-15',               # CPUs the camera server is pinned to (taskset -c)
    'camera_depth_mode': 'NEURAL',
    # Sliding-window recording of everything the cameras and the arm do (read-only to the agent).
    'stream_enabled': True,
    'stream_fps': 30,
    'stream_window_s': 600,
    'stream_depth_hz': 1.0,
    'stream_jpeg_quality': 80,
    'robot_log_hz': 10.0,
    'stream_archive': True,              # whole-session MP4 per camera per minute (<session>/archive)
    'agent_image_width': 640,
}
# Session options the launcher reads (harness.claude_cli.launcher), accepted here too.
LAUNCHER_OPTIONS = ('agent_python', 'protected_paths')


def _schema(properties=None, required=()):
    return {'type': 'object', 'properties': properties or {}, 'required': list(required),
            'additionalProperties': False}


NUMBER = {'type': 'number'}
VEC3 = {'type': 'array', 'items': NUMBER, 'minItems': 3, 'maxItems': 3}
QUAT = {'type': 'array', 'items': NUMBER, 'minItems': 4, 'maxItems': 4,
        'description': 'TCP orientation quaternion [x, y, z, w] in the base frame.'}


class CameraService:
    """Client for services.franka.camera_server, started lazily under taskset."""

    def __init__(self, options, frames_root, log_path, depth_mode, stream=None):
        self.options, self.frames_root, self.log_path = options, Path(frames_root), Path(log_path)
        self.stream = stream
        self.depth_mode = depth_mode
        directory = Path(f'/tmp/rax-cam-{os.getuid()}')  # hidden from the agent sandbox
        directory.mkdir(mode=0o700, exist_ok=True)
        self.socket_path = directory/(hashlib.sha256(str(frames_root).encode()).hexdigest()[:16] + '.sock')
        self.process = None
        self.lock = threading.Lock()

    def _start(self):
        specs = [{k: v for k, v in spec.items() if k != 'extrinsics'} for spec in self.options['cameras']]
        caps = {name: '8' for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                                       'NUMEXPR_NUM_THREADS')}
        env = {**os.environ, **caps, 'OMP_WAIT_POLICY': 'PASSIVE', 'PYTHONPATH': str(PROJECT)}
        command = ['taskset', '-c', str(self.options['camera_cpus']), sys.executable, '-m',
                   'services.franka.camera_server', '--socket', str(self.socket_path),
                   '--cameras', json.dumps(specs), '--depth-mode', self.depth_mode,
                   '--frames-root', str(self.frames_root), '--fps', str(self.options['stream_fps'])]
        if self.stream is not None:
            command += ['--stream', str(self.stream), '--window-seconds', str(self.options['stream_window_s']),
                        '--depth-hz', str(self.options['stream_depth_hz']),
                        '--stream-quality', str(self.options['stream_jpeg_quality'])]
            if self.options['stream_archive']:
                command += ['--archive', str(self.stream.parent/'archive')]
        log = open(self.log_path, 'ab')
        self.socket_path.unlink(missing_ok=True)
        self.process = subprocess.Popen(command, cwd=PROJECT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        deadline = time.monotonic() + 60
        while not self.socket_path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(f'camera server exited ({self.process.returncode}); see camera-server.log')
            if time.monotonic() > deadline:
                raise RuntimeError('camera server did not start')
            time.sleep(0.1)

    def start(self):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start()

    def request(self, payload, timeout=90.0):
        with self.lock:
            if self.process is None or self.process.poll() is not None:
                self._start()
            with socket.socket(socket.AF_UNIX) as connection:
                connection.settimeout(timeout)
                connection.connect(str(self.socket_path))
                connection.sendall((json.dumps(payload) + '\n').encode())
                line = connection.makefile('r').readline()
        if not line:
            raise RuntimeError('camera server closed the connection')
        value = json.loads(line)
        if not value.pop('ok', False):
            raise RuntimeError(value.get('error', 'camera server error'))
        return value

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(25)  # the camera server archives the minute in progress
            except subprocess.TimeoutExpired:
                self.process.kill()
        self.process = None


class FrankaBackend:
    mutating = frozenset({'franka_move_pose', 'franka_move_joints', 'franka_gripper'})

    def __init__(self, config, *, arm=None, gripper=None, cameras=None, servo=None):
        self.config = config
        unknown = (set(config.backend_options) - set(DEFAULTS) - set(ServoSettings.__dataclass_fields__)
                   - set(LAUNCHER_OPTIONS))
        if unknown:
            raise ValueError(f'unknown franka backend options: {sorted(unknown)}')
        self.options = {**DEFAULTS, **config.backend_options}
        self.box_min = np.asarray(self.options['workspace_min'], dtype=np.float64)
        self.box_max = np.asarray(self.options['workspace_max'], dtype=np.float64)
        if not np.all(self.box_min < self.box_max):
            raise ValueError('workspace_min must be below workspace_max on every axis')
        self.arm = arm or ArmBridge(self.options['arm_url'])
        self.gripper = gripper or GripperBridge(self.options['gripper_url'])
        self.servo = servo or PoseServo(self.arm, ServoSettings.from_options(config.backend_options))
        self.models = {spec['name']: CameraModel.from_spec(spec) for spec in camera_specs(self.options)}
        self.workspace = config.root/WORKSPACE_NAME
        self.frames_root = self.workspace/'runtime'/'frames'
        self.depth = config.observation_profile == 'rgbd'
        self.stream = config.root/'stream' if self.options['stream_enabled'] else None
        self.last_motion = None
        self._stop = threading.Event()
        if cameras is None:
            self.cameras = CameraService(self.options, self.frames_root, config.root/'camera-server.log',
                                         self.options['camera_depth_mode'] if self.depth else 'NONE',
                                         stream=self.stream)
            if self.stream is not None:  # record from the start of the session, not the first observe
                self.stream.mkdir(mode=0o755, exist_ok=True)
                self.cameras.start()
                threading.Thread(target=self._log_robot, name='robot-log', daemon=True).start()
        else:
            self.cameras = cameras

    def _log_robot(self):
        """Arm and gripper state into <stream>/robot/<UTC minute>.jsonl (pruned with the frames)."""
        from .camera_server import minute_name
        folder = self.stream/'robot'
        folder.mkdir(exist_ok=True)
        period = 1.0 / float(self.options['robot_log_hz'])
        width = None
        tick = 0
        while not self._stop.wait(period):
            now = time.time()
            try:
                state = self.arm.state()
                if tick % max(1, int(self.options['robot_log_hz'] / 2)) == 0:  # gripper at ~2 Hz
                    width = self.gripper.state().get('width')
                tick += 1
                tcp_p, tcp_q = link8_to_tcp(state['arm_pos'], state['arm_quat'])
                record = {'t_ms': int(now * 1000), 'tcp_position': np.round(tcp_p, 5).tolist(),
                          'tcp_quaternion': np.round(tcp_q, 5).tolist(),
                          'joints': np.round(np.asarray(state['arm_joint_position']), 4).tolist(),
                          'gripper_width': None if width is None else round(float(width), 4)}
            except Exception as exc:
                record = {'t_ms': int(now * 1000), 'error': str(exc)[:200]}
            with open(folder/f'{minute_name(now)}.jsonl', 'a') as log:
                log.write(json.dumps(record) + '\n')

    # ----- tool definitions -------------------------------------------------------------
    def tools(self):
        names = sorted(self.models)
        lo, hi = self.box_min.round(3).tolist(), self.box_max.round(3).tolist()
        return [
            {'name': 'franka_status', 'inputSchema': _schema(),
             'description': 'Robot state without moving: TCP pose (fr3_hand_tcp, base frame), joints, gripper '
                            'width, whether the operator has armed motion, the workspace box and limits.'},
            {'name': 'franka_observe', 'description':
                f'Capture the cameras ({", ".join(names)}) now. Returns downscaled images plus the TCP pose; '
                'full-resolution RGB (and depth as uint16 millimetre PNG in rgbd sessions) and a meta.json '
                'with intrinsics and camera poses are saved under runtime/frames/<frame_id>/. Use frame_id '
                'with franka_pixel_to_position. project: base-frame points to mark on the images.',
             'inputSchema': _schema({'cameras': {'type': 'array', 'items': {'type': 'string', 'enum': names}},
                                     'project': {'type': 'array', 'items': VEC3, 'maxItems': 16}})},
            {'name': 'franka_pixel_to_position', 'description':
                'Base-frame 3-D points for pixels of a saved frame, from its depth. Pixels are in the '
                'full-resolution image (width x height in meta.json), not the downscaled preview. Uses the '
                'median valid depth in a (2*radius+1)^2 window.',
             'inputSchema': _schema({'frame_id': {'type': 'string'}, 'camera': {'type': 'string', 'enum': names},
                                     'pixels': {'type': 'array', 'minItems': 1, 'maxItems': 64,
                                                'items': {'type': 'array', 'items': NUMBER, 'minItems': 2,
                                                          'maxItems': 2}},
                                     'radius': {'type': 'integer', 'minimum': 0, 'maximum': 15}},
                                    ['frame_id', 'camera', 'pixels'])},
            {'name': 'franka_move_pose', 'description':
                'Move the TCP (fr3_hand_tcp, between the fingertips) to a base-frame pose in a straight line '
                f'and wait until it settles. Workspace box {lo} .. {hi} m; at most '
                f"{self.options['max_translation_m']} m per call; speed is capped at 50 mm/s. Omit "
                'quaternion to keep the current orientation. relative=true adds position to the current TCP '
                'position. Requires operator arming. Returns the measured pose and error.',
             'inputSchema': _schema({'position': VEC3, 'quaternion': QUAT, 'relative': {'type': 'boolean'},
                                     'tolerance_mm': {'type': 'number', 'minimum': 0.5, 'maximum': 20},
                                     'timeout_s': {'type': 'number', 'minimum': 1, 'maximum': 60}},
                                    ['position'])},
            {'name': 'franka_move_joints', 'description':
                'Joint-space move along a straight joint path (Polymetis min-jerk). positions: 7 joint angles '
                '(rad); or preset "home" (libfranka ready pose). The whole path is checked by FK against the '
                'workspace box and table clearance before moving. Requires operator arming.',
             'inputSchema': _schema({'positions': {'type': 'array', 'items': NUMBER, 'minItems': 7,
                                                   'maxItems': 7},
                                     'preset': {'type': 'string', 'enum': ['home']}})},
            {'name': 'franka_gripper', 'description':
                'Franka Hand. open; move to width (m, 0..0.08, position only: does not hold an object); '
                'grasp closes to width with force (N) and keeps squeezing (use this to hold objects). '
                'Requires operator arming. Returns width and is_grasped.',
             'inputSchema': _schema({'action': {'type': 'string', 'enum': ['open', 'move', 'grasp']},
                                     'width': {'type': 'number', 'minimum': 0, 'maximum': 0.081},
                                     'force': {'type': 'number', 'minimum': 5,
                                               'maximum': self.options['gripper_max_force_n']},
                                     'speed': {'type': 'number', 'minimum': 0.005, 'maximum': 0.1}},
                                    ['action'])},
            {'name': 'franka_stream', 'description':
                'Look back in the continuous recording (every camera at '
                f"{self.options['stream_fps']} Hz and the arm at {self.options['robot_log_hz']:g} Hz, last "
                f"{self.options['stream_window_s'] / 60:g} minutes) without moving anything. seconds_ago: frames "
                'from that many seconds ago (0 = newest), labelled with their age, plus the arm state then. '
                'sheet: one contact sheet per camera of count evenly spaced frames over the last seconds. Use '
                'it to see what changed while you were thinking and to review a motion or script run. Files: '
                'runtime/stream/.',
             'inputSchema': _schema({'cameras': {'type': 'array', 'items': {'type': 'string', 'enum': names}},
                                     'seconds_ago': {'type': 'array', 'items': {'type': 'number', 'minimum': 0},
                                                     'maxItems': 8},
                                     'sheet': _schema({'seconds': {'type': 'number', 'minimum': 1},
                                                       'count': {'type': 'integer', 'minimum': 2, 'maximum': 16}},
                                                      ['seconds'])})},
            {'name': 'franka_hold', 'inputSchema': _schema(),
             'description': 'Hold the current pose (stop tracking any previous target). Always allowed.'},
        ]

    # ----- validation (before any motion) -------------------------------------------------
    def _require_armed(self):
        state = arming.read(self.config.root)
        if not state['armed']:
            raise PermissionError(f"motion is not armed ({state['reason']}); the operator arms motion")

    def _check_box(self, point, margin=0.0, what='target'):
        point = np.asarray(point, dtype=np.float64)
        if np.any(point < self.box_min - margin) or np.any(point > self.box_max + margin):
            raise ValueError(f'{what} {np.round(point, 3).tolist()} is outside the workspace box '
                             f'{self.box_min.round(3).tolist()} .. {self.box_max.round(3).tolist()}')

    def _pose_goal(self, arguments, state):
        tcp_p, tcp_q = link8_to_tcp(state['arm_pos'], state['arm_quat'])
        position = np.asarray(arguments['position'], dtype=np.float64)
        if not np.all(np.isfinite(position)):
            raise ValueError('position must be finite')
        if arguments.get('relative'):
            position = tcp_p + position
        quaternion = arguments.get('quaternion')
        if quaternion is None:
            quaternion = tcp_q
        quaternion = np.asarray(quaternion, dtype=np.float64)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 0.5:
            raise ValueError('quaternion must be a unit quaternion [x, y, z, w]')
        quaternion = quaternion/norm
        self._check_box(position)
        step = float(np.linalg.norm(position - tcp_p))
        if step > self.options['max_translation_m']:
            raise ValueError(f"move of {step:.3f} m exceeds {self.options['max_translation_m']} m per call; "
                             'split it into shorter moves')
        problem = check_line(state['arm_joint_position'], matrix(tcp_p, tcp_q), matrix(position, quaternion),
                             margin=self.options['joint_limit_margin_rad'] + self.options['ik_extra_margin_rad'])
        if problem:
            raise ValueError(f'unreachable for the controller: {problem}. Choose a pose further from that '
                             'limit (for the elbow near the base: higher, or further out in x), change the '
                             'orientation, or go home first')
        return position, quaternion

    def _joint_goal(self, arguments):
        if ('positions' in arguments) == ('preset' in arguments):
            raise ValueError('give exactly one of positions or preset')
        goal = np.asarray(HOME_Q if arguments.get('preset') == 'home' else arguments['positions'], dtype=np.float64)
        margin = self.options['joint_limit_margin_rad']
        if not np.all(np.isfinite(goal)) or np.any(goal < LOWER + margin) or np.any(goal > UPPER - margin):
            raise ValueError(f'joint target outside the joint limits minus {margin} rad')
        return goal

    def _check_joint_path(self, start, goal):
        step = float(np.max(np.abs(goal - start)))
        if step > self.options['max_joint_step_rad']:
            raise ValueError(f"joint move of {step:.2f} rad exceeds {self.options['max_joint_step_rad']} rad "
                             'per call; split it')
        for fraction in np.linspace(0.0, 1.0, 41):
            frames = joint_frames(start + fraction * (goal - start))
            self._check_box(frames[8][:3, 3], margin=self.options['workspace_guard_margin_m'],
                            what=f'TCP at {fraction:.0%} of the joint path')
            low = min(frame[2, 3] for frame in frames[3:8])
            if low < self.options['link_min_z']:
                raise ValueError(f"joint path brings a link to z={low:.3f} m at {fraction:.0%} "
                                 f"(minimum {self.options['link_min_z']} m)")
        return step

    def validate(self, name, arguments):
        if name == 'franka_move_pose':
            self._pose_goal(arguments, self.arm.state())
        elif name == 'franka_move_joints':
            self._check_joint_path(np.asarray(self.arm.state()['arm_joint_position']), self._joint_goal(arguments))
        elif name == 'franka_gripper':
            action = arguments['action']
            if action in ('move', 'grasp') and 'width' not in arguments:
                raise ValueError(f'{action} needs width')
            if action != 'grasp' and 'force' in arguments:
                raise ValueError('force applies to grasp only')
        elif name == 'franka_pixel_to_position':
            self._frame_dir(arguments['frame_id'])
        if name in self.mutating:
            self._require_armed()

    # ----- calls --------------------------------------------------------------------------
    def call(self, name, arguments):
        handler = getattr(self, '_' + name.removeprefix('franka_'), None)
        if handler is None or name not in {tool['name'] for tool in self.tools()}:
            raise ValueError(f'unknown tool {name}')
        return handler(**arguments)

    def _status(self):
        return self.status(), []

    def _hold(self):
        self.arm.hold()
        return {'held': True, **self._pose_report(self.arm.state())}, []

    def _guard(self, state):
        if not arming.read(self.config.root)['armed']:
            raise MotionStopped('the operator disarmed motion; the arm holds')
        tcp_p, _ = link8_to_tcp(state['arm_pos'], state['arm_quat'])
        self._check_box(tcp_p, margin=self.options['workspace_guard_margin_m'], what='measured TCP')

    def _move_pose(self, position, quaternion=None, relative=False, tolerance_mm=1.5, timeout_s=None):
        goal_p, goal_q = self._pose_goal({'position': position, 'quaternion': quaternion, 'relative': relative},
                                         self.arm.state())
        self._require_armed()
        timeout = min(float(timeout_s or self.options['move_timeout_s']), 60.0)
        try:
            result = self.servo.move_tcp(goal_p, goal_q, tolerance_m=tolerance_mm/1000.0,
                                         tolerance_rad=np.radians(max(0.5, tolerance_mm/3.0)),
                                         timeout_s=timeout, guard=self._guard)
        except ValueError as exc:  # measured TCP left the box: stop right there
            self.arm.hold()
            raise self._held(MotionStopped(f'{exc}; the arm holds'), goal_p) from exc
        except TargetRejected as exc:
            self.arm.hold()
            raise self._held(TargetRejected(f'the controller refused the next step towards this target (treat '
                                            f'it as a wall): {exc}; the arm holds'), goal_p) from exc
        except (MotionStopped, TimeoutError) as exc:
            raise self._held(exc, goal_p)
        self.last_motion = {'tool': 'franka_move_pose', 'at': time.time(), **result}
        return {**result, 'target_position': np.round(goal_p, 5).tolist(),
                'target_quaternion': np.round(goal_q, 5).tolist()}, []

    def _held(self, exc, goal_p):
        """Mark a handled stop: the arm holds at a measured pose, nothing else is pending."""
        report = self._pose_report(self.arm.state())
        report['distance_to_target_mm'] = round(float(np.linalg.norm(np.asarray(report['tcp_position']) - goal_p))
                                                * 1000, 1)
        exc.control_certain = True
        exc.robot_state = report
        return exc

    def _move_joints(self, positions=None, preset=None):
        arguments = {'positions': positions} if positions is not None else {'preset': preset}
        goal = self._joint_goal(arguments)
        start = np.asarray(self.arm.state()['arm_joint_position'])
        step = self._check_joint_path(start, goal)
        self._require_armed()
        duration = max(2.0, step / self.options['joint_speed_radps'])
        started = time.monotonic()
        self.arm.move_joints(goal, duration)
        state = self.arm.state()
        error = np.abs(np.asarray(state['arm_joint_position']) - goal)
        result = {'seconds': round(time.monotonic() - started, 2), 'max_joint_error_rad': round(float(error.max()), 4),
                  **self._pose_report(state)}
        self.last_motion = {'tool': 'franka_move_joints', 'at': time.time(), **result}
        return result, []

    def _gripper(self, action, width=None, force=None, speed=None):
        self._require_armed()
        speed = float(speed or self.options['gripper_speed_mps'])
        state = self.gripper.state()
        if state.get('stale'):
            raise RuntimeError('gripper state is stale: the Polymetis gripper client is dead; tell the operator')
        deadline = time.monotonic() + 5
        while self.gripper.state().get('is_moving') and time.monotonic() < deadline:
            time.sleep(0.05)  # commanding a moving hand kills franka_hand_client
        if action == 'open':
            value = self.gripper.move(state['max_width'], speed)
        elif action == 'move':
            value = self.gripper.move(width, speed)
        else:
            value = self.gripper.grasp(width, speed, float(force or 20.0))
            time.sleep(0.2)
            value = self.gripper.state()
        if value.get('skipped'):
            raise RuntimeError('the gripper was still moving; command skipped, try again')
        return {'action': action, 'width': round(float(value['width']), 4),
                'is_grasped': bool(value.get('is_grasped', False))}, []

    # ----- observation --------------------------------------------------------------------
    def _frame_dir(self, frame_id):
        if not isinstance(frame_id, str) or not frame_id.isalnum():
            raise ValueError('invalid frame_id')
        path = self.frames_root/frame_id
        if not (path/'meta.json').is_file():
            raise ValueError(f'unknown frame_id {frame_id}')
        return path

    def _observe(self, cameras=None, project=None):
        names = list(cameras or sorted(self.models))
        frame_id = time.strftime('%Y%m%d%H%M%S') + uuid.uuid4().hex[:6]
        out_dir = self.frames_root/frame_id
        before = self.arm.state()
        captured = self.cameras.request({'op': 'grab', 'cameras': names, 'depth': self.depth,
                                         'out_dir': str(out_dir)})['cameras']
        after = self.arm.state()
        tcp_p, tcp_q = link8_to_tcp(after['arm_pos'], after['arm_quat'])
        moved = float(np.linalg.norm(after['arm_pos'] - before['arm_pos']))
        T_base_tcp = matrix(tcp_p, tcp_q)
        meta = {'frame_id': frame_id, 'time': time.time(), 'tcp_position': tcp_p.tolist(),
                'tcp_quaternion': tcp_q.tolist(), 'joints': np.asarray(after['arm_joint_position']).tolist(),
                'arm_moved_during_capture_mm': round(moved * 1000, 2), 'depth_units': 'mm (uint16, 0 = invalid)',
                'cameras': {}}
        images, report = [], {}
        for name in names:
            entry = captured[name]
            T_base_cam = self.models[name].base_from_camera(T_base_tcp)
            meta['cameras'][name] = {'rgb': Path(entry['rgb']).name,
                                     'depth': Path(entry['depth']).name if entry['depth'] else None,
                                     'K': entry['K'], 'width': entry['width'], 'height': entry['height'],
                                     'T_base_cam': T_base_cam.tolist(), 'timestamp_ns': entry['timestamp_ns']}
            image = cv.imread(entry['rgb'])
            marks = []
            for point in project or []:
                uv = project_to_pixel(point, T_base_cam, entry['K'])
                marks.append(None if uv is None else [round(uv[0], 1), round(uv[1], 1)])
                if uv is not None:
                    cv.drawMarker(image, (int(round(uv[0])), int(round(uv[1]))), (0, 0, 255), cv.MARKER_CROSS, 24, 2)
            scale = self.options['agent_image_width'] / image.shape[1]
            small = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA)
            ok, jpeg = cv.imencode('.jpg', small, [cv.IMWRITE_JPEG_QUALITY, 85])
            images.append({'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(jpeg).decode()})
            report[name] = {'preview_scale': round(scale, 4), 'width': entry['width'], 'height': entry['height'],
                            'captured_ms': entry['timestamp_ns'] // 1_000_000,
                            **({'projected_pixels': marks} if project else {})}
        (out_dir/'meta.json').write_text(json.dumps(meta, indent=1))
        return {'frame_id': frame_id, 'directory': f'runtime/frames/{frame_id}', 'images_in_order': names,
                'cameras': report, 'arm_moved_during_capture_mm': meta['arm_moved_during_capture_mm'],
                **self._pose_report(after)}, images

    def _pixel_to_position(self, frame_id, camera, pixels, radius=3):
        directory = self._frame_dir(frame_id)
        meta = json.loads((directory/'meta.json').read_text())
        entry = meta['cameras'].get(camera)
        if entry is None or not entry['depth']:
            raise ValueError(f'frame {frame_id} has no depth for camera {camera}')
        depth = cv.imread(str(directory/entry['depth']), cv.IMREAD_UNCHANGED)
        T_base_cam = np.asarray(entry['T_base_cam'])
        points = []
        for u, v in pixels:
            if not (0 <= u < entry['width'] and 0 <= v < entry['height']):
                points.append({'pixel': [u, v], 'error': 'outside the image'})
                continue
            r = int(radius)
            ui, vi = int(round(u)), int(round(v))
            window = depth[max(0, vi - r):vi + r + 1, max(0, ui - r):ui + r + 1]
            valid = window[window > 0]
            if valid.size == 0:
                points.append({'pixel': [u, v], 'error': 'no valid depth here'})
                continue
            z = float(np.median(valid)) / 1000.0
            p_cam = backproject(u, v, z, entry['K'])
            p_base = (T_base_cam @ np.append(p_cam, 1.0))[:3]
            points.append({'pixel': [u, v], 'position': np.round(p_base, 4).tolist(), 'depth_m': round(z, 4),
                           'valid_fraction': round(valid.size / window.size, 2)})
        return {'frame_id': frame_id, 'camera': camera, 'points': points}, []

    def _stream(self, cameras=None, seconds_ago=None, sheet=None):
        from . import stream as recording
        if self.stream is None:
            raise ValueError('the continuous recording is disabled in this session')
        if seconds_ago is not None and sheet is not None:
            raise ValueError('give seconds_ago or sheet, not both')
        names = list(cameras or sorted(self.models))
        window_ms = int(self.options['stream_window_s'] * 1000)
        now_ms = int(time.time() * 1000)
        images, report = [], {'now_ms': now_ms, 'cameras': names}

        def encode(image, width):
            scale = width / image.shape[1]
            small = cv.resize(image, None, fx=scale, fy=scale, interpolation=cv.INTER_AREA) if scale < 1 else image
            ok, jpeg = cv.imencode('.jpg', small, [cv.IMWRITE_JPEG_QUALITY, 85])
            images.append({'type': 'image', 'mimeType': 'image/jpeg', 'data': base64.b64encode(jpeg).decode()})

        if sheet is not None:
            seconds = min(float(sheet['seconds']), window_ms / 1000)
            report['sheets'] = []
            for name in names:
                grid, times = recording.contact_sheet(self.stream, name, now_ms - int(seconds * 1000), now_ms,
                                                      int(sheet.get('count', 12)), now_ms)
                if grid is None:
                    report['sheets'].append({'camera': name, 'error': 'no frames in that span'})
                    continue
                encode(grid, 1280)
                report['sheets'].append({'camera': name, 'frame_ages_s': [round((now_ms - t) / 1000, 1)
                                                                          for t in times]})
            report['robot'] = {'start': recording.robot_at(self.stream, now_ms - int(seconds * 1000)),
                               'end': recording.robot_at(self.stream, now_ms)}
            return report, images
        ages = list(seconds_ago if seconds_ago is not None else [0])
        if len(ages) * len(names) > 12:
            raise ValueError('at most 12 images per call (seconds_ago x cameras)')
        report['frames'] = []
        for age in ages:
            target = now_ms - int(float(age) * 1000)
            if age * 1000 > window_ms:
                raise ValueError(f"only the last {self.options['stream_window_s']} s are kept")
            entry = {'seconds_ago': age, 'robot': recording.robot_at(self.stream, target), 'images': []}
            for name in names:
                found = recording.nearest(self.stream, name, target)
                if found is None:
                    entry['images'].append({'camera': name, 'error': 'no frame near that time'})
                    continue
                t, path = found
                image = recording.label(cv.imread(str(path)), f'{name} -{(now_ms - t) / 1000:.1f}s')
                encode(image, self.options['agent_image_width'])
                entry['images'].append({'camera': name, 't_ms': t, 'age_s': round((now_ms - t) / 1000, 2),
                                        'file': str(Path('runtime/stream')/path.relative_to(self.stream))})
            report['frames'].append(entry)
        return report, images

    # ----- harness hooks ------------------------------------------------------------------
    def _pose_report(self, state):
        tcp_p, tcp_q = link8_to_tcp(state['arm_pos'], state['arm_quat'])
        return {'tcp_position': np.round(tcp_p, 5).tolist(), 'tcp_quaternion': np.round(tcp_q, 5).tolist(),
                'tcp_rpy_deg': np.round(Rotation.from_quat(tcp_q).as_euler('xyz', degrees=True), 2).tolist()}

    def status(self):
        state = self.arm.state()
        value = {'robot': 'franka_fr3', 'motion': arming.read(self.config.root), **self._pose_report(state),
                 'joints': np.round(np.asarray(state['arm_joint_position']), 4).tolist(),
                 'controller_running': bool(state.get('policy_running')),
                 'workspace_box': {'min': self.box_min.tolist(), 'max': self.box_max.tolist()},
                 'table_z_m': -0.21, 'cameras': sorted(self.models), 'depth': self.depth,
                 'stream': None if self.stream is None else {
                     'directory': 'runtime/stream', 'window_s': self.options['stream_window_s'],
                     'fps': self.options['stream_fps'], 'depth_hz': self.options['stream_depth_hz'],
                     'now_ms': int(time.time() * 1000)},
                 'last_motion': self.last_motion}
        try:
            grip = self.gripper.state()
            value['gripper'] = {k: grip.get(k) for k in ('width', 'max_width', 'is_grasped', 'is_moving', 'stale')}
        except Exception as exc:
            value['gripper'] = {'error': str(exc)}
        return value

    def close(self):
        self._stop.set()
        try:
            self.arm.hold()
        except Exception:
            pass
        self.cameras.close()
