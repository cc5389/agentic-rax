"""The slice of the ZED SDK's `pyzed.sl` that services/franka/camera_server.py uses, served by Intel RealSense.

Put `hardware/realsense_shim` first on the camera server's PYTHONPATH and `import pyzed.sl as sl` opens RealSense
cameras by serial instead: colour and depth at the same resolution (848x480 by default), depth aligned to colour,
the colour intrinsics as the "left camera", depth in metres with NaN where invalid (as the ZED reports it), and
timestamps on the host's epoch clock. ZED-only settings (HD720, NEURAL depth) are ignored.

Environment:
  RAX_REALSENSE_RESOLUTION   WIDTHxHEIGHT of both streams (default 848x480, the size the rig is calibrated at)
  RAX_REALSENSE_SETTINGS     optional JSON {serial: {"exposure": n, "gain": n, "white_balance": n}}, the rig's pinned
                             colour settings (sim2real cfg/real_world.yaml cameras.*); a camera not listed stays auto
"""
from enum import Enum
import json
import os
import time

import numpy as np


class ERROR_CODE(Enum):
    SUCCESS = 0
    FAILURE = 1
    CAMERA_NOT_DETECTED = 2


RESOLUTION = Enum('RESOLUTION', 'HD2K HD1080 HD1200 HD720 SVGA VGA AUTO')
DEPTH_MODE = Enum('DEPTH_MODE', 'NONE PERFORMANCE QUALITY ULTRA NEURAL NEURAL_LIGHT NEURAL_PLUS')
UNIT = Enum('UNIT', 'MILLIMETER CENTIMETER METER INCH FOOT')
VIEW = Enum('VIEW', 'LEFT RIGHT')
MEASURE = Enum('MEASURE', 'DEPTH')
TIME_REFERENCE = Enum('TIME_REFERENCE', 'IMAGE CURRENT')


class InitParameters:
    def __init__(self):
        self.serial = None
        self.camera_resolution = RESOLUTION.HD720
        self.camera_fps = 30
        self.coordinate_units = UNIT.METER
        self.depth_mode = DEPTH_MODE.NONE
        self.depth_minimum_distance = 0.0
        self.depth_maximum_distance = 10.0

    def set_from_serial_number(self, serial):
        self.serial = str(serial)


class RuntimeParameters:
    def __init__(self):
        self.enable_depth = True


class Mat:
    def __init__(self):
        self.data = None

    def get_data(self):
        return self.data


class Timestamp:
    def __init__(self, ns):
        self.ns = int(ns)

    def get_nanoseconds(self):
        return self.ns


class _Namespace:
    def __init__(self, **values):
        self.__dict__.update(values)


def _resolution():
    width, height = os.environ.get('RAX_REALSENSE_RESOLUTION', '848x480').lower().split('x')
    return int(width), int(height)


def _settings(serial):
    path = os.environ.get('RAX_REALSENSE_SETTINGS')
    if not path:
        return {}
    with open(path) as handle:
        return json.load(handle).get(str(serial), {})


def _color_sensor(device, rs):
    try:
        return device.first_color_sensor()
    except RuntimeError:  # D405: colour rides the stereo module
        return next(s for s in device.query_sensors()
                    if any(p.stream_type() == rs.stream.color for p in s.get_stream_profiles()))


class Camera:
    def __init__(self):
        self.pipeline = self.align = self.frames = None
        self.stamp_ns = 0

    def open(self, init):
        import pyrealsense2 as rs
        self.rs = rs
        self.min_m, self.max_m = float(init.depth_minimum_distance), float(init.depth_maximum_distance)
        width, height = _resolution()
        config = rs.config()
        config.enable_device(init.serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, int(init.camera_fps))
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, int(init.camera_fps))
        self.pipeline = rs.pipeline()
        try:
            profile = self.pipeline.start(config)
        except RuntimeError as exc:
            print(f'realsense {init.serial}: {exc}', flush=True)
            self.pipeline = None
            return ERROR_CODE.CAMERA_NOT_DETECTED
        device = profile.get_device()
        self.depth_scale = device.first_depth_sensor().get_depth_scale()
        for sensor in device.query_sensors():
            if sensor.supports(rs.option.global_time_enabled):
                sensor.set_option(rs.option.global_time_enabled, 1)
        color = _color_sensor(device, rs)
        settings = _settings(init.serial)
        if settings.get('exposure') is not None or settings.get('gain') is not None:
            color.set_option(rs.option.enable_auto_exposure, 0)
            for key in ('exposure', 'gain'):
                if settings.get(key) is not None:
                    color.set_option(getattr(rs.option, key), float(settings[key]))
        if settings.get('white_balance') is not None:
            color.set_option(rs.option.enable_auto_white_balance, 0)
            color.set_option(rs.option.white_balance, float(settings['white_balance']))
        self.align = rs.align(rs.stream.color)
        intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        left = _Namespace(fx=intrinsics.fx, fy=intrinsics.fy, cx=intrinsics.ppx, cy=intrinsics.ppy,
                          width=intrinsics.width, height=intrinsics.height, disto=list(intrinsics.coeffs))
        self.information = _Namespace(serial_number=int(init.serial), camera_model=device.get_info(rs.camera_info.name),
                                      camera_configuration=_Namespace(
                                          calibration_parameters=_Namespace(left_cam=left), fps=init.camera_fps))
        return ERROR_CODE.SUCCESS

    def get_camera_information(self):
        return self.information

    def grab(self, runtime=None):
        try:
            frames = self.pipeline.wait_for_frames(1000)
        except RuntimeError:
            return ERROR_CODE.FAILURE
        self.frames, self.depth_wanted = frames, runtime is None or runtime.enable_depth
        domain = frames.get_frame_timestamp_domain()
        if domain == self.rs.timestamp_domain.global_time:  # host epoch milliseconds
            self.stamp_ns = int(frames.get_timestamp() * 1e6)
        else:
            self.stamp_ns = time.time_ns()
        return ERROR_CODE.SUCCESS

    def get_timestamp(self, reference=None):
        return Timestamp(self.stamp_ns if reference != TIME_REFERENCE.CURRENT else time.time_ns())

    def retrieve_image(self, mat, view=None):
        bgr = np.asanyarray(self.frames.get_color_frame().get_data())
        mat.data = np.concatenate([bgr, np.full(bgr.shape[:2] + (1,), 255, np.uint8)], axis=2)  # ZED gives BGRA
        return ERROR_CODE.SUCCESS

    def retrieve_measure(self, mat, measure=None):
        aligned = self.align.process(self.frames)
        depth = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32) * self.depth_scale
        depth[(depth <= 0) | (depth < self.min_m) | (depth > self.max_m)] = np.nan
        mat.data = depth
        return ERROR_CODE.SUCCESS

    def close(self):
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except RuntimeError:
                pass
            self.pipeline = None
