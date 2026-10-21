"""RAX's ZED CameraWorker over the RealSense pyzed shim, with a fake pyrealsense2: nothing here touches hardware."""
import importlib.util
from pathlib import Path
import sys
import time
import types

import numpy as np
import pytest

SHIM = Path(__file__).resolve().parents[2]/'hardware/realsense_shim/pyzed/sl.py'
W, H = 848, 480


def fake_realsense():
    rs = types.ModuleType('pyrealsense2')
    rs.stream = types.SimpleNamespace(color='color', depth='depth')
    rs.format = types.SimpleNamespace(bgr8='bgr8', z16='z16')
    rs.option = types.SimpleNamespace(global_time_enabled='gt', enable_auto_exposure='ae', exposure='exposure',
                                      gain='gain', enable_auto_white_balance='awb', white_balance='wb')
    rs.timestamp_domain = types.SimpleNamespace(global_time='global', hardware_clock='hw')
    rs.camera_info = types.SimpleNamespace(name='name')
    rs.options_set = []

    class Frame:
        def __init__(self, data):
            self.data = data

        def get_data(self):
            return self.data

    class Frames:
        def __init__(self):
            self.stamp_ms = time.time() * 1000
            depth = np.full((H, W), 500, np.uint16)  # 0.5 m at scale 1e-3
            depth[0, 0] = 0
            self.color, self.depth = Frame(np.full((H, W, 3), 7, np.uint8)), Frame(depth)

        def get_frame_timestamp_domain(self):
            return rs.timestamp_domain.global_time

        def get_timestamp(self):
            return self.stamp_ms

        def get_color_frame(self):
            return self.color

        def get_depth_frame(self):
            return self.depth

    class Sensor:
        def supports(self, option):
            return True

        def set_option(self, option, value):
            rs.options_set.append((option, value))

        def get_depth_scale(self):
            return 0.001

    class Device:
        def first_depth_sensor(self):
            return Sensor()

        def first_color_sensor(self):
            raise RuntimeError('D405: no colour sensor')

        def query_sensors(self):
            profile = types.SimpleNamespace(stream_type=lambda: rs.stream.color)
            sensor = Sensor()
            sensor.get_stream_profiles = lambda: [profile]
            return [sensor]

        def get_info(self, key):
            return 'Intel RealSense D405'

    intrinsics = types.SimpleNamespace(fx=430.0, fy=431.0, ppx=420.0, ppy=240.0, width=W, height=H, coeffs=[0] * 5)

    class Profile:
        def get_device(self):
            return Device()

        def get_stream(self, stream):
            return types.SimpleNamespace(as_video_stream_profile=lambda: types.SimpleNamespace(
                get_intrinsics=lambda: intrinsics))

    class Config:
        def enable_device(self, serial):
            self.serial = serial

        def enable_stream(self, *args):
            pass

    class Pipeline:
        def start(self, config):
            return Profile()

        def wait_for_frames(self, timeout_ms):
            time.sleep(1 / 30)
            return Frames()

        def stop(self):
            pass

    rs.config, rs.pipeline = Config, Pipeline
    rs.align = lambda stream: types.SimpleNamespace(process=lambda frames: frames)
    return rs


@pytest.fixture
def sl(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'pyrealsense2', fake_realsense())
    settings = tmp_path/'settings.json'
    settings.write_text('{"230322271147": {"exposure": 33000, "gain": 16, "white_balance": 3000}}')
    monkeypatch.setenv('RAX_REALSENSE_SETTINGS', str(settings))
    spec = importlib.util.spec_from_file_location('shim_sl', SHIM)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_camera_worker_runs_on_the_shim(sl, tmp_path):
    from services.franka.camera_server import CameraWorker, depth_png
    worker = CameraWorker(sl, {'name': 'wrist', 'serial': 230322271147, 'depth_min_m': 0.07}, fps=30,
                          depth_mode='NEURAL', stream=None, stream_quality=80, depth_hz=0)
    try:
        assert worker.K == [[430.0, 0.0, 420.0], [0.0, 431.0, 240.0], [0.0, 0.0, 1.0]]
        stamp, bgr, depth = worker.newest(True, after_ns=0)
        assert bgr.shape == (H, W, 3) and bgr.dtype == np.uint8
        assert depth.shape == (H, W) and np.isnan(depth[0, 0]) and depth[1, 1] == pytest.approx(0.5)
        assert abs(stamp / 1e9 - time.time()) < 5  # host epoch clock, like the ZED's IMAGE timestamps
        assert len(depth_png(depth)) > 0
    finally:
        worker.close()
    assert ('ae', 0) in sys.modules['pyrealsense2'].options_set
    assert ('exposure', 33000.0) in sys.modules['pyrealsense2'].options_set


def test_depth_outside_range_is_invalid(sl):
    camera = sl.Camera()
    init = sl.InitParameters()
    init.set_from_serial_number(239222300843)
    init.depth_minimum_distance = 0.6  # a D455's useful range starts beyond the fake 0.5 m
    assert camera.open(init) == sl.ERROR_CODE.SUCCESS
    assert camera.grab(sl.RuntimeParameters()) == sl.ERROR_CODE.SUCCESS
    measure = sl.Mat()
    camera.retrieve_measure(measure, sl.MEASURE.DEPTH)
    assert np.isnan(measure.get_data()).all()
    camera.close()
