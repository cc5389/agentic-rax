"""Calibration check through RAX's own backend: project the TCP, both fingertips and the flange into every camera.

Builds a FrankaBackend from $RAX_BACKEND_OPTIONS (the same cameras, extrinsics and bridges a session uses), calls
`franka_observe` with those points and saves full-resolution overlays. On a correct calibration the red crosses
sit on the fingertip pads in every view, the wrist camera included. Read-only: nothing moves.

    source site/robopil-franka/env.sh && $RAX_PYTHON -m hardware.check_calibration [--out DIR]
"""
import argparse
import json
import os
from pathlib import Path
import time

import cv2 as cv
import numpy as np

from services.controller.config import Configuration
from services.franka.backend import FrankaBackend
from services.franka.calibration import project
from services.franka.kinematics import link8_to_tcp, matrix
from services.storage_root import artifact_root

COLOURS = {'tcp': (0, 0, 255), 'finger_left': (0, 255, 255), 'finger_right': (255, 255, 0), 'flange': (255, 0, 255)}


def points(state, width):
    """Base-frame TCP, fingertip pads (TCP -/+ half the opening along the hand y axis) and flange."""
    tcp_p, tcp_q = link8_to_tcp(state['arm_pos'], state['arm_quat'])
    y = matrix(tcp_p, tcp_q)[:3, 1]
    half = 0.5 * float(width)
    return {'tcp': tcp_p, 'finger_left': tcp_p + half * y, 'finger_right': tcp_p - half * y,
            'flange': np.asarray(state['arm_pos'])}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', type=Path, default=None)
    args = parser.parse_args(argv)
    options = json.loads(Path(os.environ['RAX_BACKEND_OPTIONS']).read_text())
    root = artifact_root()/'checks'/time.strftime('%Y%m%dT%H%M%S')
    root.mkdir(parents=True)
    (root/'operator-input.json').write_text('{}')
    options = {**options, 'stream_enabled': False}
    backend = FrankaBackend(Configuration(root=root, task='calibration_check', robot='franka', backend_options=options))
    try:
        width = backend.gripper.state().get('width', 0.08)
        marks = points(backend.arm.state(), width)
        report, _ = backend.call('franka_observe', {'project': [p.tolist() for p in marks.values()]})
        frame = backend.frames_root/report['frame_id']
        meta = json.loads((frame/'meta.json').read_text())
        out = args.out or root/'overlays'
        out.mkdir(parents=True, exist_ok=True)
        for name, entry in meta['cameras'].items():
            image = cv.imread(str(frame/entry['rgb']))
            for label, point in marks.items():
                uv = project(point, np.asarray(entry['T_base_cam']), entry['K'])
                if uv is None:
                    continue
                centre = (int(round(uv[0])), int(round(uv[1])))
                cv.drawMarker(image, centre, COLOURS[label], cv.MARKER_CROSS, 18, 2)
                cv.putText(image, label, (centre[0] + 8, centre[1] - 8), cv.FONT_HERSHEY_SIMPLEX, 0.45,
                           COLOURS[label], 1, cv.LINE_AA)
            cv.imwrite(str(out/f'{name}.png'), image)
        print(json.dumps({'overlays': str(out), 'gripper_width_m': width,
                          'points': {k: np.round(v, 4).tolist() for k, v in marks.items()},
                          'pixels': {n: c.get('projected_pixels') for n, c in report['cameras'].items()}}, indent=1))
    finally:
        backend.close()


if __name__ == '__main__':
    main()
