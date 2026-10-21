"""Write RAX extrinsics files from the robopil rig's calibration.

Source: sim2real `cfg/real_world.yaml` (its active block) or an Agentics-Robotics `cfg/calibration/<label>.yaml`.
Both use the same conventions:
- `extrinsics[serial]` is T_cam_base (P_cam = T @ P_base; sim2real calibrate.py `base_to_cam`), RAX's fixed mount;
- `wristcam_N_to_flange` is T_link8_cam (camera optical frame -> link8; sim2real hand_eye_polymetis.py), and
  N is the serial's first digit (wristcam_1 = 130322270130, wristcam_2 = 230322271147).
RAX's wrist mount wants T_cam_tcp against `fr3_hand_tcp`. With RAX's own link8 -> TCP transform,
T_cam_tcp = inv(T_link8_cam) @ T_link8_tcp, so RAX's T_base_tcp @ inv(T_cam_tcp) is exactly T_base_link8 @ T_link8_cam.

    $RAX_PYTHON -m hardware.import_calibration SOURCE.yaml OUT_DIR [--left S] [--right S] [--wrist S]
"""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from services.franka.kinematics import _T_LINK8_TCP

LEFT, RIGHT, WRIST = '239222300843', '338122303713', '230322271147'


def homogeneous(value, what):
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError(f'{what}: expected a 4x4 homogeneous matrix')
    if not np.allclose(matrix[:3, :3] @ matrix[:3, :3].T, np.eye(3), atol=1e-5):
        raise ValueError(f'{what}: the rotation is not orthonormal')
    return matrix


def wrist_key(serial):
    return f'wristcam_{str(serial)[0]}_to_flange'


def convert(source, *, left=LEFT, right=RIGHT, wrist=WRIST):
    """{file name: (matrix, note)} for RAX's three cameras."""
    extrinsics = {str(k): v for k, v in source['extrinsics'].items()}
    files = {}
    for name, serial in (('left', left), ('right', right)):
        if serial not in extrinsics:
            raise KeyError(f'no extrinsics for {name} camera {serial} (have {sorted(extrinsics)})')
        files[f'{name}_T_cam_base.yaml'] = (homogeneous(extrinsics[serial], serial), f'{serial}: T_cam_base')
    key = wrist_key(wrist)
    T_link8_cam = homogeneous(source[key], key)
    files['wrist_T_cam_gripper.yaml'] = (np.linalg.inv(T_link8_cam) @ _T_LINK8_TCP,
                                         f'{wrist}: T_cam_tcp = inv({key}) @ T_link8_tcp (RAX kinematics)')
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('source', type=Path)
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--left', default=LEFT)
    parser.add_argument('--right', default=RIGHT)
    parser.add_argument('--wrist', default=WRIST)
    args = parser.parse_args(argv)
    source = yaml.safe_load(args.source.read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    for name, (matrix, note) in convert(source, left=args.left, right=args.right, wrist=args.wrist).items():
        header = f'# {note}\n# imported {stamp} from {args.source.resolve()}\n'
        (args.out_dir/name).write_text(header + yaml.safe_dump({'matrix': np.round(matrix, 12).tolist()}))
        print(f'wrote {args.out_dir/name}  ({note})')


if __name__ == '__main__':
    main()
