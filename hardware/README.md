# robopil FR3 adapters

This directory adapts the robopil rig to the interfaces that `real/franka` expects. RAX itself is unchanged apart from the rig's table height in the skills and backend, and `RAX_DENIED_HOSTS` in the launcher.

The rig:
- FR3 with a Franka Hand.
- The NUC runs Polymetis and wmrl's ZMQ joint-streaming bridge.
- Four RealSense cameras: two fixed D455s and two wrist D405s. RAX uses three of them.
- Calibration comes from sim2real.

| File | What it does |
| --- | --- |
| `zmq_http_bridge.py` | Serves RAX's arm (:50054) and gripper (:50053) HTTP bridges on 127.0.0.1, translated to the NUC ZMQ bridge. Targets are rate-limited (50 mm/s, 0.5 rad/s), IK-checked with RAX's IK, held after a 0.5 s watchdog, and refused (409) outside the lab joint limits. |
| `realsense_shim/pyzed/sl.py` | The part of `pyzed.sl` that `camera_server.py` uses, implemented on RealSense. It is registered in the robot env through a `.pth` file. |
| `import_calibration.py` | Converts sim2real or Agentics calibration into RAX extrinsics files. |
| `check_calibration.py` | Projects the TCP, fingertips and flange into every camera through RAX's backend, and saves overlay images. |

## Site files

The site files live in `site/robopil-franka/`, which git ignores (the fork is public).

```
env.sh                    RAX_PYTHON, FRANKA_CALIBRATION_DIR, RAX_BACKEND_OPTIONS, RAX_PROTECTED_PATHS,
                          RAX_DENIED_HOSTS, RAX_NUC_BRIDGE, RAX_REALSENSE_SETTINGS, PATH (claude)
franka.json               cameras, workspace box (table at z ~= -0.21), link_min_z, agent_python
realsense_settings.json   pinned exposure / gain / white balance per serial (from sim2real)
calibration/              RAX extrinsics + the source block they came from
```

## Environments

Both environments live outside the repo, because the agent sandbox cannot read the repo.

- **`~/cccheng/envs/rax`** (robot env): numpy, scipy, opencv, pyyaml, jinja2, pillow, pytest, pyzmq and pyrealsense2. It also carries `rax_realsense_shim.pth`.
- **`~/cccheng/envs/rax-agent`** (the agent's Python): numpy, scipy, opencv, pillow, pyyaml and matplotlib. It has no robot or camera SDKs.

## Bring-up

Keep the e-stop in hand throughout.

```bash
source site/robopil-franka/env.sh
# 0. nobody else streams to the NUC bridge; start Polymetis + wmrl bridge on the NUC as usual
$RAX_PYTHON -m hardware.zmq_http_bridge            # terminal 1; --max-speed 0.02 for a slower first run
curl -s 127.0.0.1:50054/state; curl -s 127.0.0.1:50053/state   # joints must match Desk
$RAX_PYTHON -m hardware.check_calibration          # crosses on the fingertip pads in left/right/wrist
franka home                                        # slow; watch the wrist-camera mount and the table
# dashboard (scripts/run_dashboard.sh): arm, then a few cm moves with franka_move_pose; gripper move/grasp/open
$RAX_PYTHON scripts/configure_session.py --task <name> --task-file task.md --session-hours 1
bash scripts/start_agent.sh --config <session>/operator-input.json
franka arm 30
```

In the session, check from the agent's side that the sandbox cannot reach `$RAX_NUC_BRIDGE` or `127.0.0.1:50054`.

Offline, run `$RAX_PYTHON -m hardware.zmq_http_bridge --mock` for a simulated NUC. The cameras are real if they are plugged in.
