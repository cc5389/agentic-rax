---
name: object-localization
description: Metric 3-D positions of visible object points in the Franka base frame from the calibrated ZED RGB-D cameras. Use before choosing any approach, grasp or placement target.
---

# Object localization (RGB-D)

Three calibrated ZED cameras: `left` and `right` are fixed exterior views, `wrist` rides
on the hand. Every `{{ robot.observe }}` call saves a frame under
`{{ vocab.frames }}<frame_id>/`: `<camera>.jpg` (848x480 RGB), `<camera>_depth.png`
(uint16 millimetres, 0 = invalid) and `meta.json` with each camera's intrinsics `K` and
pose `T_base_cam` (4x4, camera to base) at capture time.

## Procedure

1. Observe now; never reuse pixels from an earlier frame after anything moved.
2. Find the object in the full-resolution image (the tool previews are downscaled by
   `preview_scale`; divide preview pixels by it).
3. Call `{{ robot.pixel }} {frame_id, camera, pixels: [[u, v], ...]}` for a few pixels
   on the object's visible top surface. It returns base-frame points from the median
   depth around each pixel.
4. Cross-check: the same object from the other exterior camera should agree within
   about 1 cm; a point on the table should come out at z ~= -0.21 m. Disagreement means
   a wrong pixel (edge, shadow, occluder) - pick interior pixels and retry.
5. Object centre: average several top-surface points in x, y; height = their z. For a
   grasp, the TCP goes to the centre at roughly (top z - half the object height).

Depth is unreliable on object edges, shiny or transparent surfaces and closer than
0.3 m (exterior) / 0.1 m (wrist); the wrist camera gives the best close-up geometry
once it looks straight down from 0.25-0.4 m above the object.

To check a target before moving, pass it to `{{ robot.observe }}` as
`project: [[x, y, z]]`; it is drawn as a red cross on every image.

For bulk processing in Python, load the PNG with OpenCV (`cv2.IMREAD_UNCHANGED`) and
back-project with `K` and `T_base_cam` from `meta.json`.
