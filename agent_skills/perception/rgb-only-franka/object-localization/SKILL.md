---
name: object-localization
description: Locate visible object points in the Franka base frame from the calibrated RGB cameras without depth. Use before choosing any approach, grasp or placement target.
---

# Object localization (RGB only)

This session has no depth. Each `{{ robot.observe }}` frame saves RGB images and a
`meta.json` with each camera's intrinsics `K` and pose `T_base_cam`.

- Triangulate: find the same object point in the `left` and `right` images and
  intersect the two camera rays (from `K` and `T_base_cam`) in your own Python.
- Or intersect one ray with a known plane, e.g. the table top at z ~= -0.21 m for a
  point on the table, or the table plus a known object height.
- Check every estimate by projecting it back with `project: [[x, y, z]]` on a new
  `{{ robot.observe }}` and comparing the red cross with the object.
