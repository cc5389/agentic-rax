---
name: robot-interface
description: The real Franka FR3's motion and gripper tools, frames, limits, arming and failure handling. Read before composing any robot action or writing robot code.
---

# Franka FR3 robot interface

One 7-DoF Franka FR3 arm with a Franka Hand, on a table. A person stands at the
emergency stop. Everything you command happens physically.

## Frames and units

- Base frame `fr3_link0`: x forward (away from the robot base, over the table), y to
  the robot's left, z up. Metres, radians, quaternions `[x, y, z, w]`.
- The TCP is `fr3_hand_tcp`: the point between the fingertip pads, 0.1034 m beyond the
  flange. Every pose tool uses the TCP, never the flange.
- Gripper pointing straight down: the TCP z axis points along base -z. Read the current
  `tcp_quaternion` from `{{ robot.status }}` and keep it unless you need a new
  orientation; to yaw about the vertical, rotate that quaternion about base z.
- The table top is at z ~= -0.21 m. Objects' metric positions come from
  [object-localization](../object-localization/SKILL.md).

## Tools

- `{{ robot.status }}`: pose, joints, gripper, whether motion is armed, workspace box.
- `{{ robot.observe }}`: camera images now (left, right exterior; wrist on the hand).
- `franka_move_pose {position, quaternion?, relative?, tolerance_mm?, timeout_s?}`:
  straight-line TCP move that waits until settled (default 1.5 mm). At most 0.40 m per
  call, about 50 mm/s. `relative: true` adds `position` to the current TCP position.
- `franka_move_joints {positions | preset: "home"}`: joint-space move; the whole path
  is checked against the workspace box and table clearance before anything moves.
  Use `home` to recover a neutral, well-conditioned posture.
- `{{ robot.gripper }} {action: open | move | grasp, width?, force?, speed?}`: `grasp`
  closes to `width` with `force` newtons and keeps squeezing; use it to hold objects
  (`move` is position only and lets a held object push the fingers open). Max width
  0.08 m. A successful grasp reports `is_grasped: true`.
- `franka_hold`: hold the current pose.

## Limits and arming

- Motion and gripper tools work only while the operator has armed motion. If
  `status` says motion is not armed, stop and report; never try to work around it.
  The operator may disarm at any time: a running move then stops and the arm holds.
- Targets outside the workspace box or too far per call are refused before moving.
  A refusal by the controller (joint limit, IK) is a wall, not a fault: choose another
  approach (different height, orientation, or go `home` first).
- The arm is compliant (impedance control): contact produces force, not a crash, but
  never press into the table or objects. Approach from above, descend slowly in short
  steps near contact, and check the measured pose after each step.

## Grasping pattern

1. Observe, localize the object (centre and top height) from a fresh frame.
2. Open the gripper; move above the object at a safe height (>= 0.10 m above its top).
3. Yaw so the fingers close across the object's narrow dimension.
4. Descend to grasp height (fingertips around the object's middle, never below the
   table + 0.01 m), grasp, then lift vertically and check `is_grasped` and the images.

## In your own code

Your Python reaches the same tools through `api.runtime.Context` (see
`MCP_SESSION.md`): `ctx.value('franka_move_pose', position=[...])`. Robot calls are
serialized with your direct calls; keep scripts short, check every reply, and stop on
the first unexpected result.
