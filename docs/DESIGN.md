# Design note

## Architecture

```
 gui_node (PyQt5)                          controller_node (MultiThreadedExecutor)
 Qt main thread: widgets, redraw @30 Hz    control loop @100 Hz ──▶ ArmBackend (sim | Dynamixel)
 ROS thread: subscriptions ─▶ locked buffer telemetry   @50 Hz  ──▶ /joint_states, /arm_status
 buttons ── async services / action ──────▶ commands: /move_to_target, /home, /stop, /pick_place
                                              └─▶ planning worker process (IK + trajectories)
```

The nodes are thin. The logic lives in ROS-free, unit-tested classes:
- `MotionPlanner`: target → validated path
- `QuinticTrajectory`: smooth timing along that path
- `TrackingController`/`PID`: reference → command
- `ArmBackend`: command → arm

The planner never talks to motors directly; it produces a `JointCommand`. So
moving to a Dynamixel arm means writing one backend class (`read`/`write`/`set_mode`),
and nothing above it changes.

## Key decisions and trade-offs

- **Qt and ROS never block each other.** In the GUI, ROS spins in a background
  thread and only writes into a lock-protected buffer. A Qt timer redraws from
  that buffer at 30 Hz. Replies come back as queued Qt signals, and every
  outgoing call is async. In the controller, the control loop, telemetry and
  command handlers run in separate callback groups, so a long pick-and-place
  never delays a control tick.
- **Planning runs in a separate process.** IK is CPU-heavy Python. Running in a
  thread, it held the GIL and I measured control-loop stalls of up to
  **142 ms**. In a worker process it has its own GIL, and the worst stall
  dropped to **≈5 ms**.
- **Quintic, not trapezoidal, profiles.** Acceleration is continuous, so there
  are no jerks, which real servos feel as current spikes. The cost is about
  25 % higher peak speed for the same duration.
- **Paths are checked, not just endpoints.** A joint-space move between two
  valid poses can still swing a link through the floor. Every path is sampled
  against the joint limits and the ground; if the direct path fails, it goes
  via the home pose, and otherwise the move is rejected.
- **Realistic pick.** The tool hovers 0.5 above the object, descends in a
  straight line (IK at each point, checked for sudden joint jumps), and grips
  only once the arm has **settled**. My integration test caught a missed grasp
  in effort mode, where the arm lagged 0.11 m behind the trajectory's end.
- **Unreachable targets depend on the task.** A plain move to (7, 3) goes to the
  nearest reachable point (5.97, 2.56), and the GUI shows a warning banner.
  Pick-and-place refuses an unreachable target, because grasping at the wrong
  spot is pointless.
- **Three control modes, switchable while idle.**
  - *Position:* the servo tracks the path itself.
  - *Velocity:* feed-forward plus PI.
  - *Effort:* inertia feed-forward plus PID; the integral term holds against
    gravity.

  Putting D on the *velocity error*, not on the measurement, removed 8° of lag.
- **Pick-and-place is a ROS 2 action** because it is long-running, reports
  progress, and must be cancellable.

**Results:** the fixed scenario passes in all three modes, and 31 tests pass
(unit, lint, and a launch test against the real node).

## Issues in the provided `planar_arm.py` (not modified)

1. The Jacobian fallback ignores joint limits and the ground. For (0.5, 0.2)
   it silently returns q2 = −144°. I validate every IK result.
2. `wrap_angle(π)` returns −π, so joint 1 can never reach its 180° limit.
3. `reachable_target` ignores the ground and joint-limit gaps, and reports only
   with `print()`. I do my own projection.

## With more time

- A Gazebo backend
- Lifecycle nodes (torque on/off)
- Model-based gravity compensation
- Blending into a new target mid-motion
- A C++ real-time control loop
