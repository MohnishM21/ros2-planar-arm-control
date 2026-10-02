# Design note

## Architecture

```
 GUI node (PyQt5)                         controller_node (MultiThreadedExecutor)
 ┌───────────────────────────┐            ┌──────────────────────────────────────────────┐
 │ Qt main thread            │  /move_to_target, /home, /stop, /pick_place (action)        │
 │  widgets, QTimer @30 Hz ──┼──────────▶ │ commands group (reentrant)                   │
 │  ▲ queued Qt signals      │            │   PlanningWorker (separate process)          │
 │  │                        │            │     MotionPlanner ── IK + validation + path  │
 │  │ TelemetryBuffer (lock) │            │        │ QuinticTrajectory                   │
 │ ROS thread (executor) ◀───┼─────────── │ control group  @100 Hz                       │
 │  subscriptions            │ /joint_states  trajectory.sample → TrackingController     │
 └───────────────────────────┘ /arm_status│   → ArmBackend.write / step / read           │
                                          │ publish group  @50 Hz  (telemetry)           │
                                          └────────────────────┬─────────────────────────┘
                                                               ▼
                                         ArmBackend: SimulatedArm | DynamixelBackend (stub)
```

The nodes are thin. Everything that decides how the arm moves lives in plain
Python classes with no ROS dependency, and each is unit-tested on its own:
- `MotionPlanner` (target → validated trajectory)
- `QuinticTrajectory`
- `TrackingController` + `PID` (reference → command)
- `ArmBackend` (command → arm)

**Swapping in hardware.** The planner and tracker only ever produce a
`JointCommand(mode, values)` and consume a `JointReading`. A Dynamixel arm
means implementing `read`/`write`/`set_mode` with one GroupSyncRead and one
GroupSyncWrite per tick. The skeleton documents the register mapping.

**Concurrency.**
- *Controller:* the control loop, the telemetry publisher and the command
  handlers each have their own callback group. The pick-and-place action can
  sleep while a motion runs without ever delaying a control tick. Shared state
  sits behind one lock, held only to copy small snapshots.
- *GUI:* ROS spins in a daemon thread, and its callbacks never touch widgets.
  Streaming data goes through a locked ring buffer that a fixed-rate `QTimer`
  renders from, so render rate doesn't depend on message rate. One-off events
  (service and action replies) cross into Qt as queued signals. Outgoing calls
  are all `*_async`.
- *Planning runs in a separate process.* IK is CPU-heavy Python, and run in a
  thread it holds the GIL. I measured worst-case control-loop stalls of
  **19 ms** for a plain pick-and-place and **142 ms** once straight-line moves
  were added. Moving `MotionPlanner` into a single-worker
  `ProcessPoolExecutor` (spawn context) brought that down to **~4–5 ms**. The
  command thread just blocks on a future, which releases the GIL.
- *Measured:* 100 Hz loop at 10.0 ms mean period, ≤ 5 ms worst-case jitter
  during a full pick-and-place. GUI at 30 fps, `/joint_states` at 50 Hz.

## Key decisions and trade-offs

- **Quintic over trapezoidal profiles.** Acceleration is continuous, so there
  are no jerk spikes, which on real servos show up as current spikes. The cost
  is about 25 % higher peak velocity for the same duration. The duration is
  derived from per-joint velocity and acceleration limits, and all joints are
  synchronised.
- **Joint-space interpolation is checked, not trusted.** Valid start and end
  poses don't guarantee a valid path: a link can sweep through the floor on the
  way. Every trajectory is sampled against the joint limits and the ground. If
  the direct path fails, it is re-planned through a safe home pose; if that
  fails too, the move is rejected. A Cartesian straight-line planner would be
  smoother for the end-effector but needs per-step IK with branch consistency,
  so I left it out of scope.
- **Approach, grasp and retreat follow a straight line.** The tool hovers 0.5
  above each point and descends and lifts in a straight vertical line, so it
  never sweeps sideways into the part. The line is a dense IK path, each point
  seeded with the previous solution so the solver stays on one consistent
  branch; a joint jump of more than 0.15 rad between neighbours rejects the
  line. That path is then time-scaled with the same quintic profile. If the
  line fails, the planner falls back to a joint-space move.
- **Grip only once the arm is in position.** The trajectory finishing on the
  clock doesn't mean the arm is there: in effort mode the lag at the end of a
  brisk 0.5 s descent was 0.11 m. The integration test caught this as a missed
  grasp. The sequencer now waits for the hand to be within 1 cm of the target
  and nearly still (with a timeout) before gripping, like polling a servo's
  "Moving" flag.
- **Unreachable targets are handled per use case.** A plain move goes to the
  projected boundary point and reports that it did. Pick-and-place rejects the
  goal up front, since grasping at a projected point is meaningless.
- **Three control modes, switchable while idle.**
  - *Position:* the servo closes the loop, like a Dynamixel in position mode.
  - *Velocity:* velocity feed-forward plus a PI loop.
  - *Effort:* inertia feed-forward plus PID, with the integral term absorbing
    gravity.

  Two details mattered in practice. First, the D term acts on the velocity
  error (qd_ref − qd): derivative-on-measurement fought the motion it was
  meant to track, giving 8° of lag. Second, anti-windup clamps the integral
  term in output units, so its authority is explicit.
- **Custom interfaces live in their own `ament_cmake` package.** That keeps
  the control package pure `ament_python`. An action fits pick-and-place
  because it is long-running, needs progress reporting, and must be cancellable.
- **The simulated plant is deliberately simple.** It has decoupled per-joint
  inertia, damping, gravity, servo lag and hard stops. That's enough to show
  tracking error and make the controller modes meaningfully different, but it
  is not a rigid-body simulation (no coupling between joints).

## Notes on the provided `planar_arm.py` (left unmodified)

1. **The Jacobian fallback ignores joint limits and the ground.** For (0.5, 0.2)
   it silently returns q2 = −144°. I validate every IK result and reject it.
2. **`wrap_angle(π)` returns −π**, so joint 1 can never actually reach its
   180° limit. For example, (−6.5, 0) is rejected even though it is a legal
   pose.
3. **`reachable_target` only handles maximum reach.** It doesn't cover targets
   below the ground or the parts of the reach circle that the joint limits make
   unreachable, and it reports by `print()`, so callers can't detect it. Its
   `distance == 0` branch can never run. My planner does its own projection
   (also clamping to y ≥ 0) and redirects the library's stdout.

## With more time

- A Gazebo (`ros_gz`) backend behind the same `ArmBackend` interface.
- A lifecycle node for configure/activate, mapping to torque on/off on hardware.
- Gravity-compensation feed-forward from a model, instead of relying on the
  integrator.
- Live trajectory preemption with blending from the current velocity. Right
  now a new move is refused while one is running.
- Gain scheduling and auto-tuning, and a real-time executor (rclcpp) for the
  control loop.
