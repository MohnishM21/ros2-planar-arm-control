# Part 2: From simulation to physical hardware

In simulation, everything in this project is ideal:
- the "measurement" is the model's exact state
- commands take effect instantly
- the control loop never misses a tick
- the arm can't hurt anyone

None of that holds on a Dynamixel-driven arm. Here is what I would expect to
break, and what I'd do about it, using this controller and GUI as the
reference.

## 1. Timing and communication: the first bottleneck

**The problem.**
- Every control tick becomes a round trip on a half-duplex RS-485/TTL bus.
  At 1 Mbps, reading position, velocity and current from 3 servos plus writing
  goal values takes roughly 1–3 ms. USB-serial adapters add their own latency:
  the FTDI default latency timer is 16 ms unless you lower it.
- My 100 Hz Python loop shows 0.5–2 ms of jitter on a desktop. On a loaded
  embedded computer with Python's GC and the GIL, there will be occasional
  10+ ms stalls.
- With the GUI and other nodes on the same machine, DDS traffic and rendering
  compete for the CPU.

**Mitigations.**
- Use `GroupSyncRead` and `GroupSyncWrite`, so each tick is two bus transactions
  however many joints there are. Raise the baud rate, set the FTDI latency
  timer to 1 ms, and only read the registers you need.
- Move the inner loop as close to the motor as possible. Dynamixel position
  and velocity modes already close the loop at kHz rates on the servo. For
  effort control, run the loop in C++ (`ros2_control` hardware interface) on a
  real-time-capable thread (`SCHED_FIFO`, PREEMPT_RT kernel, pinned core), and
  keep Python for planning and the UI only.
- Measure everything. My controller already publishes the control-period
  mean and worst-case jitter in `/arm_status`. On hardware I'd add bus
  round-trip time and missed-deadline counts, and alarm on them.
- Use a deterministic time base. The trajectory should be sampled from the
  clock the measurement was taken against, and `dt` clamped (I already clamp
  it to 3× the nominal period) so a stall doesn't cause a huge integration step.

## 2. The plant is not the model

**The problem.**
- *Backlash and compliance.* Gear backlash, link flex and cable drag mean
  joint encoders don't tell you exactly where the tool is.
- *Friction.* Static friction and deadband: small velocity or current commands
  produce no motion at all, and then the joint jumps. My PID integral would
  wind up against static friction and overshoot.
- *Variable gravity and payload.* The gravity load changes with pose and with
  the picked object. In the sim the integrator handled it. On hardware that's
  slow and causes overshoot after each pick.
- *Saturation and heat.* Motors run into torque, current and voltage limits,
  and thermal derating. A trajectory that's feasible in simulation can demand
  more current than the servo gives.

**Mitigations.**
- Identify the plant: step and chirp tests per joint, then fit inertia,
  friction and gravity parameters. Use the result for feed-forward (a gravity
  and friction compensation term in `TrackingController`) so the PID only
  corrects small residuals.
- Plan within real limits. Derive `max_joint_velocity` and
  `max_joint_acceleration` from the measured torque budget with margin, and
  re-time the trajectory when the payload changes.
- Re-tune gains on the hardware, starting low. Add derivative filtering,
  because encoder quantisation makes raw D noisy. Use conditional integration
  so the integrator stops when output saturates or when the error is inside
  the deadband.
- Add noise and lag to the simulation before deploying. The `sim_noise_std`
  parameter and the servo time constants in `SimulatedArm` exist for this. A
  controller that only works with perfect feedback isn't ready.

## 3. Calibration and state estimation

- **Zeroing and direction.** Each servo needs a zero offset and a direction
  sign so its ticks map to the kinematic angles. This is what
  `zero_offsets`/`directions` in `DynamixelBackend` are for. A wrong offset
  makes FK lie, and the ground check becomes meaningless.
- **Kinematic calibration.** Real link lengths and offsets differ slightly
  from 3.0/2.0/1.5. Measure the tool position at many poses and fit the
  parameters. Otherwise the arm can't reliably reach (4, 2) where the object
  actually is.
- **Gripper state.** In sim, "object attached" is a distance check. On
  hardware, confirm the grasp from gripper current or position, or from a
  sensor, and fail the action if the grasp didn't happen. The action already
  has a "grasp failed" abort path for this.

## 4. Safety, which simulation lets you ignore

- **Software limits inside the hard limits.** Use soft joint limits a few
  degrees inside the mechanical ones, plus the servo's own limit registers as
  a second layer. Keep a ground margin (y ≥ 0.05, not 0), because tracking
  error lets the real arm stray from the planned path. My planner already
  checks every sampled point of the path, not just the endpoints. On hardware
  I would also check the measured state against the limits on every tick.
- **Watchdogs and E-stop.** If the controller stops receiving commands, or the
  GUI or network dies, the arm must hold or brake on its own. Dynamixels need
  a watchdog configured for this. Add a hardware E-stop that cuts motor power
  independently of software. The GUI's STOP button is a convenience, not a
  safety function.
- **Bring-up sequence.** Use a lifecycle node: configure (open the port, read
  the current pose), then activate (torque on and hold the *measured* pose, so
  the arm doesn't jump to a stale reference). Always run at reduced speed
  first.
- **Monitoring.** Track temperature, voltage, current and hardware error bits
  every cycle, and degrade gracefully (slow down, then stop) instead of
  failing mid-motion.

## 5. What would change in this codebase

- **Controller.**
  - Swap `backend:=sim` for a `DynamixelBackend`.
  - Move the 100 Hz loop into a `ros2_control` hardware interface (C++), or
    use the servos' built-in position mode and stream setpoints only.
  - Add gravity and friction feed-forward, soft limits, a watchdog and a
    lifecycle node.
  - Publish hardware diagnostics (`diagnostic_msgs`).
- **Planner.** Use real velocity and acceleration limits, re-time per payload,
  add ground and obstacle margins, and add Cartesian approach and retreat
  moves for picking.
- **GUI.**
  - Show *measured* versus *commanded* positions prominently. The reference
    "ghost" arm already exists, and on hardware the gap between the two is the
    first thing an operator should see.
  - Add servo temperature, current and error flags.
  - Add a clear torque-on/off and E-stop state, a stale-data indicator (no
    `/joint_states` for more than N ms means a red border), and keep the
    commands that move the arm deliberate (confirm before moving, disable the
    buttons while a motion is running).
  - The GUI should also never sit in the safety path. That's already true:
    it only calls services and actions, and the controller enforces every
    limit itself.

**The bottom line.** The architecture holds: the planner, the trajectory and
the interfaces don't change. The work is in the backend, the timing, the
model fidelity and safety layers that simulation never had to have.
