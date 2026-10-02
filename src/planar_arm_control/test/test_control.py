import numpy as np
import pytest

from planar_arm_control.hardware_interface import (ControlMode, JointCommand,
                                                   SimulatedArm, make_backend)
from planar_arm_control.motion_planner import MotionPlanner
from planar_arm_control.pid import PID
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.tracking import TrackingController

ARM = PlanarArm([3.0, 2.0, 1.5])
HOME = np.array([np.pi / 2, 0.0, 0.0])


def test_pid_integral_term_is_bounded():
    pid = PID([1.0], [10.0], [0.0], integral_limit=[2.0])
    for _ in range(1000):
        out = pid.update([1.0], [0.0], 0.01)
    # kp * e + clamped integral term
    assert out[0] == pytest.approx(1.0 + 2.0)


def test_pid_output_saturates():
    pid = PID([100.0], [0.0], [0.0], output_limit=[5.0])
    assert pid.update([1.0], [0.0], 0.01)[0] == pytest.approx(5.0)


def test_sim_hard_stops():
    sim = SimulatedArm(ARM.link_lengths, ARM.joint_limits, HOME)
    sim.set_mode(ControlMode.VELOCITY)
    for _ in range(500):
        sim.write(JointCommand(ControlMode.VELOCITY, np.array([2.0, 2.0, 2.0])))
        sim.step(0.01)
    q = sim.read().position
    assert np.all(q <= ARM.joint_limits[:, 1] + 1e-9)


def test_sim_holds_pose_with_gravity_torque():
    q = np.radians([60.0, -30.0, 20.0])
    sim = SimulatedArm(ARM.link_lengths, ARM.joint_limits, q)
    sim.set_mode(ControlMode.EFFORT)
    for _ in range(200):
        sim.write(JointCommand(ControlMode.EFFORT, sim.gravity_torque(sim.q)))
        sim.step(0.01)
    np.testing.assert_allclose(sim.read().position, q, atol=1e-9)


def test_unknown_backend_rejected():
    with pytest.raises(ValueError):
        make_backend("nope")


@pytest.mark.parametrize("mode", ["position", "velocity", "effort"])
def test_closed_loop_tracks_fixed_scenario(mode):
    planner = MotionPlanner(ARM)
    sim = SimulatedArm(ARM.link_lengths, ARM.joint_limits, HOME)
    sim.set_mode(ControlMode(mode))
    tracker = TrackingController(mode)
    q, dt = HOME.copy(), 0.01
    for xy in [(4.0, 2.0), (-3.0, 3.0)]:
        plan = planner.plan(xy, q)
        tracker.reset()
        t, worst = 0.0, 0.0
        while t < plan.trajectory.duration + 0.5:
            q_ref, qd_ref, qdd_ref = plan.trajectory.sample(t)
            meas = sim.read().position
            worst = max(worst, float(np.max(np.abs(q_ref - meas))))
            sim.write(tracker.compute(q_ref, qd_ref, qdd_ref, meas, dt))
            sim.step(dt)
            t += dt
        q = sim.read().position
        ee = np.asarray(ARM.end_effector(q))
        assert np.linalg.norm(ee - np.asarray(xy)) < 5e-3, f"{mode}: settled off target"
        assert np.degrees(worst) < 5.0, f"{mode}: tracking error {np.degrees(worst):.2f} deg"
        assert ARM.arm_above_base(q)
