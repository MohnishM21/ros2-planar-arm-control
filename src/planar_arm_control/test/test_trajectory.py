import numpy as np
import pytest

from planar_arm_control.trajectory import QuinticTrajectory, TrajectoryChain


def test_quintic_boundary_conditions():
    traj = QuinticTrajectory([0.0, 1.0, -1.0], [1.0, 0.0, 0.5], 2.0)
    q0, qd0, qdd0 = traj.sample(0.0)
    q1, qd1, qdd1 = traj.sample(2.0)
    np.testing.assert_allclose(q0, [0.0, 1.0, -1.0])
    np.testing.assert_allclose(q1, [1.0, 0.0, 0.5])
    for v in (qd0, qdd0, qd1, qdd1):
        np.testing.assert_allclose(v, 0.0, atol=1e-12)


def test_quintic_clamps_outside_time_range():
    traj = QuinticTrajectory([0.0], [1.0], 1.0)
    assert traj.sample(-1.0)[0][0] == pytest.approx(0.0)
    assert traj.sample(5.0)[0][0] == pytest.approx(1.0)


def test_min_duration_respects_limits():
    q0, q1 = np.zeros(3), np.array([1.5, -0.5, 0.2])
    vmax, amax = 1.0, 2.0
    T = QuinticTrajectory.min_duration(q0, q1, vmax, amax)
    traj = QuinticTrajectory(q0, q1, T)
    ts = np.linspace(0, T, 2001)
    vel = np.array([traj.sample(t)[1] for t in ts])
    acc = np.array([traj.sample(t)[2] for t in ts])
    assert np.max(np.abs(vel)) <= vmax + 1e-6
    assert np.max(np.abs(acc)) <= amax + 1e-6


def test_quintic_is_monotonic_per_joint():
    # Monotonic paths stay inside the joint-limit box spanned by the endpoints.
    traj = QuinticTrajectory([0.0, 0.0], [1.0, -2.0], 3.0)
    qs = np.array(traj.sample_positions(300))
    assert np.all(np.diff(qs[:, 0]) >= -1e-12)
    assert np.all(np.diff(qs[:, 1]) <= 1e-12)


def test_chain_is_continuous():
    a = QuinticTrajectory([0.0], [1.0], 1.0)
    b = QuinticTrajectory([1.0], [0.5], 2.0)
    chain = TrajectoryChain([a, b])
    assert chain.duration == pytest.approx(3.0)
    assert chain.sample(1.0 - 1e-9)[0][0] == pytest.approx(1.0, abs=1e-6)
    assert chain.sample(1.0 + 1e-9)[0][0] == pytest.approx(1.0, abs=1e-6)
    assert chain.sample(3.0)[0][0] == pytest.approx(0.5)
