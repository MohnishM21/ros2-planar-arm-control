import numpy as np
import pytest

from planar_arm_control.motion_planner import MotionPlanner
from planar_arm_control.planar_arm import PlanarArm

HOME = np.array([np.pi / 2, 0.0, 0.0])


@pytest.fixture
def planner():
    return MotionPlanner(PlanarArm([3.0, 2.0, 1.5]))


def reaches(planner, result):
    ee = np.asarray(planner.arm.end_effector(result.q_goal))
    return np.linalg.norm(ee - result.target) < 1e-3


def test_fixed_scenario_pick_and_place(planner):
    pick = planner.plan((4.0, 2.0), HOME)
    assert pick.ok and not pick.projected and reaches(planner, pick)
    place = planner.plan((-3.0, 3.0), pick.q_goal)
    assert place.ok and not place.projected and reaches(planner, place)


def test_unreachable_target_is_projected_and_flagged(planner):
    r = planner.plan((7.0, 3.0), HOME)
    assert r.ok and r.projected
    assert np.linalg.norm(r.target) == pytest.approx(6.5)
    # Projection is radial: same direction as the request.
    np.testing.assert_allclose(r.target / np.linalg.norm(r.target),
                               np.array([7.0, 3.0]) / np.linalg.norm([7.0, 3.0]))
    assert reaches(planner, r)


def test_target_below_ground_is_projected(planner):
    target, projected = planner.project_target((2.0, -1.0))
    assert projected and target[1] == 0.0


def test_infeasible_target_is_rejected(planner):
    # Inside the reach circle but unreachable under the joint limits.
    r = planner.plan((0.5, 0.2), HOME)
    assert not r.ok and r.trajectory is None


@pytest.mark.parametrize("xy", [(4.0, 2.0), (-3.0, 3.0), (7.0, 3.0), (2.0, -1.0),
                                (-5.5, 0.5), (5.0, 0.3), (1.0, 4.0)])
def test_every_planned_path_respects_limits_and_ground(planner, xy):
    r = planner.plan(xy, HOME)
    if not r.ok:
        pytest.skip(r.message)
    for q in r.trajectory.sample_positions(400):
        assert planner.arm.within_joint_limits(q)
        assert planner.arm.arm_above_base(q)


def test_unsafe_direct_path_is_routed_via_home(planner):
    # Arm stretched out low and to the right (after the (7, 3) edge case),
    # then asked for (2, 0): the IK picks a folded pose, and interpolating to
    # it directly sweeps the wrist below y = 0. The planner must detour.
    stretched = planner.plan((7.0, 3.0), HOME).q_goal
    direct_ok = planner.path_valid(planner._segment(
        stretched, planner.solve_ik((2.0, 0.0), stretched).q_goal, None))
    assert not direct_ok
    r = planner.plan((2.0, 0.0), stretched)
    assert r.ok and r.via_home
    assert planner.path_valid(r.trajectory)


def test_vertical_approach_is_a_straight_line(planner):
    above = planner.plan((4.0, 2.5), HOME).q_goal
    r = planner.plan_line((4.0, 2.0), above)
    assert r.ok
    for q in r.trajectory.sample_positions(300):
        x, y = planner.arm.end_effector(q)
        assert x == pytest.approx(4.0, abs=1e-3)
        assert 2.0 - 1e-3 <= y <= 2.5 + 1e-3
        assert planner.arm.within_joint_limits(q) and planner.arm.arm_above_base(q)
    np.testing.assert_allclose(planner.arm.end_effector(r.q_goal), (4.0, 2.0), atol=1e-4)


def test_line_respects_velocity_and_acceleration_limits(planner):
    above = planner.plan((-3.0, 3.5), HOME).q_goal
    traj = planner.plan_line((-3.0, 3.0), above).trajectory
    ts = np.linspace(0.0, traj.duration, 1000)
    assert max(np.max(np.abs(traj.sample(t)[1])) for t in ts) <= planner.max_velocity + 1e-6
    # Finite-difference path derivatives: allow a small numerical margin.
    assert max(np.max(np.abs(traj.sample(t)[2])) for t in ts) <= planner.max_acceleration * 1.05


def test_line_leaving_workspace_is_rejected(planner):
    edge = planner.plan((6.0, 1.0), HOME).q_goal
    r = planner.plan_line((7.0, 1.0), edge)
    assert not r.ok and "workspace" in r.message
