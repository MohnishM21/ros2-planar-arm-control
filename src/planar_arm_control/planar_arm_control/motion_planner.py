"""
motion_planner.py — Cartesian target -> validated joint-space trajectory.

Wraps the provided PlanarArm (unmodified) and adds what a controller needs
before it is allowed to move:

  1. Workspace check: targets beyond max reach, or below the ground, are
     projected onto the reachable set, and the caller is *told* it happened.
  2. IK + validation: the IK result is re-checked against joint limits, the
     ground constraint and FK error (the library's Jacobian fallback does not
     enforce limits, so we never trust it blindly).
  3. Path validation: endpoints being valid does not make the path valid —
     a joint-space interpolation can sweep a link through the floor. Every
     sample is checked; if the direct path fails we route via a safe "home"
     pose, and if that fails too the motion is rejected.

No ROS imports: unit-testable and reusable by any front end.
"""

import contextlib
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np

from planar_arm_control.trajectory import QuinticTrajectory, TrajectoryChain


@dataclass
class PlanResult:
    ok: bool
    message: str
    requested: np.ndarray
    target: np.ndarray                 # point actually planned to
    projected: bool = False
    q_goal: Optional[np.ndarray] = None
    trajectory: Optional[object] = None
    via_home: bool = False


class MotionPlanner:
    def __init__(self, arm, max_velocity=1.0, max_acceleration=2.0,
                 home_q=(np.pi / 2, 0.0, 0.0), position_tolerance=1e-3,
                 path_samples=200):
        self.arm = arm
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.home_q = np.asarray(home_q, float)
        self.position_tolerance = position_tolerance
        self.path_samples = path_samples
        self.max_reach = float(sum(arm.link_lengths))

    # -- workspace -----------------------------------------------------------
    def project_target(self, xy):
        """Return (reachable_point, was_projected). Mirrors the library's
        radial projection, plus clamps targets below the ground to y = 0."""
        target = np.asarray(xy, float).copy()
        projected = False
        if target[1] < self.arm.min_y:
            target[1] = self.arm.min_y
            projected = True
        dist = float(np.linalg.norm(target))
        if dist > self.max_reach:
            target *= self.max_reach / dist
            projected = True
        return target, projected

    # -- validation ----------------------------------------------------------
    def config_valid(self, q):
        return self.arm.within_joint_limits(q) and self.arm.arm_above_base(q)

    def path_valid(self, trajectory):
        return all(self.config_valid(q)
                   for q in trajectory.sample_positions(self.path_samples))

    # -- planning ------------------------------------------------------------
    def solve_ik(self, xy, q_current):
        requested = np.asarray(xy, float)
        target, projected = self.project_target(requested)
        # The library prints its own diagnostics; we report through ROS logs.
        with contextlib.redirect_stdout(io.StringIO()):
            q_goal = np.asarray(
                self.arm.inverse_kinematics(target, initial_guess=q_current), float)

        error = np.linalg.norm(np.asarray(self.arm.end_effector(q_goal)) - target)
        if not self.config_valid(q_goal) or error > self.position_tolerance:
            return PlanResult(False,
                              f"No valid IK solution for ({target[0]:.3f}, "
                              f"{target[1]:.3f}) within joint limits / above ground",
                              requested, target, projected)
        msg = "ok"
        if projected:
            msg = (f"Target ({requested[0]:.2f}, {requested[1]:.2f}) is outside the "
                   f"workspace; projected to ({target[0]:.3f}, {target[1]:.3f})")
        return PlanResult(True, msg, requested, target, projected, q_goal)

    def plan_joint(self, q_current, q_goal, duration=None):
        """Plan q_current -> q_goal; detour via home if the direct path is unsafe.
        Returns (trajectory or None, via_home)."""
        direct = self._segment(q_current, q_goal, duration)
        if self.path_valid(direct):
            return direct, False
        legs = [self._segment(q_current, self.home_q, None),
                self._segment(self.home_q, q_goal, None)]
        chain = TrajectoryChain(legs)
        if self.path_valid(chain):
            return chain, True
        return None, False

    def plan(self, xy, q_current, duration=None):
        result = self.solve_ik(xy, q_current)
        if not result.ok:
            return result
        traj, via_home = self.plan_joint(q_current, result.q_goal, duration)
        if traj is None:
            result.ok = False
            result.message = "No collision-free joint path to target"
            return result
        result.trajectory = traj
        result.via_home = via_home
        return result

    def _segment(self, q0, q1, duration):
        t_min = QuinticTrajectory.min_duration(q0, q1, self.max_velocity,
                                               self.max_acceleration)
        T = t_min if duration is None or duration <= 0 else max(duration, t_min)
        return QuinticTrajectory(q0, q1, T)
