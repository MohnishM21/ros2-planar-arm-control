#!/usr/bin/env python3
"""
controller_node.py — owns the arm, plans motions, closes the control loop.

Interfaces
    Publishes   /joint_states   sensor_msgs/JointState         (publish_rate_hz)
                /arm_status     planar_arm_interfaces/ArmStatus (publish_rate_hz)
    Services    /move_to_target planar_arm_interfaces/MoveToTarget
                /home           std_srvs/Trigger
                /stop           std_srvs/Trigger  (freeze in place, abort goals)
    Action      /pick_place     planar_arm_interfaces/PickPlace (feedback + cancel)
    Parameters  control_mode (position|velocity|effort, switchable while idle),
                backend, control_rate_hz, publish_rate_hz, max_joint_velocity,
                max_joint_acceleration, approach_height, gripper_dwell_s,
                sim_noise_std, ...

Threading
    MultiThreadedExecutor with three callback groups:
      control  (mutually exclusive) — the fixed-rate control loop, never blocked
      publish  (mutually exclusive) — telemetry at its own, lower rate
      commands (reentrant)          — services + action; the action's execute
                                      callback sleeps while motions run, so it
                                      must not share a thread with the loop.
    Shared state is guarded by one lock and held only for short snapshots.

Data flow per control tick
    trajectory.sample(t) -> TrackingController.compute() -> backend.write()
    backend.step(dt) (sim only) -> backend.read() -> next tick
"""

from collections import deque
import threading
import time

from geometry_msgs.msg import Point
import numpy as np
from rcl_interfaces.msg import SetParametersResult
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import (MutuallyExclusiveCallbackGroup,
                                   ReentrantCallbackGroup)
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger

from planar_arm_control.hardware_interface import ControlMode, make_backend
from planar_arm_control.motion_planner import MotionPlanner
# The provided kinematics library — do not modify it.
from planar_arm_control.planar_arm import PlanarArm
from planar_arm_control.planning_worker import PlanningWorker
from planar_arm_control.tracking import TrackingController
from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ArmStatus
from planar_arm_interfaces.srv import MoveToTarget

LINK_LENGTHS = [3.0, 2.0, 1.5]
JOINT_NAMES = ["joint1", "joint2", "joint3"]
GRASP_RADIUS = 0.05     # end-effector must be this close to the object to grasp


def point(xy):
    return Point(x=float(xy[0]), y=float(xy[1]), z=0.0)


class ControllerNode(Node):
    def __init__(self):
        super().__init__("controller_node")
        self.arm = PlanarArm(LINK_LENGTHS)

        # -- parameters ------------------------------------------------------
        self.declare_parameter("control_mode", "position")
        self.declare_parameter("backend", "sim")
        self.declare_parameter("control_rate_hz", 100.0)
        self.declare_parameter("publish_rate_hz", 50.0)
        self.declare_parameter("max_joint_velocity", 1.0)       # rad/s
        self.declare_parameter("max_joint_acceleration", 2.0)   # rad/s^2
        self.declare_parameter("initial_positions", [np.pi / 2, 0.0, 0.0])
        self.declare_parameter("gripper_dwell_s", 0.5)
        self.declare_parameter("approach_height", 0.5)          # 0 disables
        self.declare_parameter("settle_tolerance", 0.01)        # m, EE "in position"
        self.declare_parameter("settle_timeout_s", 2.0)
        self.declare_parameter("sim_noise_std", 0.0)            # rad

        gp = lambda name: self.get_parameter(name).value  # noqa: E731
        self.mode = ControlMode(gp("control_mode"))
        self.control_period = 1.0 / gp("control_rate_hz")
        q0 = np.asarray(gp("initial_positions"), float)

        # -- core objects (ROS-free) -----------------------------------------
        # Local planner for cheap queries (home pose); all IK-heavy planning
        # goes through a worker process so it can't stall the control loop.
        self.planner = MotionPlanner(self.arm)
        self.worker = PlanningWorker(LINK_LENGTHS, gp("max_joint_velocity"),
                                     gp("max_joint_acceleration"))
        backend_kwargs = {}
        if gp("backend") == "sim":
            backend_kwargs = {"link_lengths": LINK_LENGTHS,
                              "joint_limits": self.arm.joint_limits, "q0": q0,
                              "noise_std": gp("sim_noise_std")}
        self.backend = make_backend(gp("backend"), **backend_kwargs)
        self.backend.set_mode(self.mode)
        self.tracker = TrackingController(self.mode)

        # -- shared state (guarded by self.lock) -----------------------------
        self.lock = threading.Lock()
        self.trajectory = None
        self.traj_start = 0.0
        self.q_ref = q0.copy()
        self.qd_ref = np.zeros(3)
        self.qdd_ref = np.zeros(3)
        self.reading = self.backend.read()
        self.state = "IDLE"
        self.phase = ""
        self.message = "ready"
        self.requested_target = np.array(self.arm.end_effector(q0))
        self.active_target = self.requested_target.copy()
        self.target_projected = False
        self.gripper_closed = False
        self.object_pos = None
        self.object_attached = False
        self.action_active = False
        self.abort_requested = False
        self.periods = deque(maxlen=500)
        self.last_tick = None

        # -- ROS interfaces --------------------------------------------------
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.publish_group = MutuallyExclusiveCallbackGroup()
        self.command_group = ReentrantCallbackGroup()

        self.joint_pub = self.create_publisher(JointState, "joint_states", 10)
        self.status_pub = self.create_publisher(ArmStatus, "arm_status", 10)

        self.create_service(MoveToTarget, "move_to_target", self.on_move_to_target,
                            callback_group=self.command_group)
        self.create_service(Trigger, "home", self.on_home,
                            callback_group=self.command_group)
        self.create_service(Trigger, "stop", self.on_stop,
                            callback_group=self.command_group)
        self.action_server = ActionServer(
            self, PickPlace, "pick_place",
            execute_callback=self.execute_pick_place,
            goal_callback=self.on_pick_place_goal,
            cancel_callback=lambda _: CancelResponse.ACCEPT,
            callback_group=self.command_group)

        self.create_timer(self.control_period, self.control_loop,
                          callback_group=self.control_group)
        self.create_timer(1.0 / gp("publish_rate_hz"), self.publish_telemetry,
                          callback_group=self.publish_group)
        self.add_on_set_parameters_callback(self.on_parameters)

        self.get_logger().info(
            f"controller_node up: backend={gp('backend')} mode={self.mode.value} "
            f"control={gp('control_rate_hz'):.0f} Hz publish={gp('publish_rate_hz'):.0f} Hz")

    # ======================================================================
    # Control loop (fixed rate, own callback group)
    # ======================================================================
    def control_loop(self):
        now = time.monotonic()
        dt = self.control_period if self.last_tick is None else now - self.last_tick
        self.last_tick = now
        self.periods.append(dt)
        # Integrate with the real elapsed time, but never take a huge step
        # after a scheduling hiccup.
        dt = min(dt, 3.0 * self.control_period)

        reading = self.backend.read()
        with self.lock:
            if self.trajectory is not None:
                t = now - self.traj_start
                self.q_ref, self.qd_ref, self.qdd_ref = self.trajectory.sample(t)
                if self.trajectory.finished(t):
                    self.trajectory = None
                    self.qd_ref = np.zeros(3)
                    self.qdd_ref = np.zeros(3)
                    self.state = "IDLE"
            q_ref, qd_ref, qdd_ref = self.q_ref, self.qd_ref, self.qdd_ref
            self.reading = reading
            if self.object_attached:
                self.object_pos = np.array(self.arm.end_effector(reading.position))
            tracker = self.tracker

        command = tracker.compute(q_ref, qd_ref, qdd_ref, reading.position, dt)
        self.backend.write(command)
        self.backend.step(dt)

    # ======================================================================
    # Telemetry (own rate, own callback group)
    # ======================================================================
    def publish_telemetry(self):
        with self.lock:
            r = self.reading
            q_ref = self.q_ref.copy()
            msg = ArmStatus()
            msg.state = self.state
            msg.control_mode = self.mode.value
            msg.phase = self.phase
            msg.message = self.message
            msg.requested_target = point(self.requested_target)
            msg.active_target = point(self.active_target)
            msg.target_projected = self.target_projected
            msg.gripper_closed = self.gripper_closed
            msg.has_object = self.object_pos is not None
            if self.object_pos is not None:
                msg.object_position = point(self.object_pos)
            if self.trajectory is not None:
                msg.trajectory_progress = min(
                    1.0, (time.monotonic() - self.traj_start) / self.trajectory.duration)
            else:
                msg.trajectory_progress = 1.0
            periods = np.array(self.periods) if self.periods else np.array([0.0])

        stamp = self.get_clock().now().to_msg()
        js = JointState()
        js.header.stamp = stamp
        js.name = JOINT_NAMES
        js.position = r.position.tolist()
        js.velocity = r.velocity.tolist()
        js.effort = r.effort.tolist()
        self.joint_pub.publish(js)

        msg.header.stamp = stamp
        msg.desired_position = q_ref.tolist()
        msg.tracking_error = (q_ref - r.position).tolist()
        msg.end_effector = point(self.arm.end_effector(r.position))
        msg.control_period_mean_ms = float(np.mean(periods) * 1e3)
        msg.control_jitter_max_ms = float(
            np.max(np.abs(periods - self.control_period)) * 1e3)
        self.status_pub.publish(msg)

    # ======================================================================
    # Motion primitives (used by services and the action)
    # ======================================================================
    def _current_q(self):
        with self.lock:
            return self.q_ref.copy()

    def _start(self, trajectory):
        """Hand a trajectory to the control loop."""
        with self.lock:
            self.tracker.reset()
            self.trajectory = trajectory
            self.traj_start = time.monotonic()
            self.state = "MOVING"

    def _plan_cartesian(self, xy, duration=None):
        result = self.worker.plan(xy, self._current_q(), duration)
        with self.lock:
            self.requested_target = result.requested
            self.active_target = result.target
            self.target_projected = result.projected
            self.message = result.message
        if result.projected:
            self.get_logger().warn(result.message)
        if not result.ok:
            self.get_logger().error(result.message)
        elif result.via_home:
            self.get_logger().info("Direct path crosses the ground; routing via home.")
        return result

    def _busy(self):
        with self.lock:
            return self.action_active or self.state == "MOVING"

    def _wait_for_motion(self, goal_handle, on_progress):
        """
        Block (in a command-group thread) until the active motion ends.

        Returns 'done', 'canceled' or 'aborted'.
        """
        while rclpy.ok():
            with self.lock:
                moving = self.trajectory is not None
                abort = self.abort_requested
                progress = (1.0 if self.trajectory is None else min(
                    1.0, (time.monotonic() - self.traj_start) / self.trajectory.duration))
            if abort:
                return "aborted"
            if goal_handle.is_cancel_requested:
                self._freeze("pick-and-place canceled")
                return "canceled"
            on_progress(progress)
            if not moving:
                return self._wait_settled(goal_handle)
            time.sleep(0.05)
        return "aborted"

    def _wait_settled(self, goal_handle):
        """
        Wait until the measured arm is "in position" at the end of a motion.

        The trajectory finishing on the clock does not mean the arm is there:
        in effort mode the tracking lag at the end of a short, brisk move was
        ~0.1 m. Closing the gripper before settling would miss the part. This
        is the equivalent of polling a servo's "Moving" / in-position flag.
        """
        tolerance = self.get_parameter("settle_tolerance").value
        deadline = time.monotonic() + self.get_parameter("settle_timeout_s").value
        while rclpy.ok():
            with self.lock:
                reading, q_ref = self.reading, self.q_ref.copy()
                abort = self.abort_requested
            if abort:
                return "aborted"
            if goal_handle.is_cancel_requested:
                self._freeze("pick-and-place canceled")
                return "canceled"
            error = np.linalg.norm(np.subtract(self.arm.end_effector(reading.position),
                                               self.arm.end_effector(q_ref)))
            if error < tolerance and np.max(np.abs(reading.velocity)) < 0.05:
                return "done"
            if time.monotonic() > deadline:
                self.get_logger().warn(f"arm did not settle (error {error:.3f} m); continuing")
                return "done"
            time.sleep(0.02)
        return "aborted"

    def _freeze(self, reason):
        """Stop where we are: the reference snaps to the measured position."""
        with self.lock:
            self.trajectory = None
            self.q_ref = self.reading.position.copy()
            self.qd_ref = np.zeros(3)
            self.qdd_ref = np.zeros(3)
            self.state = "STOPPED"
            self.message = reason
            self.tracker.reset()
        self.get_logger().warn(reason)

    # ======================================================================
    # Services
    # ======================================================================
    def on_move_to_target(self, request, response):
        if self._busy():
            response.accepted = False
            response.message = "Arm is busy; stop the current motion first."
            return response
        result = self._plan_cartesian((request.x, request.y), request.duration)
        response.reachable = not result.projected
        response.planned_target = point(result.target)
        response.message = result.message
        response.accepted = result.ok
        if result.ok:
            response.goal_positions = result.q_goal.tolist()
            response.duration = result.trajectory.duration
            with self.lock:
                self.abort_requested = False
            self._start(result.trajectory)
        return response

    def on_home(self, _request, response):
        if self._busy():
            response.success = False
            response.message = "Arm is busy; stop the current motion first."
            return response
        traj, _ = self.worker.plan_joint(self._current_q(), self.planner.home_q)
        if traj is None:
            response.success, response.message = False, "No safe path to home"
            return response
        with self.lock:
            self.abort_requested = False
            self.requested_target = np.array(self.arm.end_effector(self.planner.home_q))
            self.active_target = self.requested_target.copy()
            self.target_projected = False
            self.message = "homing"
        self._start(traj)
        response.success, response.message = True, f"Homing ({traj.duration:.1f} s)"
        return response

    def on_stop(self, _request, response):
        with self.lock:
            self.abort_requested = True
        self._freeze("STOP requested by operator")
        response.success, response.message = True, "Stopped"
        return response

    def on_parameters(self, params):
        for p in params:
            if p.name == "control_mode":
                try:
                    mode = ControlMode(p.value)
                except ValueError:
                    return SetParametersResult(
                        successful=False, reason="mode must be position|velocity|effort")
                if self._busy():
                    return SetParametersResult(
                        successful=False, reason="can only switch mode while idle")
                with self.lock:
                    # Bumpless transfer: hold the measured pose in the new mode.
                    self.q_ref = self.reading.position.copy()
                    self.mode = mode
                    self.tracker = TrackingController(mode)
                    self.backend.set_mode(mode)
                    self.message = f"control mode -> {mode.value}"
                self.get_logger().info(f"Switched control mode to {mode.value}")
            elif p.name in ("backend", "control_rate_hz", "publish_rate_hz",
                            "initial_positions"):
                return SetParametersResult(
                    successful=False, reason=f"{p.name} is fixed at startup")
            elif p.name == "max_joint_velocity":
                self.worker.max_velocity = float(p.value)
            elif p.name == "max_joint_acceleration":
                self.worker.max_acceleration = float(p.value)
        return SetParametersResult(successful=True)

    # ======================================================================
    # Pick-and-place action
    # ======================================================================
    def on_pick_place_goal(self, _goal):
        if self._busy():
            self.get_logger().warn("Rejecting pick-and-place: arm busy")
            return GoalResponse.REJECT
        return GoalResponse.ACCEPT

    def _approach_point(self, xy):
        """Hover point above xy, or None if approach moves are off or infeasible."""
        height = self.get_parameter("approach_height").value
        if height <= 0.0:
            return None
        above = np.asarray(xy, float) + np.array([0.0, height])
        check = self.worker.solve_ik(above, self._current_q())
        if not check.ok or check.projected:
            self.get_logger().warn(
                f"approach point ({above[0]:.2f}, {above[1]:.2f}) unreachable; "
                "moving to the target directly")
            return None
        return above

    def _plan_line(self, xy):
        """Straight-line Cartesian move; falls back to a joint-space move."""
        result = self.worker.plan_line(xy, self._current_q())
        if result.ok:
            with self.lock:
                self.requested_target = result.requested
                self.active_target = result.target
                self.target_projected = False
            return result
        self.get_logger().warn(f"straight-line move failed ({result.message}); "
                               "using a joint-space move")
        return self._plan_cartesian(xy)

    def execute_pick_place(self, goal_handle):
        goal = goal_handle.request
        pick = np.array([goal.pick.x, goal.pick.y])
        place = np.array([goal.place.x, goal.place.y])
        result = PickPlace.Result()
        feedback = PickPlace.Feedback()

        def finish(success, message, status):
            with self.lock:
                self.action_active = False
                self.phase = ""
                self.message = message
            getattr(goal_handle, status)()
            result.success, result.message = success, message
            # rclpy forbids one call site logging at varying severities.
            if success:
                self.get_logger().info(f"pick-and-place: {message}")
            else:
                self.get_logger().warn(f"pick-and-place: {message}")
            return result

        # Validate both targets before moving: picking at a *projected* point
        # would grasp thin air, so an unreachable pick/place is an error here
        # (unlike a plain move, which goes to the closest reachable point).
        pick_check = self.worker.solve_ik(pick, self._current_q())
        checks = [("pick", pick_check)]
        if pick_check.ok:
            checks.append(("place", self.worker.solve_ik(place, pick_check.q_goal)))
        for name, check in checks:
            if not check.ok or check.projected:
                with self.lock:
                    self.requested_target = check.requested
                    self.active_target = check.target
                    self.target_projected = check.projected
                return finish(False, f"{name} target unreachable. {check.message}",
                              "abort")

        # The sequence as data: (phase, action, argument). With an approach
        # height the tool hovers above each point and moves in and out along
        # a straight vertical line, so it never sweeps sideways into the part.
        pick_above, place_above = self._approach_point(pick), self._approach_point(place)
        steps = []
        if pick_above is not None:
            steps += [("MOVE_TO_PICK", "move", pick_above),
                      ("DESCEND_TO_PICK", "line", pick),
                      ("PICK", "grip", True),
                      ("LIFT", "line", pick_above)]
        else:
            steps += [("MOVE_TO_PICK", "move", pick), ("PICK", "grip", True)]
        if place_above is not None:
            steps += [("MOVE_TO_PLACE", "move", place_above),
                      ("DESCEND_TO_PLACE", "line", place),
                      ("PLACE", "grip", False),
                      ("RETREAT", "line", place_above)]
        else:
            steps += [("MOVE_TO_PLACE", "move", place), ("PLACE", "grip", False)]
        if goal.return_home:
            steps.append(("RETURN_HOME", "home", None))

        def report(index, phase_progress):
            with self.lock:
                self.phase = steps[index][0]
                ee = self.arm.end_effector(self.reading.position)
            feedback.phase = steps[index][0]
            feedback.progress = (index + phase_progress) / len(steps)
            feedback.end_effector = point(ee)
            goal_handle.publish_feedback(feedback)

        with self.lock:
            self.action_active = True
            self.abort_requested = False
            self.object_pos = pick.copy()      # spawn the simulated payload
            self.object_attached = False
            self.gripper_closed = False

        for i, (phase, kind, arg) in enumerate(steps):
            report(i, 0.0)
            if kind == "grip":
                closing = arg
                with self.lock:
                    self.gripper_closed = closing
                    ee = np.array(self.arm.end_effector(self.reading.position))
                    grasped = (closing and self.object_pos is not None
                               and np.linalg.norm(ee - self.object_pos) < GRASP_RADIUS)
                    self.object_attached = grasped
                if closing and not grasped:
                    return finish(False, "grasp failed: object not under gripper", "abort")
                time.sleep(self.get_parameter("gripper_dwell_s").value)
                if goal_handle.is_cancel_requested:
                    self._freeze("pick-and-place canceled")
                    return finish(False, "canceled by operator", "canceled")
            else:
                if kind == "home":
                    traj, _ = self.worker.plan_joint(self._current_q(),
                                                     self.planner.home_q)
                    ok, why = traj is not None, "no safe path home"
                else:
                    plan = self._plan_line(arg) if kind == "line" else self._plan_cartesian(arg)
                    traj, ok, why = plan.trajectory, plan.ok, plan.message
                if not ok:
                    return finish(False, f"{phase} failed: {why}", "abort")
                self._start(traj)
                outcome = self._wait_for_motion(goal_handle, lambda p: report(i, p))
                if outcome == "canceled":
                    return finish(False, "canceled by operator", "canceled")
                if outcome == "aborted":
                    return finish(False, "aborted by STOP", "abort")
            report(i, 1.0)

        return finish(True, f"placed object at ({place[0]:.2f}, {place[1]:.2f})",
                      "succeed")

    def destroy_node(self):
        self.backend.disable()
        self.backend.close()
        self.worker.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
