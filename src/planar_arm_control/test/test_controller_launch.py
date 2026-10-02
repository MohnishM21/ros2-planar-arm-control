"""
Integration test: launch the real controller (no GUI) and drive it over ROS.

Run with:  colcon test --packages-select planar_arm_control
"""

import time
import unittest

import launch
import launch_ros.actions
import launch_testing.actions
import pytest
import rclpy
from rclpy.action import ActionClient

from planar_arm_interfaces.action import PickPlace
from planar_arm_interfaces.msg import ArmStatus
from planar_arm_interfaces.srv import MoveToTarget


@pytest.mark.launch_test
def generate_test_description():
    controller = launch_ros.actions.Node(
        package="planar_arm_control", executable="controller_node",
        parameters=[{"control_mode": "effort", "gripper_dwell_s": 0.2,
                     "max_joint_velocity": 2.0, "max_joint_acceleration": 4.0}])
    return launch.LaunchDescription([controller, launch_testing.actions.ReadyToTest()])


class TestController(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rclpy.init()
        cls.node = rclpy.create_node("controller_test_client")

    @classmethod
    def tearDownClass(cls):
        cls.node.destroy_node()
        rclpy.shutdown()

    def wait(self, future, timeout):
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=timeout)
        self.assertTrue(future.done(), "timed out")
        return future.result()

    def run_pick_place(self, pick, place, timeout=40.0):
        client = ActionClient(self.node, PickPlace, "pick_place")
        self.assertTrue(client.wait_for_server(timeout_sec=10.0))
        goal = PickPlace.Goal()
        goal.pick.x, goal.pick.y = pick
        goal.place.x, goal.place.y = place
        handle = self.wait(client.send_goal_async(goal), 5.0)
        self.assertTrue(handle.accepted)
        return self.wait(handle.get_result_async(), timeout).result

    def latest_status(self):
        msgs = []
        sub = self.node.create_subscription(ArmStatus, "arm_status", msgs.append, 10)
        end = time.time() + 3.0
        while not msgs and time.time() < end:
            rclpy.spin_once(self.node, timeout_sec=0.1)
        self.node.destroy_subscription(sub)
        self.assertTrue(msgs, "no /arm_status received")
        return msgs[-1]

    def test_1_fixed_scenario(self):
        result = self.run_pick_place((4.0, 2.0), (-3.0, 3.0))
        self.assertTrue(result.success, result.message)
        st = self.latest_status()
        # Object left at the place point; tool retreated 0.5 straight above it.
        self.assertAlmostEqual(st.object_position.x, -3.0, delta=0.01)
        self.assertAlmostEqual(st.object_position.y, 3.0, delta=0.01)
        self.assertAlmostEqual(st.end_effector.x, -3.0, delta=0.01)
        self.assertAlmostEqual(st.end_effector.y, 3.5, delta=0.01)
        self.assertFalse(st.gripper_closed)

    def test_2_unreachable_pick_is_aborted(self):
        result = self.run_pick_place((7.0, 3.0), (-3.0, 3.0), timeout=10.0)
        self.assertFalse(result.success)
        self.assertIn("unreachable", result.message)

    def test_3_unreachable_move_is_projected(self):
        client = self.node.create_client(MoveToTarget, "move_to_target")
        self.assertTrue(client.wait_for_service(timeout_sec=5.0))
        resp = self.wait(client.call_async(MoveToTarget.Request(x=7.0, y=3.0)), 5.0)
        self.assertTrue(resp.accepted)
        self.assertFalse(resp.reachable)
        self.assertAlmostEqual((resp.planned_target.x ** 2 + resp.planned_target.y ** 2) ** 0.5,
                               6.5, places=3)
        self.assertTrue(self.latest_status().target_projected)
