"""
bringup.launch.py — controller + telemetry GUI.

    ros2 launch planar_arm_control bringup.launch.py
    ros2 launch planar_arm_control bringup.launch.py control_mode:=effort
    ros2 launch planar_arm_control bringup.launch.py gui:=false   # headless
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    args = [
        DeclareLaunchArgument("control_mode", default_value="position",
                              description="position | velocity | effort"),
        DeclareLaunchArgument("backend", default_value="sim",
                              description="sim | dynamixel"),
        DeclareLaunchArgument("control_rate_hz", default_value="100.0"),
        DeclareLaunchArgument("publish_rate_hz", default_value="50.0"),
        DeclareLaunchArgument("max_joint_velocity", default_value="1.0"),
        DeclareLaunchArgument("max_joint_acceleration", default_value="2.0"),
        DeclareLaunchArgument("sim_noise_std", default_value="0.0"),
        DeclareLaunchArgument("approach_height", default_value="0.5",
                              description="hover height above pick/place; 0 disables"),
        DeclareLaunchArgument("gui", default_value="true"),
    ]
    lc = LaunchConfiguration

    controller = Node(
        package="planar_arm_control",
        executable="controller_node",
        name="controller_node",
        output="screen",
        parameters=[{
            "control_mode": lc("control_mode"),
            "backend": lc("backend"),
            "control_rate_hz": lc("control_rate_hz"),
            "publish_rate_hz": lc("publish_rate_hz"),
            "max_joint_velocity": lc("max_joint_velocity"),
            "max_joint_acceleration": lc("max_joint_acceleration"),
            "sim_noise_std": lc("sim_noise_std"),
            "approach_height": lc("approach_height"),
        }],
    )
    gui = Node(
        package="planar_arm_control",
        executable="gui_node",
        name="gui_node",
        output="screen",
        condition=IfCondition(lc("gui")),
    )
    return LaunchDescription(args + [controller, gui])
