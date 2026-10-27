#!/usr/bin/env python3

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import AppendEnvironmentVariable
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    turtlebot3_gazebo_dir = get_package_share_directory("turtlebot3_gazebo")
    ros_gz_sim_dir = get_package_share_directory("ros_gz_sim")
    turtlebot3_model = "waffle_pi"

    use_sim_time = LaunchConfiguration("use_sim_time")

    world = os.path.join(turtlebot3_gazebo_dir, "worlds", "empty_world.world")
    sdf_path = os.path.join(
        turtlebot3_gazebo_dir,
        "models",
        f"turtlebot3_{turtlebot3_model}",
        "model.sdf",
    )
    urdf_path = os.path.join(
        turtlebot3_gazebo_dir,
        "urdf",
        f"turtlebot3_{turtlebot3_model}.urdf",
    )

    with open(urdf_path, "r", encoding="utf-8") as urdf_file:
        robot_description = urdf_file.read()

    set_gz_resource_path = AppendEnvironmentVariable(
        "GZ_SIM_RESOURCE_PATH",
        os.path.join(turtlebot3_gazebo_dir, "models"),
    )

    gz_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": ["-r -s -v2 ", world],
            "on_exit_shutdown": "true",
        }.items(),
    )

    gz_client = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ros_gz_sim_dir, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": "-g -v2 "}.items(),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[
            {
                "use_sim_time": use_sim_time,
                "robot_description": robot_description,
            }
        ],
    )

    spawn_turtlebot = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-name",
            turtlebot3_model,
            "-file",
            sdf_path,
            "-x",
            "0.0",
            "-y",
            "0.0",
            "-z",
            "0.01",
        ],
        output="screen",
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "cmd_vel@geometry_msgs/msg/TwistStamped]gz.msgs.Twist",
            "odom@nav_msgs/msg/Odometry[gz.msgs.Odometry",
            "scan@sensor_msgs/msg/LaserScan[gz.msgs.LaserScan",
            "imu@sensor_msgs/msg/Imu[gz.msgs.IMU",
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "use_sim_time",
                default_value="true",
                description="Use simulation clock if true.",
            ),
            set_gz_resource_path,
            gz_server,
            gz_client,
            robot_state_publisher,
            spawn_turtlebot,
            bridge,
        ]
    )