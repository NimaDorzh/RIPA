#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    video_device = LaunchConfiguration("video_device")

    usb_camera_node = Node(
        package="usb_cam",
        executable="usb_cam_node_exe",
        name="usb_cam",
        output="screen",
        parameters=[
            {
                "video_device": video_device,
                "image_width": 1280,
                "image_height": 720,
                "pixel_format": "mjpeg",
                "camera_frame_id": "camera",
                "io_method": "mmap",
            }
        ],
        remappings=[
            ("image_raw", "/camera/image_raw"),
        ],
    )

    ocr_node = Node(
        package="llm_robot_controller",
        executable="ocr_node",
        name="ocr_node",
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "video_device",
                default_value="/dev/video0",
                description="USB camera device path.",
            ),
            usb_camera_node,
            ocr_node,
        ]
    )