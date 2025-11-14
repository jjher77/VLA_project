#!/usr/bin/env python3
"""ROS 2 launch file for the multi-camera publisher script."""

from __future__ import annotations

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            Node(
                package="openpi",
                executable="ros_multi_camera_publisher.py",
                name="multi_camera_publisher",
                output="screen",
                parameters=[
                    {
                        "top_serial": "343122300152",
                        "front_serial": "343622300813",
                        "wrist_device": "/dev/video2",
                        "width": 640,
                        "height": 480,
                        "fps": 30.0,
                        "top_topic": "/camera/top/color/image_raw",
                        "front_topic": "/camera/front/color/image_raw",
                        "wrist_topic": "/camera/wrist/color/image_raw",
                    }
                ],
            )
        ]
    )


if __name__ == "__main__":
    generate_launch_description()
