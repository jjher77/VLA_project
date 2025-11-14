#!/usr/bin/env python3
"""ROS 2 node that publishes three external cameras (two RealSense + one wrist cam).

Usage (ROS 2 Humble):
    source /opt/ros/humble/setup.bash
    python3 openpi/examples/ros_multi_camera_publisher.py --ros-args \
        -p wrist_device:=/dev/video2

ROS 2 parameters:
    top_serial      (str)  RealSense serial for the top camera (default: 343122300152)
    front_serial    (str)  RealSense serial for the front camera (default: 343622300813)
    wrist_device    (str)  Video device path for the wrist camera (default: /dev/video2)
    width           (int)  Image width (default: 640)
    height          (int)  Image height (default: 480)
    fps             (float) Publish rate (default: 30.0)
    top_topic       (str)  ROS topic name (default: /camera/top/color/image_raw)
    front_topic     (str)  ROS topic name (default: /camera/front/color/image_raw)
    wrist_topic     (str)  ROS topic name (default: /camera/wrist/color/image_raw)
    top_frame_id    (str)  Frame id for TF (default: camera_top_optical_frame)
    front_frame_id  (str)  Frame id for TF (default: camera_front_optical_frame)
    wrist_frame_id  (str)  Frame id for TF (default: camera_wrist_optical_frame)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pyrealsense2 as rs
import rclpy
try:
    from rcl_interfaces.msg import ParameterDescriptor
except ImportError:  # pragma: no cover - fallback if interface not available
    ParameterDescriptor = None
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


@dataclass
class CameraConfig:
    serial: Optional[str]
    device: Optional[str]
    topic: str
    frame_id: str
    width: int
    height: int
    fps: int


class RealSenseCamera:
    """Wrapper around a RealSense color stream."""

    def __init__(self, cfg: CameraConfig, logger) -> None:
        if not cfg.serial:
            raise ValueError("Missing RealSense serial number.")

        self._cfg = cfg
        self._logger = logger
        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_device(cfg.serial)
        self._config.enable_stream(
            rs.stream.color,
            cfg.width,
            cfg.height,
            rs.format.bgr8,
            cfg.fps,
        )

        self._started = False
        self._start()

    def _start(self) -> None:
        try:
            self._pipeline.start(self._config)
            self._started = True
            self._logger.info(f"[RealSense] Started serial={self._cfg.serial}")
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Failed to start RealSense {self._cfg.serial}: {exc}") from exc

    def read(self) -> Optional[np.ndarray]:
        if not self._started:
            return None
        try:
            frames = self._pipeline.wait_for_frames(timeout_ms=int(1000 / self._cfg.fps))
            frame = frames.get_color_frame()
            if frame:
                return np.asanyarray(frame.get_data())
        except Exception as exc:  # noqa: BLE001
            self._logger.warn(f"[RealSense] serial={self._cfg.serial} read failed: {exc}")
            return None
        return None

    def stop(self) -> None:
        if self._started:
            self._pipeline.stop()
            self._started = False


class OpenCVCamera:
    """Wrapper for generic UVC cameras accessed via OpenCV."""

    def __init__(self, cfg: CameraConfig, logger) -> None:
        if not cfg.device:
            raise ValueError("Missing device path for OpenCV camera.")
        self._cfg = cfg
        self._logger = logger
        self._capture = cv2.VideoCapture(cfg.device, cv2.CAP_V4L2)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera device {cfg.device}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        self._capture.set(cv2.CAP_PROP_FPS, cfg.fps)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._logger.info(f"[OpenCV] Started device={cfg.device}")

    def read(self) -> Optional[np.ndarray]:
        ret, frame = self._capture.read()
        if not ret:
            return None
        return frame

    def stop(self) -> None:
        self._capture.release()


class MultiCameraPublisher(Node):
    """ROS 2 node that publishes all camera streams."""

    def __init__(self) -> None:
        super().__init__("multi_camera_publisher")
        self._bridge = CvBridge()

        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 15.0)
        dyn_desc = ParameterDescriptor(dynamic_typing=True) if ParameterDescriptor else None

        def declare(name: str, default):
            if dyn_desc:
                self.declare_parameter(name, default, dyn_desc)
            else:
                self.declare_parameter(name, default)

        declare("top_serial", "343122300152")
        declare("front_serial", "343622300813")
        declare("wrist_device", "/dev/video2")
        declare("top_topic", "/camera/top/color/image_raw")
        declare("front_topic", "/camera/front/color/image_raw")
        declare("wrist_topic", "/camera/wrist/color/image_raw")
        declare("top_frame_id", "camera_top_optical_frame")
        declare("front_frame_id", "camera_front_optical_frame")
        declare("wrist_frame_id", "camera_wrist_optical_frame")

        width = int(self.get_parameter("width").value)
        height = int(self.get_parameter("height").value)
        fps = float(self.get_parameter("fps").value)

        self._configs = {
            "top": CameraConfig(
                serial=str(self.get_parameter("top_serial").value),
                device=None,
                topic=str(self.get_parameter("top_topic").value),
                frame_id=str(self.get_parameter("top_frame_id").value),
                width=width,
                height=height,
                fps=int(fps),
            ),
            "front": CameraConfig(
                serial=str(self.get_parameter("front_serial").value),
                device=None,
                topic=str(self.get_parameter("front_topic").value),
                frame_id=str(self.get_parameter("front_frame_id").value),
                width=width,
                height=height,
                fps=int(fps),
            ),
            "wrist": CameraConfig(
                serial=None,
                device=str(self.get_parameter("wrist_device").value),
                topic=str(self.get_parameter("wrist_topic").value),
                frame_id=str(self.get_parameter("wrist_frame_id").value),
                width=width,
                height=height,
                fps=int(fps),
            ),
        }

        self._camera_publishers = {
            name: self.create_publisher(Image, cfg.topic, 10)
            for name, cfg in self._configs.items()
        }

        try:
            self._sources = {
                "top": RealSenseCamera(self._configs["top"], self.get_logger()),
                "front": RealSenseCamera(self._configs["front"], self.get_logger()),
                "wrist": OpenCVCamera(self._configs["wrist"], self.get_logger()),
            }
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Camera initialization failed: {exc}")
            raise

        timer_period = 1.0 / fps if fps > 0 else 0.033
        self._timer = self.create_timer(timer_period, self._publish_all)

    def _publish_all(self) -> None:
        for name, source in self._sources.items():
            image = source.read()
            if image is None:
                continue
            cfg = self._configs[name]
            msg = self._bridge.cv2_to_imgmsg(image, encoding="bgr8")
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = cfg.frame_id
            self._camera_publishers[name].publish(msg)

    def destroy_node(self) -> bool:
        for source in self._sources.values():
            source.stop()
        return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = MultiCameraPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
