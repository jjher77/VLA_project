#!/usr/bin/env python3
"""Doosan data recorder node that logs synchronized observations/actions per episode."""

from __future__ import annotations

import argparse
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import Trigger
import message_filters

CLOCK = rclpy.clock.Clock()


@dataclass
class Sample:
    top_image: np.ndarray
    wrist_image: np.ndarray
    front_image: np.ndarray
    joint_pos: np.ndarray
    gripper_pos: np.ndarray
    timestamp: float


@dataclass
class EpisodeBuffer:
    samples: list[Sample] = field(default_factory=list)
    prompt: str = ""

    def append(self, sample: Sample) -> None:
        self.samples.append(sample)

    def to_dict(self) -> dict[str, Any]:
        if not self.samples:
            raise ValueError("No samples recorded.")
        top_imgs = np.stack([s.top_image for s in self.samples], axis=0)
        wrist_imgs = np.stack([s.wrist_image for s in self.samples], axis=0)
        front_imgs = np.stack([s.front_image for s in self.samples], axis=0)
        joint_pos = np.stack([s.joint_pos for s in self.samples], axis=0)
        gripper_pos = np.stack([s.gripper_pos for s in self.samples], axis=0)
        timestamps = np.array([s.timestamp for s in self.samples], dtype=np.float64)
        actions = joint_pos[1:] - joint_pos[:-1]
        gripper_actions = gripper_pos[1:] - gripper_pos[:-1]
        actions = np.concatenate([actions, gripper_actions], axis=-1)
        return {
            "observations/top_image": top_imgs,
            "observations/wrist_image": wrist_imgs,
            "observations/front_image": front_imgs,
            "observations/joint_position": joint_pos,
            "observations/gripper_position": gripper_pos,
            "actions": actions,
            "prompt": np.asarray(self.prompt, dtype=np.object_),
            "timestamps": timestamps,
        }


class DoosanDataRecorder(Node):
    def __init__(self, args):
        super().__init__("doosan_data_recorder")
        self._args = args
        self._bridge = CvBridge()
        qos = QoSProfile(depth=10)
        self._top_sub = message_filters.Subscriber(self, Image, args.top_topic, qos_profile=qos)
        self._wrist_sub = message_filters.Subscriber(self, Image, args.wrist_topic, qos_profile=qos)
        self._front_sub = message_filters.Subscriber(self, Image, args.front_topic, qos_profile=qos)
        self._arm_sub = message_filters.Subscriber(self, JointState, args.joint_topic, qos_profile=qos)
        self._gripper_sub = message_filters.Subscriber(self, JointState, args.gripper_topic, qos_profile=qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [
                self._top_sub,
                self._wrist_sub,
                self._front_sub,
                self._arm_sub,
                self._gripper_sub,
            ],
            queue_size=10,
            slop=args.sync_slop,
        )
        self._sync.registerCallback(self._sync_cb)
        self._buffer_lock = threading.Lock()
        self._buffer: EpisodeBuffer | None = None
        self._recording = False
        self._latest_sample: Sample | None = None
        self._save_dir = pathlib.Path(args.output_dir).expanduser()
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._start_srv = self.create_service(Trigger, f"{args.namespace}/start", self._start_recording)
        self._stop_srv = self.create_service(Trigger, f"{args.namespace}/stop", self._stop_recording)
        self._prompt_param = args.default_prompt
        self._hz = args.sample_rate
        self._timer = self.create_timer(1.0 / self._hz, self._periodic_append)
        self.get_logger().info(
            f"Recorder ready. Services: {args.namespace}/start, {args.namespace}/stop. Saving to {self._save_dir}"
        )

    def _sync_cb(self, top: Image, wrist: Image, front: Image, joint: JointState, gripper: JointState):
        with self._buffer_lock:
            self._latest_sample = Sample(
                top_image=self._bridge.imgmsg_to_cv2(top, desired_encoding="rgb8"),
                wrist_image=self._bridge.imgmsg_to_cv2(wrist, desired_encoding="rgb8"),
                front_image=self._bridge.imgmsg_to_cv2(front, desired_encoding="rgb8"),
                joint_pos=np.asarray(joint.position[:6], dtype=np.float32),
                gripper_pos=np.asarray(gripper.position, dtype=np.float32),
                timestamp=time.time(),
            )

    def _periodic_append(self):
        with self._buffer_lock:
            if not self._recording or self._buffer is None or self._latest_sample is None:
                return
            self._buffer.append(self._latest_sample)

    def _start_recording(self, request, response):
        with self._buffer_lock:
            if self._recording:
                response.success = False
                response.message = "Already recording."
                return response
            self._buffer = EpisodeBuffer(prompt=self._prompt_param)
            self._recording = True
            response.success = True
            response.message = "Recording started."
            self.get_logger().info("Recording started.")
            return response

    def _stop_recording(self, request, response):
        with self._buffer_lock:
            if not self._recording or self._buffer is None:
                response.success = False
                response.message = "Recorder not running."
                return response
            buffer = self._buffer
            self._buffer = None
            self._recording = False
        try:
            data = buffer.to_dict()
            seq_name = f"seq_{int(time.time())}.npz"
            save_path = self._save_dir / seq_name
            np.savez_compressed(save_path, **data)
            response.success = True
            response.message = f"Saved {save_path}"
            self.get_logger().info(f"Saved episode with {len(buffer.samples)} samples -> {save_path}")
        except Exception as e:
            response.success = False
            response.message = f"Failed to save: {e}"
            self.get_logger().error(response.message)
        return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-topic", default="/camera/top/color/image_raw")
    parser.add_argument("--wrist-topic", default="/camera/wrist/color/image_raw")
    parser.add_argument("--front-topic", default="/camera/front/color/image_raw")
    parser.add_argument("--joint-topic", default="/dsr01/joint_states")
    parser.add_argument("--gripper-topic", default="/joint_states")
    parser.add_argument("--output-dir", default="/tmp/doosan_dataset")
    parser.add_argument("--namespace", default="/doosan_data_recorder")
    parser.add_argument("--default-prompt", default="Clean up the trash on the table.")
    parser.add_argument("--sample-rate", type=float, default=10.0)
    parser.add_argument("--sync-slop", type=float, default=0.05)
    args = parser.parse_args()
    rclpy.init()
    node = DoosanDataRecorder(args)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
