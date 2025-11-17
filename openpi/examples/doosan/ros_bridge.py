#!/usr/bin/env python3
"""ROS2 bridge that streams Doosan observations to an OpenPI policy server."""

from __future__ import annotations

import argparse
import threading
from collections import deque

import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState

from dsr_msgs2.srv import MoveJoint

from openpi_client.websocket_client_policy import WebsocketClientPolicy


def _to_numpy(msg: Image, bridge: CvBridge) -> np.ndarray:
    cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
    return np.asarray(cv_img)


class DoosanPolicyBridge(Node):
    def __init__(self, args):
        super().__init__("doosan_policy_bridge")
        self._args = args
        self._bridge = CvBridge()
        self._policy = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)

        self._lock = threading.Lock()
        self._top_image: np.ndarray | None = None
        self._wrist_image: np.ndarray | None = None
        self._front_image: np.ndarray | None = None
        self._joint_state: np.ndarray | None = None
        self._gripper: float = 0.0

        qos = rclpy.qos.QoSProfile(depth=1)
        self.create_subscription(Image, args.top_topic, self._top_cb, qos)
        self.create_subscription(Image, args.wrist_topic, self._wrist_cb, qos)
        self.create_subscription(Image, args.front_topic, self._front_cb, qos)
        self.create_subscription(JointState, args.joint_topic, self._joint_cb, qos)

        self._move_joint_cli = self.create_client(MoveJoint, f"/{args.robot_id}/motion/move_joint")
        while not self._move_joint_cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /%s/motion/move_joint ..." % args.robot_id)
        self._pending_motion = None

        self._log_buffer = deque(maxlen=10)
        self.create_timer(args.publish_period, self._tick)
        self.get_logger().info(
            f"Streaming to policy server ws://{args.policy_host}:{args.policy_port} "
            f"with prompt '{args.prompt}'"
        )

    # --- ROS callbacks ----------------------------------------------------- #
    def _top_cb(self, msg: Image):
        with self._lock:
            self._top_image = _to_numpy(msg, self._bridge)

    def _wrist_cb(self, msg: Image):
        with self._lock:
            self._wrist_image = _to_numpy(msg, self._bridge)

    def _front_cb(self, msg: Image):
        with self._lock:
            self._front_image = _to_numpy(msg, self._bridge)

    def _joint_cb(self, msg: JointState):
        with self._lock:
            positions = np.asarray(msg.position, dtype=np.float32)
            if positions.size == 0:
                return
            self._joint_state = positions[:6]
            if positions.size > 6:
                self._gripper = float(positions[6])

    # --- Policy loop ------------------------------------------------------- #
    def _tick(self):
        with self._lock:
            if any(
                data is None
                for data in (self._top_image, self._wrist_image, self._front_image, self._joint_state)
            ):
                return
            obs = {
                "observation/top_image": self._top_image.copy(),
                "observation/wrist_image": self._wrist_image.copy(),
                "observation/front_image": self._front_image.copy(),
                "observation/joint_position": self._joint_state.copy(),
                "observation/gripper_position": np.array([self._gripper], dtype=np.float32),
                "prompt": self._args.prompt,
            }
        result = self._policy.infer(obs)
        actions = result["actions"].squeeze()
        self._log_buffer.append(actions.tolist())
        if len(self._log_buffer) == self._log_buffer.maxlen:
            self.get_logger().info(f"Latest policy actions: {self._log_buffer[-1]}")
            self._log_buffer.clear()
        self._send_motion(actions)

    def _send_motion(self, actions: np.ndarray):
        if self._pending_motion is not None and not self._pending_motion.done():
            return

        actions = np.asarray(actions)
        if actions.ndim == 1:
            seq = actions
        elif actions.ndim == 2:
            seq = actions[0]
        elif actions.ndim == 3:
            seq = actions[0, 0]
        else:
            self.get_logger().warn("Unexpected actions shape: %s", actions.shape)
            return

        # Policy deltas are in radians; convert to degrees for Doosan MoveJoint.
        rad_to_deg = 180.0 / np.pi
        delta = seq[:6] * self._args.delta_scale * rad_to_deg
        req = MoveJoint.Request()
        req.pos = delta.tolist()
        req.vel = self._args.move_vel
        req.acc = self._args.move_acc
        req.time = self._args.move_time
        req.mode = 1  # relative move
        req.blend_type = 0
        req.sync_type = 0

        self._pending_motion = self._move_joint_cli.call_async(req)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--prompt", default="청소해 줘")
    parser.add_argument("--publish-period", type=float, default=0.2)
    parser.add_argument("--robot-id", default="dsr01")
    parser.add_argument("--top-topic", default="/camera/top/color/image_raw")
    parser.add_argument("--wrist-topic", default="/camera/wrist/color/image_raw")
    parser.add_argument("--front-topic", default="/camera/front/color/image_raw")
    parser.add_argument("--joint-topic", default="/dsr01/joint_states")
    parser.add_argument("--delta-scale", type=float, default=0.05, help="Scale factor applied to policy deltas.")
    parser.add_argument("--move-vel", type=float, default=60.0)
    parser.add_argument("--move-acc", type=float, default=60.0)
    parser.add_argument("--move-time", type=float, default=0.0)
    args = parser.parse_args()

    rclpy.init()
    bridge = DoosanPolicyBridge(args)
    try:
        rclpy.spin(bridge)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
