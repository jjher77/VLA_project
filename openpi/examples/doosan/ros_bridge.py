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

from dsr_msgs2.msg import ServojRtStream
from dsr_msgs2.srv import MoveJoint

from openpi_client.websocket_client_policy import WebsocketClientPolicy

try:
    from dsr_example2.dsr_example.dsr_example.simple.adaptive_gripper_drl import GripperController
except ImportError:  # pragma: no cover - optional dependency
    GripperController = None


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
        self._gripper_controller: GripperController | None = None
        self._last_gripper_stroke: int | None = None

        qos = rclpy.qos.QoSProfile(depth=1)
        self.create_subscription(Image, args.top_topic, self._top_cb, qos)
        self.create_subscription(Image, args.wrist_topic, self._wrist_cb, qos)
        self.create_subscription(Image, args.front_topic, self._front_cb, qos)
        self.create_subscription(JointState, args.joint_topic, self._joint_cb, qos)

        self._pending_motion = None
        self._move_joint_cli = None
        self._servoj_pub = None
        if args.motion_interface == "move_joint":
            self._move_joint_cli = self.create_client(MoveJoint, f"/{args.robot_id}/motion/move_joint")
            while not self._move_joint_cli.wait_for_service(timeout_sec=1.0):
                self.get_logger().info("Waiting for /%s/motion/move_joint ..." % args.robot_id)
        else:
            self._servoj_pub = self.create_publisher(ServojRtStream, f"/{args.robot_id}/servoj_rt_stream", 10)

        if args.enable_gripper:
            if GripperController is None:
                self.get_logger().error(
                    "Gripper control was requested, but adaptive_gripper_drl.py could not be imported."
                )
            else:
                try:
                    self._gripper_controller = GripperController(self, namespace=args.robot_id)
                    if args.gripper_init:
                        self._gripper_controller.initialize()
                except Exception as exc:  # pragma: no cover - ROS runtime failure
                    self.get_logger().error(f"Failed to initialize gripper controller: {exc}")
                    self._gripper_controller = None

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
            joint_state = self._joint_state.copy()
        result = self._policy.infer(obs)
        actions = result["actions"].squeeze()
        self._log_buffer.append(actions.tolist())
        if len(self._log_buffer) == self._log_buffer.maxlen:
            self.get_logger().info(f"Latest policy actions: {self._log_buffer[-1]}")
            self._log_buffer.clear()
        self._send_motion(actions, joint_state)

    def _send_motion(self, actions: np.ndarray, joint_state: np.ndarray):
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

        if self._args.motion_interface == "move_joint":
            self._send_move_joint(seq)
        else:
            self._send_servoj(seq, joint_state)

    def _send_move_joint(self, seq: np.ndarray):
        if self._move_joint_cli is None:
            return
        if self._pending_motion is not None and not self._pending_motion.done():
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
        if seq.size >= 7:
            self._command_gripper(seq[6])

    def _send_servoj(self, seq: np.ndarray, joint_state: np.ndarray):
        if self._servoj_pub is None:
            return
        if joint_state is None or joint_state.shape[0] < 6:
            self.get_logger().warn("Joint state unavailable for servoj_rt command.")
            return
        rad_to_deg = 180.0 / np.pi
        current_deg = joint_state[:6] * rad_to_deg
        delta_deg = seq[:6] * self._args.delta_scale * rad_to_deg
        target = (current_deg + delta_deg).tolist()
        msg = ServojRtStream()
        msg.pos = target
        msg.vel = [float(self._args.servoj_vel)] * 6
        msg.acc = [float(self._args.servoj_acc)] * 6
        msg.time = float(self._args.servoj_time)
        self._servoj_pub.publish(msg)
        if seq.size >= 7:
            self._command_gripper(seq[6])

    def _command_gripper(self, gripper_value: float):
        if self._gripper_controller is None:
            return
        clipped = float(np.clip(gripper_value, -1.0, 1.0))
        fraction = (clipped + 1.0) * 0.5
        stroke_range = self._args.gripper_max_stroke - self._args.gripper_min_stroke
        if stroke_range <= 0:
            stroke_range = 1
        stroke = int(round(self._args.gripper_min_stroke + fraction * stroke_range))
        if self._last_gripper_stroke is not None:
            if abs(stroke - self._last_gripper_stroke) < self._args.gripper_deadband:
                return
        try:
            if self._args.gripper_fast:
                success = self._gripper_controller.fast_move(stroke)
            else:
                success = self._gripper_controller.move(stroke)
            if success:
                self._last_gripper_stroke = stroke
        except Exception as exc:  # pragma: no cover - ROS runtime failure
            self.get_logger().error(f"Failed to command gripper: {exc}")

    def destroy_node(self):
        if self._gripper_controller is not None:
            try:
                self._gripper_controller.terminate()
            except Exception:  # pragma: no cover
                pass
        super().destroy_node()


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
    parser.add_argument(
        "--motion-interface",
        choices=("servoj_rt", "move_joint"),
        default="servoj_rt",
        help="Use high-rate servoj_rt streaming or legacy MoveJoint requests.",
    )
    parser.add_argument("--servoj-vel", type=float, default=30.0, help="Joint velocity for servoj_rt (deg/s).")
    parser.add_argument("--servoj-acc", type=float, default=60.0, help="Joint acceleration for servoj_rt (deg/s^2).")
    parser.add_argument("--servoj-time", type=float, default=0.0, help="Integration time for servoj_rt.")
    parser.add_argument("--enable-gripper", action="store_true", help="Enable adaptive gripper control.")
    parser.add_argument(
        "--gripper-min-stroke", type=int, default=0, help="Minimum stroke value for the adaptive gripper (typically open)."
    )
    parser.add_argument(
        "--gripper-max-stroke", type=int, default=740, help="Maximum stroke value for the adaptive gripper (typically closed)."
    )
    parser.add_argument(
        "--gripper-deadband",
        type=int,
        default=5,
        help="Only send a new gripper command if the requested stroke changes by at least this amount.",
    )
    parser.add_argument(
        "--gripper-fast",
        action="store_true",
        help="Use fast_move (open-loop) instead of adaptive move when commanding the gripper.",
    )
    parser.add_argument(
        "--gripper-init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize the gripper Modbus connection on startup. Disable if already initialized elsewhere.",
    )
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
