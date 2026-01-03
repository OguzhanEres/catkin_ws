#!/usr/bin/env python3
"""
DRL decision node running PPO policy to output raw velocity commands.
- Consumes downsampled, normalized LiDAR + target bbox offsets + UAV velocities.
- Publishes raw Twist to be filtered by the safety layer.
"""
from typing import Optional

import numpy as np
import rospy
import torch
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from stable_baselines3 import PPO
from std_msgs.msg import Float32MultiArray


STATE_SIZE = 24  # 20 lidar + 2 target offset + 2 velocity


def replace_inf_with_max(ranges: np.ndarray, max_range: float) -> np.ndarray:
    clean = np.copy(ranges)
    clean[np.isinf(clean)] = max_range
    return clean


def downsample_and_normalize_lidar(msg: LaserScan, target_count: int = 20) -> np.ndarray:
    ranges = np.array(msg.ranges, dtype=np.float32)
    ranges = replace_inf_with_max(ranges, msg.range_max)
    idx = np.linspace(0, len(ranges) - 1, target_count).astype(int)
    sliced = ranges[idx]
    return sliced / msg.range_max


class DRLAgentNode:
    def __init__(self):
        rospy.init_node("drl_agent", anonymous=False)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=1)
        self.target_sub = rospy.Subscriber("/target/bbox", Float32MultiArray, self._target_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber("/mavros/local_position/odom", Odometry, self._odom_cb, queue_size=1)
        self.cmd_pub = rospy.Publisher("/cmd_vel_raw", Twist, queue_size=1)

        self.model_path = rospy.get_param("~model_path", "ppo_policy.zip")
        self.device = rospy.get_param("~device", "cpu")
        self.model: Optional[PPO] = self._load_model(self.model_path, self.device)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_target: Optional[Float32MultiArray] = None
        self.latest_odom: Optional[Odometry] = None

        self.timer = rospy.Timer(rospy.Duration.from_sec(0.05), self._tick)  # 20 Hz

    def _load_model(self, path: str, device: str) -> Optional[PPO]:
        if path is None:
            rospy.logwarn("Model path not provided; node will not publish commands")
            return None
        try:
            model = PPO.load(path, device=device)
            rospy.loginfo(f"Loaded PPO model from {path} on {device}")
            return model
        except Exception as exc:  # pragma: no cover - runtime dependency
            rospy.logerr(f"Failed to load PPO model: {exc}")
            return None

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _target_cb(self, msg: Float32MultiArray):
        self.latest_target = msg

    def _odom_cb(self, msg: Odometry):
        self.latest_odom = msg

    def _tick(self, _event):
        if not self._ready():
            return
        obs = self._build_state()
        if obs is None or self.model is None:
            return
        action, _ = self.model.predict(obs, deterministic=True)
        cmd = Twist()
        cmd.linear.x = float(action[0])
        cmd.angular.z = float(action[1])
        self.cmd_pub.publish(cmd)

    def _ready(self) -> bool:
        return self.latest_scan is not None and self.latest_target is not None and self.latest_odom is not None

    def _build_state(self) -> Optional[np.ndarray]:
        scan = self.latest_scan
        if scan is None:
            return None
        lidar_vec = downsample_and_normalize_lidar(scan, target_count=20)

        target = self.latest_target
        data = target.data
        if len(data) < 2:
            rospy.logwarn_throttle(5.0, "Target bbox missing entries")
            return None
        target_vec = np.array([data[0], data[1]], dtype=np.float32)

        odom = self.latest_odom
        vx = odom.twist.twist.linear.x
        vyaw = odom.twist.twist.angular.z
        vel_vec = np.array([vx, vyaw], dtype=np.float32)

        state = np.concatenate([lidar_vec, target_vec, vel_vec]).astype(np.float32)
        if state.shape[0] != STATE_SIZE:
            rospy.logwarn_throttle(5.0, f"State size mismatch: {state.shape[0]}")
            return None
        return state


if __name__ == "__main__":
    try:
        node = DRLAgentNode()
        rospy.loginfo("DRL agent node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
