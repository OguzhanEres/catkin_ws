#!/usr/bin/env python3
"""
PPO + LSTM agent node (inference-only).
Consumes vectorized sensor inputs and publishes a raw route (Path).
"""
from typing import Dict, Optional, Tuple

import numpy as np
import rospy
import torch
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, String, Empty


class PPOLSTMActorCritic(torch.nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, lstm_layers: int):
        super().__init__()
        self.feature = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
        )
        self.lstm = torch.nn.LSTM(hidden_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.actor_exploration = torch.nn.Linear(hidden_size, action_dim)
        self.actor_tracking = torch.nn.Linear(hidden_size, action_dim)
        self.critic = torch.nn.Linear(hidden_size, 1)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

    def forward_features(
        self, obs: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.feature(obs)
        x = x.unsqueeze(1)
        out, hidden = self.lstm(x, hidden)
        return out.squeeze(1), hidden

    def act(
        self, obs: torch.Tensor, hidden: Tuple[torch.Tensor, torch.Tensor], mode: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        feat, hidden = self.forward_features(obs, hidden)
        if mode == "tracking":
            mean = self.actor_tracking(feat)
        else:
            mean = self.actor_exploration(feat)
        action = torch.tanh(mean)
        value = self.critic(feat)
        return action, value, hidden


class PPOLSTMAgentNode:
    def __init__(self):
        rospy.init_node("ppo_lstm_agent", anonymous=False)

        self.lidar_topic = rospy.get_param("~lidar_vec_topic", "/agent/lidar_vec")
        self.camera_topic = rospy.get_param("~camera_vec_topic", "/agent/camera_vec")
        self.imu_topic = rospy.get_param("~imu_vec_topic", "/agent/imu_vec")
        self.gyro_topic = rospy.get_param("~gyro_vec_topic", "/agent/gyro_vec")
        self.gps_topic = rospy.get_param("~gps_vec_topic", "/agent/gps_vec")
        self.mode_topic = rospy.get_param("~mode_topic", "/agent/mode")

        self.frame_id = rospy.get_param("~frame_id", "local_enu")
        self.route_length = int(rospy.get_param("~route_length", 6))
        self.step_size = float(rospy.get_param("~step_size", 1.0))

        self.device_name = rospy.get_param("~device", "cpu")
        self.hidden_size = int(rospy.get_param("~hidden_size", 128))
        self.lstm_layers = int(rospy.get_param("~lstm_layers", 1))
        self.action_dim = int(rospy.get_param("~action_dim", 3))
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "")

        self.mode = "exploration"
        self.model: Optional[PPOLSTMActorCritic] = None
        self.obs_dim: Optional[int] = None
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

        self.latest: Dict[str, Optional[np.ndarray]] = {
            "lidar": None,
            "camera": None,
            "imu": None,
            "gyro": None,
            "gps": None,
        }
        self.expected_sizes: Dict[str, int] = {}

        self.route_pub = rospy.Publisher("/agent/route_raw", Path, queue_size=1)

        self.lidar_sub = rospy.Subscriber(self.lidar_topic, Float32MultiArray, self._vec_cb("lidar"), queue_size=1)
        self.camera_sub = rospy.Subscriber(self.camera_topic, Float32MultiArray, self._vec_cb("camera"), queue_size=1)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Float32MultiArray, self._vec_cb("imu"), queue_size=1)
        self.gyro_sub = rospy.Subscriber(self.gyro_topic, Float32MultiArray, self._vec_cb("gyro"), queue_size=1)
        self.gps_sub = rospy.Subscriber(self.gps_topic, Float32MultiArray, self._vec_cb("gps"), queue_size=1)

        self.mode_sub = rospy.Subscriber(self.mode_topic, String, self._mode_cb, queue_size=1)
        self.reset_sub = rospy.Subscriber("/agent/reset_lstm", Empty, self._reset_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration.from_sec(0.1), self._tick)

    def _vec_cb(self, name: str):
        def _handler(msg: Float32MultiArray):
            vec = np.array(msg.data, dtype=np.float32)
            if name in self.expected_sizes:
                if vec.size != self.expected_sizes[name]:
                    rospy.logwarn_throttle(2.0, "%s size mismatch: %d != %d", name, vec.size, self.expected_sizes[name])
                    return
            else:
                self.expected_sizes[name] = vec.size
            self.latest[name] = vec

        return _handler

    def _mode_cb(self, msg: String):
        value = msg.data.strip().lower()
        if value not in ("exploration", "tracking"):
            rospy.logwarn_throttle(2.0, "Unknown mode: %s", value)
            return
        if value != self.mode:
            self.mode = value
            self._reset_hidden()
            rospy.loginfo("Agent mode set to: %s", self.mode)

    def _reset_cb(self, _msg: Empty):
        self._reset_hidden()
        rospy.loginfo("LSTM state reset requested")

    def _reset_hidden(self):
        if self.model is None:
            return
        device = torch.device(self.device_name)
        h = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=device)
        c = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=device)
        self.hidden = (h, c)

    def _ready(self) -> bool:
        return all(self.latest[key] is not None for key in self.latest)

    def _init_model(self, obs_dim: int):
        device = torch.device(self.device_name)
        self.model = PPOLSTMActorCritic(obs_dim, self.action_dim, self.hidden_size, self.lstm_layers).to(device)
        self._reset_hidden()
        if self.checkpoint_path:
            try:
                state = torch.load(self.checkpoint_path, map_location=device)
                self.model.load_state_dict(state, strict=False)
                rospy.loginfo("Loaded checkpoint: %s", self.checkpoint_path)
            except Exception as exc:
                rospy.logwarn("Checkpoint load failed: %s", exc)

    def _build_obs(self) -> Optional[np.ndarray]:
        if not self._ready():
            return None
        parts = [self.latest["lidar"], self.latest["camera"], self.latest["imu"], self.latest["gyro"], self.latest["gps"]]
        if any(p is None for p in parts):
            return None
        return np.concatenate(parts).astype(np.float32)

    def _build_route(self, action: np.ndarray) -> Path:
        direction = action[:3].astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        for i in range(1, self.route_length + 1):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(direction[0] * self.step_size * i)
            pose.pose.position.y = float(direction[1] * self.step_size * i)
            pose.pose.position.z = float(direction[2] * self.step_size * i)
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)

        return path

    def _tick(self, _event):
        obs = self._build_obs()
        if obs is None:
            return
        if self.model is None:
            self.obs_dim = int(obs.size)
            self._init_model(self.obs_dim)
        if self.model is None or self.hidden is None:
            return

        device = torch.device(self.device_name)
        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_t, _value, self.hidden = self.model.act(obs_t, self.hidden, self.mode)
        action = action_t.squeeze(0).cpu().numpy()
        route = self._build_route(action)
        self.route_pub.publish(route)


if __name__ == "__main__":
    try:
        node = PPOLSTMAgentNode()
        rospy.loginfo("PPO LSTM agent node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
