#!/usr/bin/env python3
"""
PPO + LSTM trainer node for Gazebo + ArduPilot.
Publishes route actions, observes progress toward a fixed target, and updates policy.
"""
import argparse
import atexit
import csv
import math
import os
import signal
import sys
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
import torch
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState, ModelStates
from gazebo_msgs.srv import SetModelState
from mavros_msgs.msg import State
from mavros_msgs.msg import StatusText
from mavros_msgs.msg import WaypointReached
from mavros_msgs.srv import CommandBool, CommandTOL, ParamGet, ParamPull, ParamSet, SetMode
from mavros_msgs.srv import WaypointClear
from mavros_msgs.msg import ParamValue
from nav_msgs.msg import Odometry
from nav_msgs.msg import Path
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

    def forward_sequence(
        self,
        obs_seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = self.feature(obs_seq)
        out, hidden = self.lstm(x, hidden)
        return out, hidden

    def act_step(
        self,
        obs: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor],
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        obs_seq = obs.unsqueeze(0).unsqueeze(0)
        out, hidden = self.forward_sequence(obs_seq, hidden)
        feat = out.squeeze(0).squeeze(0)
        mean = self.actor_tracking(feat) if mode == "tracking" else self.actor_exploration(feat)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        logp = dist.log_prob(action).sum(-1)
        value = self.critic(feat).squeeze(-1)
        return action, logp, value, hidden

    def evaluate_actions(
        self,
        obs_seq: torch.Tensor,
        actions: torch.Tensor,
        mode: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out, _ = self.forward_sequence(obs_seq.unsqueeze(0))
        feat = out.squeeze(0)
        mean = self.actor_tracking(feat) if mode == "tracking" else self.actor_exploration(feat)
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        logp = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        value = self.critic(feat).squeeze(-1)
        return logp, entropy, value


class PPOLSTMTrainerNode:
    _PREARM_BLOCKING_PHRASES = (
        "Need Position Estimate",
        "requires position",
        "EKF velocity variance",
        "EKF variance",
        "EKF compass variance",
        "EKF3 Yaw inconsistent",
        "EKF3 yaw inconsistent",
        "GPS Glitch",
        "Compass error",
        "vel error",
        "compass variance",
        "mag anomaly",
        "Crash: Disarming",
    )

    def __init__(self, args):
        rospy.init_node("ppo_lstm_trainer", anonymous=False)

        self.mode = rospy.get_param("~mode", args.mode)
        self.epochs = int(rospy.get_param("~epochs", args.epochs))
        self.loop_forever = bool(rospy.get_param("~loop_forever", False))
        # LSTM needs longer episodes to learn temporal patterns (500-1000 steps)
        self.max_steps = int(rospy.get_param("~max_steps", 500))
        self.update_epochs = int(rospy.get_param("~update_epochs", 5))
        self.gamma = float(rospy.get_param("~gamma", 0.99))  # Increased for longer episodes
        self.gae_lambda = float(rospy.get_param("~gae_lambda", 0.95))
        self.clip_ratio = float(rospy.get_param("~clip_ratio", 0.2))
        self.lr = float(rospy.get_param("~lr", 1e-4))  # Reduced for stability
        self.value_coef = float(rospy.get_param("~value_coef", 0.5))
        self.entropy_coef = float(rospy.get_param("~entropy_coef", 0.05))  # Increased for exploration
        self.step_timeout = float(rospy.get_param("~step_timeout", 4.0))
        self.use_reached_event = bool(rospy.get_param("~use_reached_event", True))
        self.pose_timeout = float(rospy.get_param("~pose_timeout", 1.0))
        self.auto_start_mode = rospy.get_param("~auto_start_mode", "AUTO")
        self.start_auto_after_takeoff = bool(rospy.get_param("~start_auto_after_takeoff", True))
        self.enforce_guided_only = bool(rospy.get_param("~enforce_guided_only", True))
        self.required_mode = rospy.get_param("~required_mode", "GUIDED")
        self.allow_auto_mode = bool(rospy.get_param("~allow_auto_mode", False))
        self.reset_each_epoch = bool(rospy.get_param("~reset_each_epoch", True))
        self.preflight_gate = bool(rospy.get_param("~preflight_gate", True))
        self.preflight_timeout = float(rospy.get_param("~preflight_timeout", 45.0))
        self.preflight_stable_time = float(rospy.get_param("~preflight_stable_time", 2.0))
        self.disable_arming_checks = bool(rospy.get_param("~disable_arming_checks", False))
        self.arm_attempts = int(rospy.get_param("~arm_attempts", 6))

        self.target_x = float(rospy.get_param("~target_x", 20.0))
        self.target_y = float(rospy.get_param("~target_y", 20.0))
        self.target_z = float(rospy.get_param("~target_z", 1.0))
        self.target_in_world = bool(rospy.get_param("~target_in_world", True))
        self.goal_radius = float(rospy.get_param("~goal_radius", 1.0))  # Reach threshold for intermediate goal
        self.step_penalty = float(rospy.get_param("~step_penalty", 0.01))  # Small existence penalty
        self.progress_scale = float(rospy.get_param("~progress_scale", 10.0))  # Increased for better signal
        self.intermediate_bonus = float(rospy.get_param("~intermediate_bonus", 10.0))  # Reward for reaching waypoint
        self.crash_penalty = float(rospy.get_param("~crash_penalty", -50.0))  # Large penalty for crash
        self.timeout_penalty = float(rospy.get_param("~timeout_penalty", -10.0))  # Penalty for timeout
        self.min_alt = float(rospy.get_param("~min_alt", 0.3))  # Minimum safe altitude
        self.max_dist = float(rospy.get_param("~max_dist", 100.0))  # Max distance before out of bounds
        self.collision_distance = float(rospy.get_param("~collision_distance", 0.5))  # LIDAR collision threshold

        # === CONTINUOUS FLOW (Carrot-Stick) PARAMETERS ===
        self.max_steps = int(rospy.get_param("~max_steps", 500))  # Episode length
        self.min_new_target_dist = float(rospy.get_param("~min_new_target_dist", 5.0))  # Min distance for new target
        self.max_new_target_dist = float(rospy.get_param("~max_new_target_dist", 30.0))  # Max distance for new target
        # Map boundaries for random target generation
        self.map_x_min = float(rospy.get_param("~map_x_min", -28.0))
        self.map_x_max = float(rospy.get_param("~map_x_max", 23.0))
        self.map_y_min = float(rospy.get_param("~map_y_min", -28.0))
        self.map_y_max = float(rospy.get_param("~map_y_max", 23.0))
        # Counter for intermediate goals reached in current episode
        self._goals_reached_this_episode = 0
        self._current_step = 0
        # Warm-up period: disable termination checks for first N steps after episode start
        # This prevents instant death due to sensor noise or spawn position issues
        self.warmup_steps = int(rospy.get_param("~warmup_steps", 10))

        self.spawn_x = float(rospy.get_param("~spawn_x", -25.0))
        self.spawn_y = float(rospy.get_param("~spawn_y", -25.0))
        self.spawn_z = float(rospy.get_param("~spawn_z", 0.1))
        self.spawn_yaw = float(rospy.get_param("~spawn_yaw", 0.785))
        self.model_name = rospy.get_param("~model_name", "iris_px4_sensors")
        self.takeoff_alt = float(rospy.get_param("~takeoff_alt", 1.0))
        self.send_takeoff_cmd = bool(rospy.get_param("~send_takeoff_cmd", False))
        self.reset_wait = float(rospy.get_param("~reset_wait", 1.0))
        # Minimum altitude threshold before DRL starts sending movement commands
        # e.g., if takeoff_alt=3.0, min_takeoff_check_alt=1.5 means wait until drone reaches 1.5m
        self.min_takeoff_check_alt = float(rospy.get_param("~min_takeoff_check_alt", 0.5))

        # === SAFE SPAWN & EPISODE LENGTH FIX ===
        # Minimum distance to goal at spawn (prevents instant success)
        self.min_goal_dist_at_spawn = float(rospy.get_param("~min_goal_dist_at_spawn", 10.0))
        # Minimum LIDAR clearance at spawn (prevents spawning inside obstacles)
        self.min_lidar_clearance = float(rospy.get_param("~min_lidar_clearance", 2.0))
        # Maximum spawn attempts before using default position
        self.max_spawn_attempts = int(rospy.get_param("~max_spawn_attempts", 10))
        # Random spawn area (for curriculum learning)
        self.random_spawn = bool(rospy.get_param("~random_spawn", False))
        self.spawn_area_x = tuple(rospy.get_param("~spawn_area_x", [-30.0, -20.0]))
        self.spawn_area_y = tuple(rospy.get_param("~spawn_area_y", [-30.0, -20.0]))
        # Curriculum level (0=easy, 1=medium, 2=hard)
        self.curriculum_level = int(rospy.get_param("~curriculum_level", 0))
        # Safety Net: Maximum drift distance before force respawn (meters)
        self.max_drift_distance = float(rospy.get_param("~max_drift_distance", 100.0))
        # Yaw alignment reward scale (encourages drone to face target, prevents crab walking)
        self.yaw_alignment_scale = float(rospy.get_param("~yaw_alignment_scale", 0.3))
        # Frame stacking for temporal information
        self.frame_stack_size = int(rospy.get_param("~frame_stack_size", 1))
        # Reward normalization/clipping
        self.reward_clip = float(rospy.get_param("~reward_clip", 10.0))
        self.use_reward_normalization = bool(rospy.get_param("~use_reward_normalization", True))
        # Running reward stats for normalization
        self._reward_mean = 0.0
        self._reward_var = 1.0
        self._reward_count = 0
        # Frame stack buffer
        self._frame_buffer: List[np.ndarray] = []
        # Curriculum learning tracking
        self._curriculum_success_count = 0
        self._curriculum_episode_count = 0
        self._curriculum_success_threshold = 0.7  # 70% success rate to advance
        self._curriculum_window = 20  # Episodes to evaluate
        self._curriculum_successes_window: List[bool] = []

        self.route_length = int(rospy.get_param("~route_length", 1))
        self.step_size = float(rospy.get_param("~step_size", 1.0))
        self.frame_id = rospy.get_param("~frame_id", "local_enu")
        self.use_z_action = bool(rospy.get_param("~use_z_action", False))
        self.max_route_z = float(rospy.get_param("~max_route_z", 0.0))

        self.device_name = rospy.get_param("~device", "cpu")
        self.hidden_size = int(rospy.get_param("~hidden_size", 128))
        self.lstm_layers = int(rospy.get_param("~lstm_layers", 1))
        self.action_dim = int(rospy.get_param("~action_dim", 3))
        self.checkpoint_path = rospy.get_param("~checkpoint_path", "ppo_lstm_checkpoint.pt")
        self.auto_param_tune = bool(rospy.get_param("~auto_param_tune", True))
        self.auto_arm = bool(rospy.get_param("~auto_arm", True))
        self.offboard_prestream_sec = float(rospy.get_param("~offboard_prestream_sec", 1.0))

        self.latest_vecs: Dict[str, Optional[np.ndarray]] = {
            "lidar": None,
            "camera": None,
            "imu": None,
            "gyro": None,
            "gps": None,
        }
        self.last_pose: Optional[PoseStamped] = None
        self.last_gazebo_pose: Optional[PoseStamped] = None
        self._last_pose_stamp: Optional[rospy.Time] = None
        self.last_local_odom: Optional[Odometry] = None
        self._last_local_odom_stamp: Optional[rospy.Time] = None
        self._gazebo_index: Optional[int] = None
        self._pose_source_notice = False
        self._ned_pose_notice = False
        self._params_ready = False
        self._require_restart = False
        self.last_state: Optional[State] = None
        # GPS normalization: spawn GPS as reference point
        self._spawn_gps: Optional[np.ndarray] = None

        self.route_pub = rospy.Publisher("/agent/route_raw", Path, queue_size=1)
        self.mode_pub = rospy.Publisher("/agent/mode", String, queue_size=1, latch=True)

        self.lidar_sub = rospy.Subscriber("/agent/lidar_vec", Float32MultiArray, self._vec_cb("lidar"), queue_size=1)
        self.camera_sub = rospy.Subscriber("/agent/camera_vec", Float32MultiArray, self._vec_cb("camera"), queue_size=1)
        self.imu_sub = rospy.Subscriber("/agent/imu_vec", Float32MultiArray, self._vec_cb("imu"), queue_size=1)
        self.gyro_sub = rospy.Subscriber("/agent/gyro_vec", Float32MultiArray, self._vec_cb("gyro"), queue_size=1)
        self.gps_sub = rospy.Subscriber("/agent/gps_vec", Float32MultiArray, self._vec_cb("gps"), queue_size=1)
        self.pose_sub = rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=1)
        self.odom_sub = rospy.Subscriber("/mavros/global_position/local", Odometry, self._odom_cb, queue_size=1)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=1)
        # Step completion can come from the FCU mission interface or from the local setpoint follower.
        self.reached_sub = rospy.Subscriber("/mavros/mission/reached", WaypointReached, self._reached_cb, queue_size=1)
        self.step_reached_sub = rospy.Subscriber("/agent/wp_reached", Empty, self._step_reached_cb, queue_size=1)
        self.gazebo_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_cb, queue_size=1)
        self.statustext_sub = rospy.Subscriber("/mavros/statustext/recv", StatusText, self._statustext_cb, queue_size=10)

        self.reached_event = threading.Event()

        self.set_mode_srv = self._wait_service("/mavros/set_mode", SetMode)
        self.arm_srv = self._wait_service("/mavros/cmd/arming", CommandBool)
        self.takeoff_srv = self._wait_service("/mavros/cmd/takeoff", CommandTOL)
        self.param_set_srv = self._wait_service("/mavros/param/set", ParamSet)
        self.param_get_srv = self._wait_service("/mavros/param/get", ParamGet)
        self.param_pull_srv = self._wait_service("/mavros/param/pull", ParamPull)
        self.wp_clear_srv = self._wait_service("/mavros/mission/clear", WaypointClear)
        self.set_state_srv = self._wait_service("/gazebo/set_model_state", SetModelState)

        self.model: Optional[PPOLSTMActorCritic] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.obs_dim: Optional[int] = None

        self._target_local, self._target_world = self._resolve_targets()
        self._prearm_bad_until = rospy.Time(0)
        self._last_statustext = ""
        self._last_statustext_stamp = rospy.Time(0)

        # Metrics tracking
        self._metrics_path = rospy.get_param("~metrics_path", 
            os.path.join(os.path.dirname(self.checkpoint_path) if self.checkpoint_path else ".", "metrics.csv"))
        self._epoch_metrics: List[Dict] = []
        self._current_epoch = 0
        self._shutdown_requested = False
        
        # Register shutdown handlers for graceful exit
        self._register_shutdown_handlers()

    def _register_shutdown_handlers(self):
        """Register handlers to save state on shutdown (Ctrl+C, SIGTERM, etc.)"""
        def _save_on_shutdown(signum=None, frame=None):
            if self._shutdown_requested:
                return
            self._shutdown_requested = True
            rospy.logwarn("Shutdown signal received, saving state...")
            self._emergency_save()
            if signum is not None:
                sys.exit(0)
        
        # Register with atexit for normal exits
        atexit.register(self._emergency_save)
        
        # Register signal handlers
        signal.signal(signal.SIGINT, _save_on_shutdown)
        signal.signal(signal.SIGTERM, _save_on_shutdown)
        
        # ROS shutdown hook
        rospy.on_shutdown(self._emergency_save)

    def _emergency_save(self):
        """Save checkpoint and metrics on emergency shutdown"""
        try:
            if self.model is not None and self.checkpoint_path:
                torch.save(self.model.state_dict(), self.checkpoint_path)
                rospy.loginfo("Emergency checkpoint saved: %s", self.checkpoint_path)
        except Exception as e:
            rospy.logerr("Failed to save emergency checkpoint: %s", e)
        
        try:
            self._save_metrics(force=True)
        except Exception as e:
            rospy.logerr("Failed to save emergency metrics: %s", e)

    def _save_metrics(self, force: bool = False):
        """Save collected metrics to CSV file"""
        if not self._epoch_metrics and not force:
            return
        
        if not self._metrics_path:
            return
            
        try:
            # Ensure directory exists
            metrics_dir = os.path.dirname(self._metrics_path)
            if metrics_dir:
                os.makedirs(metrics_dir, exist_ok=True)
            
            file_exists = os.path.exists(self._metrics_path)
            with open(self._metrics_path, 'a', newline='') as f:
                if self._epoch_metrics:
                    fieldnames = list(self._epoch_metrics[0].keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerows(self._epoch_metrics)
                    rospy.loginfo("Saved %d epoch metrics to %s", len(self._epoch_metrics), self._metrics_path)
                    self._epoch_metrics = []
        except Exception as e:
            rospy.logerr("Failed to save metrics: %s", e)

    def _record_epoch_metrics(self, epoch: int, steps: int, total_reward: float,
                              success: bool, final_dist: Optional[float]):
        """Record metrics for a completed epoch"""
        import time
        metrics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'epoch': epoch,
            'steps': steps,
            'total_reward': round(total_reward, 4),
            'success': int(success),
            'final_distance': round(final_dist, 4) if final_dist else -1,
            'mode': self.mode,
            'curriculum_level': self.curriculum_level,
            'goal_radius': self.goal_radius,
            'min_lidar_dist': round(self._get_min_lidar_distance() or -1, 4),
        }
        self._epoch_metrics.append(metrics)

        # Auto-save every 5 epochs
        if len(self._epoch_metrics) >= 5:
            self._save_metrics()

    def _resolve_targets(self) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        target_world = (self.target_x, self.target_y, self.target_z)
        if self.target_in_world:
            target_local = (
                self.target_x - self.spawn_x,
                self.target_y - self.spawn_y,
                self.target_z - self.spawn_z,
            )
            return target_local, target_world
        target_local = (self.target_x, self.target_y, self.target_z)
        target_world = (
            self.target_x + self.spawn_x,
            self.target_y + self.spawn_y,
            self.target_z + self.spawn_z,
        )
        return target_local, target_world

    def _generate_random_target(self) -> Tuple[float, float, float]:
        """
        Generate a new random target within map boundaries.
        Ensures target is at least min_new_target_dist away from current position.
        """
        current = self._current_position()
        if current is None:
            current = (self.spawn_x, self.spawn_y, self.takeoff_alt)

        for _ in range(50):  # Max attempts
            new_x = np.random.uniform(self.map_x_min, self.map_x_max)
            new_y = np.random.uniform(self.map_y_min, self.map_y_max)
            new_z = self.takeoff_alt  # Keep same altitude

            # Calculate distance from current position
            dx = new_x - current[0]
            dy = new_y - current[1]
            dist = math.sqrt(dx * dx + dy * dy)

            # Check if within acceptable range
            if self.min_new_target_dist <= dist <= self.max_new_target_dist:
                return (new_x, new_y, new_z)

        # Fallback: random position if no valid found
        return (
            np.random.uniform(self.map_x_min, self.map_x_max),
            np.random.uniform(self.map_y_min, self.map_y_max),
            self.takeoff_alt
        )

    def _set_new_target(self, x: float, y: float, z: float):
        """Update target coordinates and recalculate target positions."""
        self.target_x = x
        self.target_y = y
        self.target_z = z
        self._target_local, self._target_world = self._resolve_targets()
        rospy.loginfo("NEW TARGET SET: (%.2f, %.2f, %.2f)", x, y, z)

    def _vec_cb(self, name: str):
        def _handler(msg: Float32MultiArray):
            self.latest_vecs[name] = np.array(msg.data, dtype=np.float32)
        return _handler

    def _pose_cb(self, msg: PoseStamped):
        self.last_pose = msg
        if msg.header.stamp and msg.header.stamp != rospy.Time(0):
            self._last_pose_stamp = msg.header.stamp
        else:
            self._last_pose_stamp = rospy.Time.now()

    def _odom_cb(self, msg: Odometry):
        self.last_local_odom = msg
        if msg.header.stamp and msg.header.stamp != rospy.Time(0):
            self._last_local_odom_stamp = msg.header.stamp
        else:
            self._last_local_odom_stamp = rospy.Time.now()

    def _state_cb(self, msg: State):
        self.last_state = msg

    def _gazebo_cb(self, msg: ModelStates):
        if not msg.name:
            return
        idx = self._gazebo_index
        if idx is None or idx >= len(msg.name) or msg.name[idx] != self.model_name:
            idx = None
            if self.model_name in msg.name:
                idx = msg.name.index(self.model_name)
            else:
                for i, name in enumerate(msg.name):
                    if name.endswith(self.model_name) or self.model_name in name:
                        idx = i
                        break
            if idx is None:
                return
            self._gazebo_index = idx
        pose = msg.pose[self._gazebo_index]
        stamped = PoseStamped()
        stamped.header.stamp = rospy.Time.now()
        stamped.header.frame_id = "world"
        stamped.pose = pose
        self.last_gazebo_pose = stamped

    def _reached_cb(self, _msg: WaypointReached):
        self.reached_event.set()

    def _step_reached_cb(self, _msg: Empty):
        self.reached_event.set()

    def _statustext_cb(self, msg: StatusText):
        text = (msg.text or "").strip()
        if not text:
            return
        now = rospy.Time.now()
        self._last_statustext = text
        self._last_statustext_stamp = now
        for phrase in self._PREARM_BLOCKING_PHRASES:
            if phrase in text:
                self._prearm_bad_until = now + rospy.Duration.from_sec(max(0.5, self.preflight_stable_time))
                break

    def _wait_service(self, name, srv_type):
        rospy.loginfo("Waiting for service: %s", name)
        rospy.wait_for_service(name)
        return rospy.ServiceProxy(name, srv_type)

    def _wait_ready(self):
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if all(v is not None for v in self.latest_vecs.values()) and (
                self.last_pose is not None or self.last_gazebo_pose is not None
            ):
                return
            rate.sleep()

    def _ensure_param_cache(self):
        if self._params_ready:
            return
        for _ in range(5):
            try:
                resp = self.param_pull_srv(force_pull=True)
                if resp.success:
                    self._params_ready = True
                    return
            except rospy.ServiceException as exc:
                rospy.logwarn("Param pull failed: %s", exc)
            rospy.sleep(0.5)
        rospy.logwarn("Param pull did not complete; param updates may fail")

    def _param_exists(self, name: str) -> bool:
        try:
            resp = self.param_get_srv(param_id=name)
            return resp.success
        except rospy.ServiceException as exc:
            rospy.logwarn("Param get failed (%s): %s", name, exc)
            return False

    def _get_param_int(self, name: str) -> Optional[int]:
        try:
            resp = self.param_get_srv(param_id=name)
        except rospy.ServiceException as exc:
            rospy.logwarn("Param get failed (%s): %s", name, exc)
            return None
        if not resp.success:
            return None
        if resp.value.integer != 0 or resp.value.real == 0.0:
            return int(resp.value.integer)
        return int(round(resp.value.real))

    def _set_param(self, name: str, value: int):
        self._ensure_param_cache()
        if not self._param_exists(name):
            rospy.logwarn("Param not found: %s", name)
            return False
        try:
            req = ParamValue(integer=value, real=0.0)
            resp = self.param_set_srv(param_id=name, value=req)
            if not resp.success:
                rospy.logwarn("Param set rejected: %s", name)
            return resp.success
        except rospy.ServiceException as exc:
            rospy.logwarn("Param set failed (%s): %s", name, exc)
        return False

    def _configure_fcu_params(self):
        self._ensure_param_cache()
        gps_type = self._get_param_int("GPS1_TYPE")
        if gps_type is not None and gps_type == 0:
            if self._set_param("GPS1_TYPE", 1):
                self._require_restart = True
                rospy.logwarn("GPS1_TYPE was 0; set to 1. Restart ArduPilot required.")
        self._set_param("GPS2_TYPE", 0)
        self._set_param("AHRS_GPS_USE", 1)
        self._set_param("EK3_SRC1_POSXY", 3)
        self._set_param("EK3_SRC1_VELXY", 3)
        self._set_param("EK3_SRC1_VELZ", 3)

    def _clear_mission(self):
        try:
            self.wp_clear_srv()
        except rospy.ServiceException as exc:
            rospy.logwarn("Mission clear failed: %s", exc)

    def _set_mode(self, mode: str):
        try:
            resp = self.set_mode_srv(base_mode=0, custom_mode=mode)
            if not resp.mode_sent:
                rospy.logwarn("Set mode rejected: %s", mode)
        except rospy.ServiceException as exc:
            rospy.logwarn("Set mode failed: %s", exc)

    def _prestream_offboard(self):
        if self.offboard_prestream_sec <= 0.0:
            return
        
        # Wait for EKF to stabilize after reset
        rospy.sleep(0.5)
        
        hold = Path()
        hold.header.frame_id = self.frame_id
        pose = PoseStamped()
        pose.pose.orientation.w = 1.0
        hold.poses = [pose]
        end_time = rospy.Time.now() + rospy.Duration.from_sec(self.offboard_prestream_sec)
        rate = rospy.Rate(20)  # Increased rate for smoother control
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            stamp = rospy.Time.now()
            hold.header.stamp = stamp
            hold.poses[0].header.stamp = stamp
            hold.poses[0].header.frame_id = hold.header.frame_id
            # Send zero offset - setpoint follower will interpret as "hold current position"
            hold.poses[0].pose.position.x = 0.0
            hold.poses[0].pose.position.y = 0.0
            hold.poses[0].pose.position.z = self.takeoff_alt  # Target takeoff altitude
            self.route_pub.publish(hold)
            rate.sleep()

    def _arm(self, value: bool):
        try:
            resp = self.arm_srv(value=value)
            if not resp.success:
                rospy.logwarn("Arming rejected")
        except rospy.ServiceException as exc:
            rospy.logwarn("Arming failed: %s", exc)

    def _takeoff(self, altitude: float):
        try:
            resp = self.takeoff_srv(altitude=altitude, latitude=0.0, longitude=0.0, min_pitch=0.0, yaw=0.0)
            if not resp.success:
                rospy.logwarn("Takeoff rejected")
        except rospy.ServiceException as exc:
            rospy.logwarn("Takeoff failed: %s", exc)

    def _wait_for_preflight(self, timeout_s: float) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(timeout_s)
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.last_state is None or not self.last_state.connected:
                rate.sleep()
                continue
            if self._current_position() is None:
                rate.sleep()
                continue
            if self.enforce_guided_only and self.last_state.mode != self.required_mode:
                self._set_mode(self.required_mode)
            if rospy.Time.now() <= self._prearm_bad_until:
                rate.sleep()
                continue
            return True
        rospy.logwarn("Preflight gate timeout. Last status: %s", self._last_statustext)
        return False

    def _get_min_lidar_distance(self) -> Optional[float]:
        """Get minimum distance from LIDAR readings (obstacle clearance check)."""
        lidar = self.latest_vecs.get("lidar")
        if lidar is None:
            return None
        # LIDAR values are normalized to [0, 1], multiply by max range (typically 10-30m)
        # Assuming max range of 10m for safety check
        lidar_max_range = 10.0
        min_normalized = float(np.min(lidar))
        return min_normalized * lidar_max_range

    def _compute_goal_distance_at_position(self, x: float, y: float, z: float) -> float:
        """Compute distance to goal from a given position."""
        if self.target_in_world:
            tx, ty, tz = self.target_x, self.target_y, self.target_z
        else:
            tx = self.target_x + self.spawn_x
            ty = self.target_y + self.spawn_y
            tz = self.target_z + self.spawn_z
        dx = x - tx
        dy = y - ty
        dz = z - tz
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _generate_safe_spawn_position(self) -> Tuple[float, float, float, float]:
        """
        Generate a safe spawn position that:
        1. Is at least min_goal_dist_at_spawn meters from the goal
        2. Has at least min_lidar_clearance meters from obstacles

        Returns: (x, y, z, yaw) spawn position
        """
        import random

        # Default position as fallback
        default_pos = (self.spawn_x, self.spawn_y, self.spawn_z, self.spawn_yaw)

        if not self.random_spawn:
            # Check if default position meets minimum goal distance
            goal_dist = self._compute_goal_distance_at_position(
                self.spawn_x, self.spawn_y, self.takeoff_alt
            )
            if goal_dist < self.min_goal_dist_at_spawn:
                rospy.logwarn(
                    "Default spawn (%.1f, %.1f) is only %.1fm from goal (min: %.1fm). "
                    "Consider adjusting spawn_x/spawn_y or target position.",
                    self.spawn_x, self.spawn_y, goal_dist, self.min_goal_dist_at_spawn
                )
            return default_pos

        # Try random positions
        for attempt in range(self.max_spawn_attempts):
            # Generate random position within spawn area
            x = random.uniform(self.spawn_area_x[0], self.spawn_area_x[1])
            y = random.uniform(self.spawn_area_y[0], self.spawn_area_y[1])
            z = self.spawn_z
            yaw = random.uniform(0, 2 * math.pi)

            # Check goal distance
            goal_dist = self._compute_goal_distance_at_position(x, y, self.takeoff_alt)
            if goal_dist < self.min_goal_dist_at_spawn:
                rospy.logdebug(
                    "Spawn attempt %d: (%.1f, %.1f) too close to goal (%.1fm)",
                    attempt + 1, x, y, goal_dist
                )
                continue

            # Position is valid
            rospy.loginfo(
                "Safe spawn found at (%.1f, %.1f) after %d attempts, goal_dist=%.1fm",
                x, y, attempt + 1, goal_dist
            )
            return (x, y, z, yaw)

        rospy.logwarn(
            "Could not find safe spawn after %d attempts. Using default position.",
            self.max_spawn_attempts
        )
        return default_pos

    def _check_spawn_safety(self) -> bool:
        """
        Check if current spawn position is safe (LIDAR clearance).
        Must be called AFTER spawning and waiting for sensor data.
        Returns True if safe, False otherwise.
        """
        min_dist = self._get_min_lidar_distance()
        if min_dist is None:
            rospy.logwarn("LIDAR data not available for spawn safety check")
            return True  # Assume safe if no data

        if min_dist < self.min_lidar_clearance:
            rospy.logwarn(
                "Spawn position unsafe: LIDAR min distance %.2fm < clearance %.2fm",
                min_dist, self.min_lidar_clearance
            )
            return False

        rospy.logdebug("Spawn safety check passed: min LIDAR distance %.2fm", min_dist)
        return True

    # =========================================================================
    # CURRICULUM LEARNING
    # =========================================================================
    def _get_curriculum_config(self) -> Dict:
        """
        Get configuration for current curriculum level.

        Level 0 (Easy): No obstacles, learn basic navigation
        Level 1 (Medium): Static obstacles, learn avoidance
        Level 2 (Hard): City environment, full challenge

        IMPORTANT: Long episodes (500+ steps) are needed for LSTM to learn
        temporal patterns and develop coherent navigation strategies.
        """
        configs = {
            0: {  # EASY - Learn basic navigation (Stage 1 world)
                "name": "Easy (No obstacles)",
                "goal_radius": 1.5,           # Intermediate goal reach threshold
                "max_dist": 150.0,            # Very tolerant bounds
                "max_steps": 1000,            # Long episodes for continuous learning
                "min_goal_dist": 5.0,         # Min distance for new random target
                "collision_distance": 0.3,    # Only crash on direct hit
                "intermediate_bonus": 15.0,   # Reward for reaching each waypoint
                "crash_penalty": -30.0,       # Moderate crash penalty
                "timeout_penalty": -5.0,      # Small timeout penalty
                "target_bias": 0.7,           # Strong guidance toward goal
                "progress_scale": 10.0,       # Good progress signal
            },
            1: {  # MEDIUM - Learn obstacle avoidance (Stage 2 world)
                "name": "Medium (Static obstacles)",
                "goal_radius": 1.2,           # Tighter reach threshold
                "max_dist": 150.0,            # Tolerant bounds for long distance
                "max_steps": 1000,            # Long episodes for continuous learning
                "min_goal_dist": 8.0,         # Slightly farther targets
                "collision_distance": 0.5,    # Standard collision
                "intermediate_bonus": 12.0,   # Slightly less reward
                "crash_penalty": -50.0,       # Harder crash penalty
                "timeout_penalty": -10.0,     # Moderate timeout penalty
                "target_bias": 0.4,           # Less guidance, more autonomy
                "progress_scale": 10.0,
            },
            2: {  # HARD - Full challenge (Stage 3 city world)
                "name": "Hard (City environment)",
                "goal_radius": 1.0,           # Precise goal required
                "max_dist": 150.0,            # Tolerant bounds
                "max_steps": 1000,            # Long episodes for complex navigation
                "min_goal_dist": 10.0,        # Farther targets for challenge
                "collision_distance": 0.5,    # Standard collision
                "intermediate_bonus": 10.0,   # Standard waypoint reward
                "crash_penalty": -50.0,       # Full crash penalty
                "timeout_penalty": -15.0,     # Higher timeout penalty
                "target_bias": 0.2,           # Minimal guidance, full autonomy
                "progress_scale": 10.0,
            },
        }
        return configs.get(self.curriculum_level, configs[2])

    def _apply_curriculum_config(self):
        """Apply current curriculum level configuration."""
        config = self._get_curriculum_config()

        self.goal_radius = config["goal_radius"]
        self.max_dist = config["max_dist"]
        self.max_steps = config["max_steps"]
        self.min_new_target_dist = config["min_goal_dist"]
        self.intermediate_bonus = config["intermediate_bonus"]
        self.crash_penalty = config["crash_penalty"]
        self.timeout_penalty = config["timeout_penalty"]
        self.collision_distance = config["collision_distance"]
        self.progress_scale = config["progress_scale"]

        rospy.loginfo(
            "=== Curriculum Level %d: %s ===",
            self.curriculum_level, config["name"]
        )
        rospy.loginfo(
            "  goal_radius=%.1f, max_steps=%d, collision_dist=%.1f",
            config["goal_radius"], config["max_steps"], config["collision_distance"]
        )
        rospy.loginfo(
            "  rewards: intermediate=%.0f, crash=%.0f, timeout=%.0f, target_bias=%.1f",
            config["intermediate_bonus"], config["crash_penalty"],
            config["timeout_penalty"], config["target_bias"]
        )

    def _update_curriculum(self, success: bool):
        """
        Update curriculum based on recent performance.
        Advances to harder level if success rate is high enough.
        """
        self._curriculum_successes_window.append(success)

        # Keep window size limited
        if len(self._curriculum_successes_window) > self._curriculum_window:
            self._curriculum_successes_window.pop(0)

        # Check if we have enough data
        if len(self._curriculum_successes_window) < self._curriculum_window:
            return

        # Calculate success rate
        success_rate = sum(self._curriculum_successes_window) / len(self._curriculum_successes_window)

        # Check for advancement
        if success_rate >= self._curriculum_success_threshold:
            if self.curriculum_level < 2:  # Max level is 2
                self.curriculum_level += 1
                self._curriculum_successes_window = []  # Reset window
                self._apply_curriculum_config()
                rospy.logwarn(
                    "=== CURRICULUM ADVANCED TO LEVEL %d! Success rate: %.1f%% ===",
                    self.curriculum_level, success_rate * 100
                )

    def _get_target_bias(self) -> float:
        """Get target bias based on curriculum level."""
        config = self._get_curriculum_config()
        return config.get("target_bias", 0.5)

    def _reset_pose(self):
        # Generate safe spawn position
        spawn_x, spawn_y, spawn_z, spawn_yaw = self._generate_safe_spawn_position()

        state = ModelState()
        state.model_name = self.model_name
        state.pose.position.x = spawn_x
        state.pose.position.y = spawn_y
        state.pose.position.z = spawn_z
        qz = math.sin(spawn_yaw * 0.5)
        qw = math.cos(spawn_yaw * 0.5)
        state.pose.orientation.z = qz
        state.pose.orientation.w = qw
        state.twist.linear.x = 0.0
        state.twist.linear.y = 0.0
        state.twist.linear.z = 0.0
        state.twist.angular.x = 0.0
        state.twist.angular.y = 0.0
        state.twist.angular.z = 0.0
        try:
            self.set_state_srv(state)
            # Update spawn position for target calculations
            self.spawn_x = spawn_x
            self.spawn_y = spawn_y
            self.spawn_z = spawn_z
            self.spawn_yaw = spawn_yaw
            # Recalculate target positions
            self._target_local, self._target_world = self._resolve_targets()
        except rospy.ServiceException as exc:
            rospy.logwarn("Reset pose failed: %s", exc)

    def _reset_episode(self):
        """
        Reset episode state for new training episode.
        - Resets step counter to 0
        - Resets goals reached counter
        - Resets GPS spawn reference (for normalization)
        - Generates initial target at safe distance from spawn
        - Resets drone position and arms
        """
        # === RESET EPISODE COUNTERS ===
        self._current_step = 0
        self._goals_reached_this_episode = 0
        # Reset GPS spawn reference so next GPS reading becomes new reference
        self._spawn_gps = None

        # === GENERATE INITIAL TARGET ===
        # Ensure target is at least min_goal_dist_at_spawn away from spawn position
        initial_target = self._generate_random_target()
        self._set_new_target(initial_target[0], initial_target[1], initial_target[2])
        rospy.loginfo("Episode reset. Initial target: (%.2f, %.2f, %.2f)",
                     initial_target[0], initial_target[1], initial_target[2])

        arm_mode = self.required_mode
        if arm_mode.upper() == "OFFBOARD":
            self._prestream_offboard()
        if self.auto_param_tune and self.disable_arming_checks:
            self._set_param("ARMING_SKIPCHK", -1)
            self._set_param("ARMING_NEED_LOC", 0)

        self._set_mode(arm_mode)
        if self.auto_arm:
            self._arm(False)
            rospy.sleep(0.3)
        self._clear_mission()
        self._reset_pose()
        rospy.sleep(self.reset_wait)

        if self.auto_arm:
            if self.preflight_gate and not self._wait_for_preflight(self.preflight_timeout):
                self._reset_hidden()
                return
            for _ in range(max(1, self.arm_attempts)):
                if self.last_state is not None and self.last_state.mode != arm_mode:
                    self._set_mode(arm_mode)
                    rospy.sleep(0.2)
                self._arm(True)
                rospy.sleep(0.6)
                if self.last_state is not None and self.last_state.armed:
                    break
                rospy.sleep(0.6)

        if self.send_takeoff_cmd:
            current_pos = self._current_position()
            current_alt = None
            if current_pos:
                current_alt = current_pos[2] if current_pos[2] >= 0.0 else -current_pos[2]
            if current_alt is not None and current_alt < self.takeoff_alt - 0.2 and self.last_state is not None and self.last_state.armed:
                if self.enforce_guided_only and self.last_state.mode != arm_mode:
                    self._set_mode(arm_mode)
                    rospy.sleep(0.2)
                self._takeoff(self.takeoff_alt)
                rospy.sleep(0.5)

                # PHASE 1: Wait for drone to reach minimum safe altitude before DRL can start
                # e.g., takeoff=3m, min_check=1.5m -> wait until 1.5m first
                rospy.loginfo("Waiting for drone to reach minimum altitude (%.1fm)...", self.min_takeoff_check_alt)
                takeoff_deadline = rospy.Time.now() + rospy.Duration(20.0)
                while not rospy.is_shutdown() and rospy.Time.now() < takeoff_deadline:
                    pos = self._current_position()
                    if pos is not None:
                        alt = pos[2] if pos[2] >= 0.0 else -pos[2]
                        if alt >= self.min_takeoff_check_alt:
                            rospy.loginfo("Minimum altitude reached: %.2fm (threshold: %.1fm)",
                                         alt, self.min_takeoff_check_alt)
                            break
                    rospy.sleep(0.1)

                # PHASE 2: Wait for drone to reach target takeoff altitude
                rospy.loginfo("Waiting for target altitude (%.1fm)...", self.takeoff_alt)
                takeoff_deadline = rospy.Time.now() + rospy.Duration(15.0)
                while not rospy.is_shutdown() and rospy.Time.now() < takeoff_deadline:
                    pos = self._current_position()
                    if pos is not None:
                        alt = pos[2] if pos[2] >= 0.0 else -pos[2]
                        if alt >= self.takeoff_alt - 0.3:
                            rospy.loginfo("Target altitude reached: %.2fm", alt)
                            break
                    rospy.sleep(0.1)

                # PHASE 3: Stabilization time after reaching altitude
                rospy.loginfo("Stabilizing for 3 seconds...")
                rospy.sleep(3.0)

            if self.allow_auto_mode and self.start_auto_after_takeoff:
                deadline = rospy.Time.now() + rospy.Duration(10.0)
                while not rospy.is_shutdown() and rospy.Time.now() < deadline:
                    pos = self._current_position()
                    if pos is None:
                        rospy.sleep(0.2)
                        continue
                    alt = pos[2] if pos[2] >= 0.0 else -pos[2]
                    if alt >= self.takeoff_alt - 0.2:
                        break
                    rospy.sleep(0.2)
                if self.last_state is not None and self.last_state.armed and self.last_state.mode != self.auto_start_mode:
                    self._set_mode(self.auto_start_mode)
        self._reset_hidden()

    def _wait_until_airborne(self, timeout_s: float) -> bool:
        deadline = rospy.Time.now() + rospy.Duration(timeout_s)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.last_state is not None and self.last_state.armed:
                pos = self._current_position()
                if pos is not None:
                    alt = pos[2] if pos[2] >= 0.0 else -pos[2]
                    if alt >= (self.takeoff_alt - 0.2):
                        return True
            rospy.sleep(0.2)
        return False

    def _reset_hidden(self):
        if self.model is None:
            return
        device = torch.device(self.device_name)
        h = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=device)
        c = torch.zeros(self.lstm_layers, 1, self.hidden_size, device=device)
        self.hidden = (h, c)

    def _current_position(self) -> Optional[Tuple[float, float, float]]:
        if self.last_pose is not None and self._last_pose_stamp is not None:
            age = (rospy.Time.now() - self._last_pose_stamp).to_sec()
            if 0.0 <= age <= self.pose_timeout:
                pos = self.last_pose.pose.position
                z = float(pos.z)
                if z < -0.5 and not self._ned_pose_notice:
                    rospy.logwarn("MAVROS local_position appears NED (z<0). Using abs(z) for altitude checks.")
                    self._ned_pose_notice = True
                return float(pos.x), float(pos.y), z
            rospy.logwarn_throttle(
                5.0,
                "MAVROS local_position stale (age=%.2fs); falling back to Gazebo pose",
                age,
            )
        if self.last_pose is not None and self._last_pose_stamp is None:
            pos = self.last_pose.pose.position
            return float(pos.x), float(pos.y), float(pos.z)
        if self.last_local_odom is not None and self._last_local_odom_stamp is not None:
            age = (rospy.Time.now() - self._last_local_odom_stamp).to_sec()
            if 0.0 <= age <= self.pose_timeout:
                pos = self.last_local_odom.pose.pose.position
                z = float(pos.z)
                if z < -0.5 and not self._ned_pose_notice:
                    rospy.logwarn("MAVROS odometry appears NED (z<0). Using abs(z) for altitude checks.")
                    self._ned_pose_notice = True
                return float(pos.x), float(pos.y), z
        if self.last_gazebo_pose is not None:
            if not self._pose_source_notice:
                rospy.logwarn("Using Gazebo pose for trainer (local_position unavailable)")
                self._pose_source_notice = True
            pos = self.last_gazebo_pose.pose.position
            return float(pos.x), float(pos.y), float(pos.z)
        return None

    def _using_local_pose(self) -> bool:
        now = rospy.Time.now()
        if self._last_pose_stamp is not None and self.last_pose is not None:
            age = (now - self._last_pose_stamp).to_sec()
            if 0.0 <= age <= self.pose_timeout:
                return True
        if self._last_local_odom_stamp is not None and self.last_local_odom is not None:
            age = (now - self._last_local_odom_stamp).to_sec()
            if 0.0 <= age <= self.pose_timeout:
                return True
        return False

    def _get_current_yaw(self) -> float:
        """Get current drone yaw from IMU quaternion."""
        imu = self.latest_vecs.get("imu")
        if imu is None or len(imu) < 4:
            return 0.0
        # IMU vec format: [qx, qy, qz, qw, ax, ay, az]
        qx, qy, qz, qw = imu[0], imu[1], imu[2], imu[3]
        # Quaternion to yaw (rotation around Z axis)
        yaw = math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        return yaw

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-pi, pi] range."""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def _normalize_gps(self) -> Optional[np.ndarray]:
        """Normalize GPS relative to spawn position.

        Converts raw lat/lon/alt to meters relative to spawn,
        then normalizes to [-1, 1] range for neural network input.

        Returns:
            3-element array [norm_dlat, norm_dlon, norm_dalt] or None if GPS unavailable
        """
        gps = self.latest_vecs.get("gps")
        if gps is None or len(gps) < 3:
            return None

        # First GPS becomes spawn reference
        if self._spawn_gps is None:
            self._spawn_gps = np.array(gps, dtype=np.float64)
            rospy.loginfo("GPS spawn reference set: lat=%.6f lon=%.6f alt=%.1f",
                         self._spawn_gps[0], self._spawn_gps[1], self._spawn_gps[2])

        # Convert lat/lon difference to meters
        # 1 degree latitude ≈ 111,000 meters
        # 1 degree longitude ≈ 111,000 * cos(latitude) meters
        dlat_m = (gps[0] - self._spawn_gps[0]) * 111000.0
        dlon_m = (gps[1] - self._spawn_gps[1]) * 111000.0 * math.cos(math.radians(gps[0]))
        dalt_m = gps[2] - self._spawn_gps[2]

        # Normalize to [-1, 1] range
        # Assuming max 50m movement in XY, 10m in Z
        norm_gps = np.array([
            np.clip(dlat_m / 50.0, -1.0, 1.0),
            np.clip(dlon_m / 50.0, -1.0, 1.0),
            np.clip(dalt_m / 10.0, -1.0, 1.0),
        ], dtype=np.float32)

        return norm_gps

    def _build_single_obs(self) -> Optional[np.ndarray]:
        """Build a single observation frame (without stacking).

        Observation structure:
        - target_vec: [dir_x, dir_y, dir_z, norm_dist, sin(yaw_error), cos(yaw_error)]
          - dir_x/y/z: Normalized direction to target (world frame)
          - norm_dist: Normalized distance to target [0, 1]
          - yaw_error: Angle between drone heading and target direction (sin/cos encoded)
        - lidar: 180 normalized LIDAR distances
        - camera: 768 grayscale pixels (32x24)
        - imu: 7 values [qx,qy,qz,qw,ax,ay,az]
        - gyro: 3 values [gx,gy,gz]
        - gps: 3 values [norm_dlat, norm_dlon, norm_dalt] - normalized relative to spawn
        """
        # Check required sensors
        required_keys = ["lidar", "camera", "imu", "gyro", "gps"]
        if not all(self.latest_vecs.get(k) is not None for k in required_keys):
            return None

        # Get current position and compute direction to target
        current = self._current_position()
        if current is None:
            return None

        # Use local or world target based on pose source
        if self._using_local_pose():
            tx, ty, tz = self._target_local
        else:
            tx, ty, tz = self._target_world

        # Compute relative vector to target (normalized direction + distance)
        dx = tx - current[0]
        dy = ty - current[1]
        dz = tz - current[2]
        dist = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Normalize direction vector
        if dist > 1e-6:
            dir_x = dx / dist
            dir_y = dy / dist
            dir_z = dz / dist
        else:
            dir_x, dir_y, dir_z = 0.0, 0.0, 0.0

        # Compute yaw error: difference between drone heading and target direction
        # This helps the agent understand if it needs to turn
        current_yaw = self._get_current_yaw()
        target_yaw = math.atan2(dy, dx)  # Direction to target in XY plane
        yaw_error = self._wrap_angle(target_yaw - current_yaw)

        # Target info: [dir_x, dir_y, dir_z, normalized_distance, sin(yaw_error), cos(yaw_error)]
        # Using sin/cos encoding for yaw_error avoids discontinuity at +-pi
        norm_dist = min(dist / self.max_dist, 1.0)
        target_vec = np.array([
            dir_x, dir_y, dir_z,       # Direction to target (world frame)
            norm_dist,                  # Normalized distance
            math.sin(yaw_error),        # Yaw error (sin component)
            math.cos(yaw_error),        # Yaw error (cos component)
        ], dtype=np.float32)

        # Get normalized GPS (relative to spawn, in meters, clipped to [-1,1])
        norm_gps = self._normalize_gps()
        if norm_gps is None:
            return None

        parts = [
            target_vec,  # 6 values: direction to target + distance + yaw error
            self.latest_vecs["lidar"],
            self.latest_vecs["camera"],
            self.latest_vecs["imu"],
            self.latest_vecs["gyro"],
            norm_gps,  # 3 values: normalized GPS relative to spawn (meters -> [-1,1])
        ]
        return np.concatenate(parts).astype(np.float32)

    def _reset_frame_buffer(self):
        """Reset the frame buffer for a new episode."""
        self._frame_buffer = []

    def _build_obs(self) -> Optional[np.ndarray]:
        """
        Build observation with optional frame stacking.
        Frame stacking helps the agent understand temporal dynamics
        (e.g., velocity of moving obstacles).
        """
        single_obs = self._build_single_obs()
        if single_obs is None:
            return None

        if self.frame_stack_size <= 1:
            return single_obs

        # Add to frame buffer
        self._frame_buffer.append(single_obs)

        # Keep buffer at correct size
        while len(self._frame_buffer) > self.frame_stack_size:
            self._frame_buffer.pop(0)

        # If buffer not full yet, pad with copies of current frame
        if len(self._frame_buffer) < self.frame_stack_size:
            padding = [single_obs] * (self.frame_stack_size - len(self._frame_buffer))
            frames = padding + self._frame_buffer
        else:
            frames = self._frame_buffer

        # Stack frames along feature dimension
        return np.concatenate(frames).astype(np.float32)

    def _distance_to_target(self) -> Optional[float]:
        current = self._current_position()
        if current is None:
            return None
        if self._using_local_pose():
            tx, ty, tz = self._target_local
        else:
            tx, ty, tz = self._target_world
        dx = current[0] - tx
        dy = current[1] - ty
        dz = current[2] - tz
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))

    def _distance_to_spawn(self) -> Optional[float]:
        """Calculate Euclidean distance from current position to spawn point."""
        current = self._current_position()
        if current is None:
            return None
        dx = current[0] - self.spawn_x
        dy = current[1] - self.spawn_y
        dz = current[2] - self.spawn_z
        return float(math.sqrt(dx * dx + dy * dy + dz * dz))

    def _build_route(self, action: np.ndarray) -> Path:
        direction = action[:3].astype(np.float32)
        if not self.use_z_action:
            direction[2] = 0.0
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            direction = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            direction = direction / norm

        # Calculate yaw angle from movement direction (so drone faces forward)
        # LIDAR is front 180 degrees, so drone nose should point in movement direction
        yaw = math.atan2(direction[1], direction[0])
        # Convert yaw to quaternion (rotation around Z axis)
        qz = math.sin(yaw * 0.5)
        qw = math.cos(yaw * 0.5)

        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = self.frame_id

        for i in range(1, self.route_length + 1):
            pose = PoseStamped()
            pose.header = path.header
            pose.pose.position.x = float(direction[0] * self.step_size * i)
            pose.pose.position.y = float(direction[1] * self.step_size * i)
            z = float(direction[2] * self.step_size * i)
            if self.max_route_z > 0.0:
                z = max(-self.max_route_z, min(self.max_route_z, z))
            pose.pose.position.z = z
            # Set orientation so drone faces movement direction
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = qz
            pose.pose.orientation.w = qw
            path.poses.append(pose)
        return path

    def _wait_step(self):
        if self.use_reached_event:
            self.reached_event.clear()
            self.reached_event.wait(self.step_timeout)
        else:
            rospy.sleep(self.step_timeout)

    def _update_reward_stats(self, reward: float):
        """Update running mean and variance for reward normalization."""
        self._reward_count += 1
        delta = reward - self._reward_mean
        self._reward_mean += delta / self._reward_count
        delta2 = reward - self._reward_mean
        self._reward_var += delta * delta2

    def _normalize_reward(self, reward: float) -> float:
        """Normalize reward using running statistics and clip."""
        if not self.use_reward_normalization:
            return np.clip(reward, -self.reward_clip, self.reward_clip)

        # Update statistics
        self._update_reward_stats(reward)

        # Normalize (avoid division by zero)
        if self._reward_count > 1:
            std = math.sqrt(self._reward_var / (self._reward_count - 1))
            if std > 1e-8:
                reward = (reward - self._reward_mean) / std

        # Clip to prevent extreme values
        return float(np.clip(reward, -self.reward_clip, self.reward_clip))

    def _compute_reward(self, prev_dist: float, step_count: int) -> Tuple[float, bool, str]:
        """
        Compute reward using CONTINUOUS FLOW (Carrot-Stick) method.

        Episode terminates (done=True) ONLY when:
        1. COLLISION: UAV hits obstacle (LIDAR < collision_distance)
        2. OUT_OF_BOUNDS: UAV flies outside safe zone
        3. ALTITUDE_CRASH: UAV flies too low
        4. TIMEOUT: Episode exceeds max_steps

        When UAV reaches goal (distance < goal_radius):
        - Give intermediate bonus (+10)
        - Generate NEW random target
        - Continue episode (done=False) to preserve LSTM memory

        WARMUP PERIOD: First N steps have no termination checks (except timeout)
        to allow drone to stabilize after spawn.

        Returns: (reward, done, termination_reason)
        """
        dist = self._distance_to_target()
        if dist is None:
            return 0.0, False, ""

        current = self._current_position()
        min_lidar_dist = self._get_min_lidar_distance()

        # Filter out very low LIDAR readings (likely self-collision with drone body/propellers)
        # Real obstacles won't be closer than 0.05m - sensor noise or self-view
        if min_lidar_dist is not None and min_lidar_dist < 0.05:
            min_lidar_dist = None  # Ignore, not a real obstacle

        reward = 0.0
        done = False
        termination_reason = ""

        # =====================================================================
        # WARMUP PERIOD - Skip termination checks for first N steps
        # This prevents instant death due to sensor noise or spawn issues
        # =====================================================================
        in_warmup = step_count <= self.warmup_steps
        if in_warmup and step_count == 1:
            rospy.loginfo("WARMUP: Termination checks disabled for first %d steps", self.warmup_steps)

        # =====================================================================
        # EPISODE TERMINATION CONDITIONS (done = True)
        # These are the ONLY conditions that end an episode
        # SKIPPED during warmup period (except TIMEOUT)
        # =====================================================================

        # --- CONDITION 1: COLLISION - Hit an obstacle ---
        # Only check after warmup period
        if not in_warmup and min_lidar_dist is not None and min_lidar_dist < self.collision_distance:
            reward = self.crash_penalty
            done = True
            termination_reason = "COLLISION"
            pos_str = f"pos=({current[0]:.1f},{current[1]:.1f},{current[2]:.1f})" if current else "pos=unknown"
            rospy.logwarn("COLLISION! LIDAR=%.2fm %s step=%d goals=%d",
                         min_lidar_dist, pos_str, step_count, self._goals_reached_this_episode)
            return reward, done, termination_reason

        # --- CONDITION 2: OUT OF BOUNDS - Outside map boundaries ---
        # Only check after warmup period, use generous boundaries
        if not in_warmup and current:
            x, y, z = current[0], current[1], current[2]
            # Very generous bounds: map is roughly -30 to +25, allow 10m margin
            bound_margin = 10.0
            x_min_bound = self.map_x_min - bound_margin
            x_max_bound = self.map_x_max + bound_margin
            y_min_bound = self.map_y_min - bound_margin
            y_max_bound = self.map_y_max + bound_margin

            out_reason = ""
            if x < x_min_bound:
                out_reason = f"x={x:.1f} < x_min={x_min_bound:.1f}"
            elif x > x_max_bound:
                out_reason = f"x={x:.1f} > x_max={x_max_bound:.1f}"
            elif y < y_min_bound:
                out_reason = f"y={y:.1f} < y_min={y_min_bound:.1f}"
            elif y > y_max_bound:
                out_reason = f"y={y:.1f} > y_max={y_max_bound:.1f}"

            if out_reason:
                reward = self.crash_penalty
                done = True
                termination_reason = "OUT_OF_BOUNDS"
                rospy.logwarn("OUT_OF_BOUNDS! %s pos=(%.1f,%.1f,%.1f) step=%d",
                             out_reason, x, y, z, step_count)
                return reward, done, termination_reason

        # --- CONDITION 3: ALTITUDE CRASH - Too low ---
        # Only check after warmup period (drone may be taking off)
        if not in_warmup and current:
            alt = current[2] if current[2] >= 0.0 else -current[2]
            if alt < self.min_alt:
                reward = self.crash_penalty
                done = True
                termination_reason = "ALTITUDE_CRASH"
                rospy.logwarn("ALTITUDE_CRASH! alt=%.2fm < min=%.2fm pos=(%.1f,%.1f) step=%d",
                             alt, self.min_alt, current[0], current[1], step_count)
                return reward, done, termination_reason

        # --- CONDITION 4: TIMEOUT - Max steps reached (always checked) ---
        if step_count >= self.max_steps:
            reward = self.timeout_penalty
            done = True
            termination_reason = "TIMEOUT"
            rospy.loginfo("TIMEOUT! %d steps reached. Goals this episode: %d",
                         step_count, self._goals_reached_this_episode)
            return reward, done, termination_reason

        # --- CONDITION 5: DRIFT_EXCEEDED - Too far from spawn (Safety Net) ---
        # Prevents agent from wandering off and making learning impossible
        if not in_warmup:
            drift_dist = self._distance_to_spawn()
            if drift_dist is not None and drift_dist > self.max_drift_distance:
                reward = self.crash_penalty
                done = True
                termination_reason = "DRIFT_EXCEEDED"
                rospy.logwarn("DRIFT_EXCEEDED! %.1fm > %.1fm limit. Forcing respawn.",
                             drift_dist, self.max_drift_distance)
                return reward, done, termination_reason

        # =====================================================================
        # CARROT (HAVUÇ): INTERMEDIATE GOAL REACHED - Generate New Target
        # Episode continues, LSTM memory preserved!
        # =====================================================================

        if dist <= self.goal_radius:
            # Give intermediate bonus
            reward = self.intermediate_bonus
            self._goals_reached_this_episode += 1

            rospy.loginfo("GOAL REACHED! #%d at step %d. Distance: %.2fm. Generating new target...",
                         self._goals_reached_this_episode, step_count, dist)

            # Generate and set new random target
            new_target = self._generate_random_target()
            self._set_new_target(new_target[0], new_target[1], new_target[2])

            # Continue episode (LSTM memory preserved!)
            return reward, False, "INTERMEDIATE_GOAL"

        # =====================================================================
        # SOPA (STICK): DENSE REWARD SHAPING (episode continues)
        # =====================================================================

        # --- 1. PROGRESS REWARD (most important) ---
        # Positive if moving toward goal, negative if moving away
        progress = (prev_dist - dist) * self.progress_scale
        reward += progress

        # --- 2. EXISTENCE PENALTY (encourages faster completion) ---
        # Small negative reward per step to encourage efficiency
        reward -= self.step_penalty

        # --- 3. OBSTACLE PROXIMITY PENALTY (safety signal) ---
        # Gradual penalty as UAV gets closer to obstacles
        if min_lidar_dist is not None:
            warning_distance = 2.0  # Start warning below this
            if min_lidar_dist < warning_distance:
                # Linear penalty: closer = worse
                proximity_penalty = (warning_distance - min_lidar_dist) * 0.5
                reward -= proximity_penalty

        # --- 4. YAW ALIGNMENT PENALTY (prevents crab walking) ---
        # Penalize drone for not facing the target direction
        # This forces the drone to turn its nose toward the goal
        current_yaw = self._get_current_yaw()
        if current:
            tx, ty = self._target_local[:2] if self._using_local_pose() else self._target_world[:2]
            dx_yaw = tx - current[0]
            dy_yaw = ty - current[1]
            target_yaw = math.atan2(dy_yaw, dx_yaw)
            yaw_error = abs(self._wrap_angle(target_yaw - current_yaw))
            # yaw_error: 0 = perfectly aligned, pi = facing opposite direction
            # Normalize to [0, 1] and apply penalty
            yaw_penalty = (yaw_error / math.pi) * self.yaw_alignment_scale
            reward -= yaw_penalty

        # Clip reward to prevent extreme values
        reward = float(np.clip(reward, -self.reward_clip, self.reward_clip))

        return reward, done, termination_reason

    def _init_model(self, obs_dim: int):
        device = torch.device(self.device_name)
        self.model = PPOLSTMActorCritic(obs_dim, self.action_dim, self.hidden_size, self.lstm_layers).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self._reset_hidden()
        if self.checkpoint_path:
            checkpoint_dir = os.path.dirname(self.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            try:
                state = torch.load(self.checkpoint_path, map_location=device, weights_only=True)
                self.model.load_state_dict(state)
                rospy.loginfo("Loaded checkpoint: %s", self.checkpoint_path)
            except Exception as exc:
                rospy.logwarn("Checkpoint load failed: %s", exc)

    def _compute_gae(self, rewards, values, dones, last_value):
        adv = np.zeros_like(rewards, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(len(rewards))):
            next_value = last_value if t == len(rewards) - 1 else values[t + 1]
            next_nonterminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_nonterminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def train(self):
        self.mode_pub.publish(String(data=self.mode))
        self._wait_ready()
        if self.auto_param_tune:
            self._configure_fcu_params()
            if self._require_restart:
                rospy.logerr("GPS config updated. Restart ArduPilot then relaunch training.")
                return

        obs = self._build_obs()
        if obs is None:
            return
        self.obs_dim = int(obs.size)
        self._init_model(self.obs_dim)

        device = torch.device(self.device_name)

        epoch_index = 0
        if self.epochs <= 0:
            return

        while not rospy.is_shutdown():
            for _ in range(self.epochs):
                epoch_index += 1
                if rospy.is_shutdown():
                    return

                if self.reset_each_epoch:
                    # Apply curriculum configuration before reset
                    self._apply_curriculum_config()
                    self._reset_frame_buffer()
                    self._reset_episode()
                    if not self._wait_until_airborne(30.0):
                        rospy.logwarn("Episode start failed (not airborne). Retrying next epoch.")
                        continue
                    # Check spawn safety after sensors are ready
                    rospy.sleep(0.5)  # Wait for LIDAR data
                    if not self._check_spawn_safety():
                        rospy.logwarn("Unsafe spawn detected, will retry with new position next epoch")
                        # Don't continue here - let the episode run, agent will learn from it
                else:
                    # Safety Net: Check if agent drifted too far from spawn
                    drift_dist = self._distance_to_spawn()
                    if drift_dist is not None and drift_dist > self.max_drift_distance:
                        rospy.logwarn("Drift limit exceeded (%.1fm > %.1fm) - forcing respawn to base",
                                     drift_dist, self.max_drift_distance)
                        self._reset_episode()
                        if not self._wait_until_airborne(30.0):
                            rospy.logwarn("Respawn failed (not airborne). Continuing anyway.")
                    else:
                        # LSTM hidden state preserved across epochs for continuity
                        # Only reset when episode truly resets (reset_each_epoch=True)
                        # self._reset_hidden()  # DISABLED: Prevents "lobotomy" - agent keeps memory
                        self._reset_frame_buffer()
                obs_list: List[np.ndarray] = []
                action_list: List[np.ndarray] = []
                logp_list: List[float] = []
                value_list: List[float] = []
                reward_list: List[float] = []
                done_list: List[bool] = []

                prev_dist = self._distance_to_target() or 0.0
                step_count = 0
                termination_reason = ""

                # =====================================================================
                # EPISODE LOOP - Runs until done condition or max_steps (TIMEOUT)
                # =====================================================================
                for step_count in range(1, self.max_steps + 1):
                    obs = self._build_obs()
                    if obs is None or self.model is None or self.hidden is None:
                        rospy.sleep(0.1)
                        continue
                    if self.enforce_guided_only and self.last_state is not None and self.last_state.armed:
                        if self.last_state.mode != self.required_mode:
                            self._set_mode(self.required_mode)
                            rospy.sleep(0.2)
                            continue
                    obs_t = torch.from_numpy(obs).float().to(device)
                    with torch.no_grad():
                        action_t, logp_t, value_t, self.hidden = self.model.act_step(obs_t, self.hidden, self.mode)
                    action = action_t.cpu().numpy()

                    # Add bias toward target direction based on curriculum level
                    # Early training: strong guidance, later: more autonomy
                    target_bias = self._get_target_bias()
                    target_dir = obs[:3]  # dir_x, dir_y, dir_z from observation
                    # Blend policy action with target direction
                    action[:3] = (1.0 - target_bias) * action[:3] + target_bias * target_dir

                    route = self._build_route(action)
                    self.route_pub.publish(route)
                    self._wait_step()

                    # Compute reward with step count for logging
                    reward, done, termination_reason = self._compute_reward(prev_dist, step_count)
                    dist = self._distance_to_target() or prev_dist
                    prev_dist = dist

                    obs_list.append(obs)
                    action_list.append(action)
                    logp_list.append(float(logp_t.item()))
                    value_list.append(float(value_t.item()))
                    reward_list.append(float(reward))
                    done_list.append(bool(done))

                    # Log progress every 50 steps
                    if step_count % 50 == 0:
                        rospy.loginfo("Epoch %d Step %d/%d: dist=%.2f, reward=%.2f, goals=%d",
                                     epoch_index, step_count, self.max_steps, dist, reward,
                                     self._goals_reached_this_episode)

                    if done:
                        break

                # =====================================================================
                # EPISODE END SUMMARY
                # =====================================================================
                # Log final episode stats
                rospy.loginfo("=== Episode %d Complete ===", epoch_index)
                rospy.loginfo("  Steps: %d, Goals Reached: %d, Reason: %s",
                             step_count, self._goals_reached_this_episode, termination_reason)

                # Ensure done is set for training (timeout is handled in _compute_reward)
                if not done_list or not done_list[-1]:
                    done_list[-1] = True

                if self.model is None or self.optimizer is None or not obs_list:
                    continue

                obs_seq = torch.from_numpy(np.stack(obs_list)).float().to(device)
                actions = torch.from_numpy(np.stack(action_list)).float().to(device)
                old_logp = torch.from_numpy(np.array(logp_list, dtype=np.float32)).to(device)
                values = np.array(value_list, dtype=np.float32)
                rewards = np.array(reward_list, dtype=np.float32)
                dones = np.array(done_list, dtype=np.float32)

                last_value = 0.0
                if not done_list[-1]:
                    with torch.no_grad():
                        last_obs = torch.from_numpy(obs_list[-1]).float().to(device)
                        last_value = float(self.model.act_step(last_obs, self.hidden, self.mode)[2].item())

                adv, returns = self._compute_gae(rewards, values, dones, last_value)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                adv_t = torch.from_numpy(adv).float().to(device)
                returns_t = torch.from_numpy(returns).float().to(device)

                for _ in range(self.update_epochs):
                    logp, entropy, value = self.model.evaluate_actions(obs_seq, actions, self.mode)
                    ratio = torch.exp(logp - old_logp)
                    clip_adv = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * adv_t
                    policy_loss = -(torch.min(ratio * adv_t, clip_adv)).mean()
                    value_loss = 0.5 * (returns_t - value).pow(2).mean()
                    entropy_loss = -entropy.mean()

                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()

                if self.checkpoint_path:
                    torch.save(self.model.state_dict(), self.checkpoint_path)

                # Record epoch metrics
                final_dist = self._distance_to_target()
                success = termination_reason == "SUCCESS"

                # Update curriculum learning based on success
                self._update_curriculum(success)

                self._record_epoch_metrics(
                    epoch=epoch_index,
                    steps=len(obs_list),
                    total_reward=float(rewards.sum()),
                    success=success,
                    final_dist=final_dist
                )

                # Detailed epoch summary
                rospy.loginfo("=== Epoch %d Complete ===", epoch_index)
                rospy.loginfo("  Result: %s | Steps: %d | Reward: %.2f | Final Dist: %.2fm",
                             termination_reason, len(obs_list), rewards.sum(),
                             final_dist if final_dist else -1)

                if not self.loop_forever and epoch_index >= self.epochs:
                    self._save_metrics(force=True)
                    return

            if not self.loop_forever:
                self._save_metrics(force=True)
                return


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["exploration", "tracking"], default="exploration")
    parser.add_argument("--epochs", type=int, default=10)
    return parser.parse_known_args()[0]


def main():
    args = _parse_args()
    node = PPOLSTMTrainerNode(args)
    node.train()


if __name__ == "__main__":
    main()
