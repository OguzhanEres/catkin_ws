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
        self.max_steps = int(rospy.get_param("~max_steps", 120))
        self.update_epochs = int(rospy.get_param("~update_epochs", 5))
        self.gamma = float(rospy.get_param("~gamma", 0.98))
        self.gae_lambda = float(rospy.get_param("~gae_lambda", 0.95))
        self.clip_ratio = float(rospy.get_param("~clip_ratio", 0.2))
        self.lr = float(rospy.get_param("~lr", 3e-4))
        self.value_coef = float(rospy.get_param("~value_coef", 0.5))
        self.entropy_coef = float(rospy.get_param("~entropy_coef", 0.01))
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

        self.target_x = float(rospy.get_param("~target_x", 0.0))
        self.target_y = float(rospy.get_param("~target_y", 18.0))
        self.target_z = float(rospy.get_param("~target_z", 1.0))
        self.target_in_world = bool(rospy.get_param("~target_in_world", True))
        self.goal_radius = float(rospy.get_param("~goal_radius", 1.2))
        self.step_penalty = float(rospy.get_param("~step_penalty", 0.02))
        self.progress_scale = float(rospy.get_param("~progress_scale", 1.0))
        self.success_bonus = float(rospy.get_param("~success_bonus", 5.0))
        self.crash_penalty = float(rospy.get_param("~crash_penalty", -5.0))
        self.min_alt = float(rospy.get_param("~min_alt", 0.1))
        self.max_dist = float(rospy.get_param("~max_dist", 80.0))

        self.spawn_x = float(rospy.get_param("~spawn_x", -25.0))
        self.spawn_y = float(rospy.get_param("~spawn_y", -25.0))
        self.spawn_z = float(rospy.get_param("~spawn_z", 0.1))
        self.spawn_yaw = float(rospy.get_param("~spawn_yaw", 0.785))
        self.model_name = rospy.get_param("~model_name", "iris_with_lidar_camera")
        self.takeoff_alt = float(rospy.get_param("~takeoff_alt", 2.0))
        self.send_takeoff_cmd = bool(rospy.get_param("~send_takeoff_cmd", False))
        self.reset_wait = float(rospy.get_param("~reset_wait", 1.0))

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

    def _reset_pose(self):
        state = ModelState()
        state.model_name = self.model_name
        state.pose.position.x = self.spawn_x
        state.pose.position.y = self.spawn_y
        state.pose.position.z = self.spawn_z
        qz = math.sin(self.spawn_yaw * 0.5)
        qw = math.cos(self.spawn_yaw * 0.5)
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
        except rospy.ServiceException as exc:
            rospy.logwarn("Reset pose failed: %s", exc)

    def _reset_episode(self):
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
                rospy.sleep(1.0)

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

    def _build_obs(self) -> Optional[np.ndarray]:
        if not all(v is not None for v in self.latest_vecs.values()):
            return None
        parts = [
            self.latest_vecs["lidar"],
            self.latest_vecs["camera"],
            self.latest_vecs["imu"],
            self.latest_vecs["gyro"],
            self.latest_vecs["gps"],
        ]
        return np.concatenate(parts).astype(np.float32)

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

    def _build_route(self, action: np.ndarray) -> Path:
        direction = action[:3].astype(np.float32)
        if not self.use_z_action:
            direction[2] = 0.0
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
            z = float(direction[2] * self.step_size * i)
            if self.max_route_z > 0.0:
                z = max(-self.max_route_z, min(self.max_route_z, z))
            pose.pose.position.z = z
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        return path

    def _wait_step(self):
        if self.use_reached_event:
            self.reached_event.clear()
            self.reached_event.wait(self.step_timeout)
        else:
            rospy.sleep(self.step_timeout)

    def _compute_reward(self, prev_dist: float) -> Tuple[float, bool]:
        dist = self._distance_to_target()
        if dist is None:
            return 0.0, False
        reward = (prev_dist - dist) * self.progress_scale - self.step_penalty
        done = False
        if dist <= self.goal_radius:
            reward += self.success_bonus
            done = True
        if dist >= self.max_dist:
            reward += self.crash_penalty
            done = True
        current = self._current_position()
        if current:
            alt = current[2] if current[2] >= 0.0 else -current[2]
            if alt < self.min_alt:
                reward += self.crash_penalty
                done = True
        return reward, done

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
                    self._reset_episode()
                    if not self._wait_until_airborne(30.0):
                        rospy.logwarn("Episode start failed (not airborne). Retrying next epoch.")
                        continue
                else:
                    self._reset_hidden()
                obs_list: List[np.ndarray] = []
                action_list: List[np.ndarray] = []
                logp_list: List[float] = []
                value_list: List[float] = []
                reward_list: List[float] = []
                done_list: List[bool] = []

                prev_dist = self._distance_to_target() or 0.0

                for _ in range(self.max_steps):
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

                    route = self._build_route(action)
                    self.route_pub.publish(route)
                    self._wait_step()

                    reward, done = self._compute_reward(prev_dist)
                    dist = self._distance_to_target() or prev_dist
                    prev_dist = dist

                    obs_list.append(obs)
                    action_list.append(action)
                    logp_list.append(float(logp_t.item()))
                    value_list.append(float(value_t.item()))
                    reward_list.append(float(reward))
                    done_list.append(bool(done))

                    if done:
                        break

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
                success = done_list[-1] if done_list else False
                if success and final_dist is not None and final_dist <= self.goal_radius:
                    success = True
                else:
                    success = False
                self._record_epoch_metrics(
                    epoch=epoch_index,
                    steps=len(obs_list),
                    total_reward=float(rewards.sum()),
                    success=success,
                    final_dist=final_dist
                )
                
                rospy.loginfo("Epoch %d done, steps=%d, reward=%.2f", epoch_index, len(obs_list), rewards.sum())

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
