#!/usr/bin/env python3
"""
Env bridge for macro-step control with episode-aware routes.
Handles reset without restarting SITL: Gazebo reset + MAVROS prep.
Publishes /agent/route_raw as drl/Route (episode_id + Path).

IMPORTANT: Do NOT use SetModelState to move the drone - it corrupts IMU/EKF.
Only use /gazebo/reset_world which properly resets physics state.
"""
import time
from typing import Dict, Optional

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty
from mavros_msgs.srv import SetMode, CommandBool
from mavros_msgs.msg import State, StatusText

from drl.msg import Route
from drl_utils.timeouts import wait_until, call_with_retry


class EnvBridge:
    def __init__(self):
        self.episode_id = 0
        self.pose: Optional[PoseStamped] = None
        self.scan: Optional[LaserScan] = None
        self.state: Optional[State] = None
        self.takeoff_alt = rospy.get_param("~takeoff_alt", 1.3)
        self.last_reset_failed = False
        self.offboard_mode = rospy.get_param("~offboard_mode", "OFFBOARD")
        self.preflight_timeout = float(rospy.get_param("~preflight_timeout", 45.0))
        self.preflight_stable_time = float(rospy.get_param("~preflight_stable_time", 2.0))
        self._prearm_bad_until = rospy.Time(0)
        self._last_statustext = ""
        self._last_statustext_stamp = rospy.Time(0)
        self.stationary_reset_seconds = float(rospy.get_param("~stationary_reset_seconds", 15.0))
        self.stationary_distance_threshold = float(
            rospy.get_param("~stationary_distance_threshold", 0.2)
        )
        self.takeoff_wait_timeout = float(rospy.get_param("~takeoff_wait_timeout", 15.0))
        self.takeoff_min_alt_ratio = float(rospy.get_param("~takeoff_min_alt_ratio", 0.9))
        self._last_move_time = rospy.Time(0)
        self._last_move_pos = None
        self._prearm_blocking_phrases = (
            "PreArm:",
            "Need Position Estimate",
            "Accels inconsistent",
            "Gyros inconsistent",
            "EKF attitude is bad",
            "AHRS: not using configured AHRS type",
            "EKF variance",
            "EKF compass variance",
            "GPS Glitch",
            "Compass error",
        )

        rospy.Subscriber("/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=1)
        rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=1)
        rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=1)
        rospy.Subscriber("/mavros/statustext/recv", StatusText, self._statustext_cb, queue_size=10)
        self.route_pub = rospy.Publisher("/agent/route_raw", Route, queue_size=1)

        self.set_mode = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.reset_world = rospy.ServiceProxy("/gazebo/reset_world", Empty)

        self.goal = np.array([10.0, 0.0])
        self.collision_thresh = 0.5
        self.max_range = 15.0
        # min_alt is 80% of takeoff altitude - allows collision detection during low-altitude flight
        self.min_alt = self.takeoff_alt * 0.8
        self.max_steps = 400
        self.last_action = np.zeros(3, dtype=np.float32)
        self.step_timeout = 0.05
        self.step_count = 0

    def _pose_cb(self, msg):  # noqa: D401
        self.pose = msg
        self._update_stationary_tracker(msg)

    def _scan_cb(self, msg):  # noqa: D401
        self.scan = msg

    def _state_cb(self, msg):  # noqa: D401
        self.state = msg

    def _statustext_cb(self, msg: StatusText):
        text = (msg.text or "").strip()
        if not text:
            return
        now = rospy.Time.now()
        self._last_statustext = text
        self._last_statustext_stamp = now
        
        # DEBUG: Print all FCU messages to understand arming refusal
        rospy.loginfo(f"FCU (StatusText): {text}")
        
        for phrase in self._prearm_blocking_phrases:
            if phrase in text:
                self._prearm_bad_until = now + rospy.Duration.from_sec(max(0.5, self.preflight_stable_time))
                break

    def _wait_for_prearm_clear(self) -> bool:
        deadline = rospy.Time.now() + rospy.Duration.from_sec(self.preflight_timeout)
        rate = rospy.Rate(5)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.state is None or not self.state.connected:
                rate.sleep()
                continue
            if self.pose is None:
                rate.sleep()
                continue
            if rospy.Time.now() <= self._prearm_bad_until:
                rate.sleep()
                continue
            return True
        rospy.logwarn("Preflight gate timeout. Last status: %s", self._last_statustext)
        return False

    def reset(self):
        self.episode_id += 1
        self.last_reset_failed = False
        is_first_episode = (self.episode_id == 1)
        self._last_move_time = rospy.Time(0)
        self._last_move_pos = None
        self._publish_reset_route()

        rospy.loginfo(f"========== EPISODE {self.episode_id} RESET {'(FIRST)' if is_first_episode else ''} ==========")
        
        # Step 1: Disarm and reset world (skip on first episode - EKF already stable)
        if not is_first_episode:
            if self.state and self.state.armed:
                rospy.loginfo("Step 1: Disarming...")
                call_with_retry(lambda: self.arm_srv(False))
                time.sleep(0.5)
            else:
                rospy.loginfo("Step 1: Already disarmed, skipping disarm command.")
            
            rospy.loginfo("Resetting Gazebo world (drone returns to initial pose)...")
            try:
                self.reset_world()
            except Exception as e:
                rospy.logwarn(f"reset_world failed: {e}")
            
            rospy.loginfo("Waiting for EKF stabilization after reset...")
            time.sleep(8.0)
        else:
            rospy.loginfo("First episode - skipping disarm/reset (EKF already stable)")
            time.sleep(1.0)
        
        if not self._wait_for_prearm_clear():
            rospy.logwarn("Preflight checks not ready; aborting reset.")
            self.last_reset_failed = True
            return self._get_obs()
        
        # Step 2: Pre-stream setpoints before OFFBOARD
        rospy.loginfo("Step 2: Pre-streaming setpoints...")
        self._prestream_setpoints(duration=2.0)
        
        # Step 3: Set OFFBOARD mode (PX4)
        rospy.loginfo("Step 3: Setting OFFBOARD mode...")
        call_with_retry(lambda: self.set_mode(base_mode=0, custom_mode=self.offboard_mode))
        time.sleep(0.5)
        
        # Step 4: ARM - use multiple methods for reliability
        rospy.loginfo("Step 4: Arming...")
        
        # Method 1: Normal arm service
        for attempt in range(3):
            try:
                resp = self.arm_srv(True)
                rospy.loginfo(f"Arm service response: {resp.success}")
            except Exception as e:
                rospy.logwarn(f"Arm service failed: {e}")
            time.sleep(0.5)
        
        # Wait for arm to take effect (don't check state, just wait)
        rospy.loginfo("Waiting 3 seconds for arm to take effect...")
        time.sleep(3.0)
        
        # Check armed state but don't abort if not armed - try anyway
        armed = False
        if self.state is not None:
            armed = self.state.armed
            rospy.loginfo(f"Armed state from callback: {armed}")
        
        # Also try direct topic read
        try:
            state_msg = rospy.wait_for_message("/mavros/state", State, timeout=1.0)
            armed = state_msg.armed
            rospy.loginfo(f"Armed state from topic: {armed}")
        except Exception as e:
            rospy.logwarn(f"Could not read state topic: {e}")
        
        if armed:
            rospy.loginfo("Armed successfully!")
        else:
            rospy.logerr("CRITICAL: Arming failed after all attempts (Service + Force Arm). Aborting episode reset.")
            self.last_reset_failed = True
            return self._get_obs()
        
        time.sleep(0.5)
        
        # Step 5: Continue streaming setpoints with takeoff altitude
        rospy.loginfo(f"Step 5: Sending takeoff setpoints to {self.takeoff_alt}m...")
        self._stream_takeoff_setpoints(duration=2.0)

        # Wait for takeoff completion before allowing policy setpoints
        if not self._wait_for_takeoff():
            rospy.logwarn("Takeoff did not complete; aborting reset to trigger re-reset.")
            self.last_reset_failed = True
            return self._get_obs()
        
        if self.pose:
            rospy.loginfo(f"Ready! Current altitude: {self.pose.pose.position.z:.2f}m")
        
        self.last_action[:] = 0.0
        self.step_count = 0
        return self._get_obs()

    def _prestream_setpoints(self, duration=1.0):
        """Stream current position setpoints before arming."""
        if self.pose is None:
            rospy.logwarn("No pose available for prestream")
            return
        start = time.time()
        while time.time() - start < duration and not rospy.is_shutdown():
            self.route_pub.publish(self._make_route_msg(Point(0.0, 0.0, 0.0)))
            time.sleep(self.step_timeout)

    def _stream_takeoff_setpoints(self, duration=3.0):
        """Stream setpoints at takeoff altitude after arming."""
        if self.pose is None:
            rospy.logwarn("No pose available for takeoff setpoints")
            return
        start = time.time()
        rate = rospy.Rate(20)  # 20 Hz setpoint streaming
        while time.time() - start < duration and not rospy.is_shutdown():
            current_z = float(self.pose.pose.position.z)
            dz = max(self.takeoff_alt - current_z, 0.0)
            self.route_pub.publish(self._make_route_msg(Point(0.0, 0.0, dz)))
            rate.sleep()

    def _wait_for_takeoff(self) -> bool:
        if self.pose is None:
            rospy.logwarn("No pose available for takeoff wait; using timeout only.")
        target_alt = self.takeoff_alt * self.takeoff_min_alt_ratio
        deadline = rospy.Time.now() + rospy.Duration.from_sec(self.takeoff_wait_timeout)
        rate = rospy.Rate(5)
        rospy.loginfo(
            f"Waiting for takeoff: target >= {target_alt:.2f}m (timeout {self.takeoff_wait_timeout:.1f}s)..."
        )
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            if self.pose and self.pose.pose.position.z >= target_alt:
                return True
            rate.sleep()
        rospy.logwarn("Takeoff wait timed out; proceeding anyway.")
        return False

    def macro_step(self, action, route_length=3, step_size_scale=1.0, wait_time=1.0):
        """Publish route for this action and wait while follower executes."""
        if self.last_reset_failed:
            # immediately end episode if reset/arm failed
            return self._get_obs(), 0.0, True, {"reset_failed": True}
        if self._stationary_too_long():
            rospy.logwarn("Stationary timeout exceeded; triggering reset.")
            return self._get_obs(), -5.0, True, {"done_reason": "stationary"}
        route = self._build_route(action, route_length, step_size_scale)
        self.route_pub.publish(route)
        time.sleep(wait_time)
        obs = self._get_obs()
        reward, done, info = self.compute_reward(obs, action)
        self.step_count += 1
        if self.step_count >= self.max_steps:
            info["truncated"] = True
            done = True
        return obs, reward, done, info

    def _update_stationary_tracker(self, msg: PoseStamped):
        if msg is None:
            return
        pos = msg.pose.position
        now = rospy.Time.now()
        if self._last_move_pos is None:
            self._last_move_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            self._last_move_time = now
            return
        last = self._last_move_pos
        dist = np.linalg.norm(np.array([pos.x, pos.y, pos.z], dtype=np.float32) - last)
        if dist >= self.stationary_distance_threshold:
            self._last_move_pos = np.array([pos.x, pos.y, pos.z], dtype=np.float32)
            self._last_move_time = now

    def _stationary_too_long(self) -> bool:
        if self._last_move_time == rospy.Time(0) or self._last_move_pos is None:
            return False
        elapsed = (rospy.Time.now() - self._last_move_time).to_sec()
        return elapsed >= self.stationary_reset_seconds

    def _build_route(self, action, route_length, scale):
        action = np.asarray(action, dtype=np.float32)
        route = Route()
        route.episode_id = self.episode_id
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        norm = np.linalg.norm(action)
        direction = np.array([1.0, 0.0, 0.0], dtype=np.float32) if norm < 1e-6 else action / norm
        for i in range(1, route_length + 1):
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position.x = float(direction[0] * scale * i)
            ps.pose.position.y = float(direction[1] * scale * i)
            ps.pose.position.z = float(direction[2] * scale * i)
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        route.path = path
        return route

    def _make_route_msg(self, offset: Point):
        route = Route()
        route.episode_id = self.episode_id
        path = Path()
        path.header.stamp = rospy.Time.now()
        path.header.frame_id = "map"
        ps = PoseStamped()
        ps.header = path.header
        ps.pose.position = offset
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)
        route.path = path
        return route

    def _publish_reset_route(self):
        route = Route()
        route.episode_id = self.episode_id
        route.path = Path()
        route.path.header.stamp = rospy.Time.now()
        route.path.header.frame_id = "map"
        self.route_pub.publish(route)

    def compute_reward(self, obs: Dict, action):
        pos = np.array(obs["pos"][:2])
        dist = np.linalg.norm(self.goal - pos)
        reward = -0.01  # time penalty
        reward += -0.1 * dist  # progress shaping
        done = False
        info = {}
        # Check for crash (flip/tumble)
        if obs.get("crashed", False):
            reward -= 10.0
            done = True
            info["done_reason"] = "crash"
        if obs["collision"]:
            reward -= 5.0
            done = True
            info["done_reason"] = "collision"
        if dist < 0.5:
            reward += 5.0
            done = True
            info["done_reason"] = "goal"
        # action smoothness
        reward -= 0.05 * float(np.linalg.norm(action - self.last_action))
        self.last_action = action.copy()
        return reward, done, info

    def _get_obs(self):
        collision = False
        min_scan = self.max_range
        if self.scan:
            # Filter out invalid readings (0, inf, nan)
            valid_ranges = [r for r in self.scan.ranges if r > 0.1 and r < self.max_range and not np.isnan(r) and not np.isinf(r)]
            if valid_ranges:
                min_scan = min(valid_ranges)
            else:
                min_scan = self.max_range
            
            # Only check collision if drone is above ground (z > min_alt)
            # This prevents false collision detection during takeoff
            pos_z = 0.0
            if self.pose:
                pos_z = self.pose.pose.position.z
            
            if pos_z > self.min_alt:
                collision = min_scan < self.collision_thresh
            # When on ground or during takeoff, no collision detection
            
        pos = [0.0, 0.0, 0.0]
        crashed = False
        if self.pose:
            p = self.pose.pose.position
            pos = [p.x, p.y, p.z]
            # Check for crash: excessive tilt (roll/pitch > 60 degrees)
            q = self.pose.pose.orientation
            # Calculate roll and pitch from quaternion
            sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
            cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = abs(np.arctan2(sinr_cosp, cosr_cosp))
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = abs(np.arcsin(np.clip(sinp, -1.0, 1.0)))
            # If roll or pitch > 60 degrees (1.05 rad), consider it a crash
            if roll > 1.05 or pitch > 1.05:
                crashed = True
                rospy.logwarn(f"CRASH detected! Roll: {np.degrees(roll):.1f}° Pitch: {np.degrees(pitch):.1f}°")
        return {"pos": pos, "min_scan": min_scan, "collision": collision, "crashed": crashed}
