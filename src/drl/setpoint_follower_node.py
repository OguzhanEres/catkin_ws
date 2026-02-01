#!/usr/bin/env python3
"""
Setpoint follower node.

Consumes a smoothed Path (/agent/route_smoothed) expressed as local ENU *offsets*
relative to the vehicle position at receipt time, and drives the vehicle in
GUIDED by continuously publishing a position setpoint to MAVROS:
  /mavros/setpoint_position/local (geometry_msgs/PoseStamped)

Publishes a per-step "reached" event (std_msgs/Empty) on /agent/wp_reached when
the active target is within acceptance_radius.

This implements the "one waypoint at a time, wait for completion" control layer
without using the FCU mission interface (avoids wp_seq quirks).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math

import rospy
from geometry_msgs.msg import PoseStamped, Point
from nav_msgs.msg import Path
from mavros_msgs.msg import State
from gazebo_msgs.msg import ModelStates
from std_msgs.msg import Empty


@dataclass
class Target:
    origin: Tuple[float, float, float]
    offset: Tuple[float, float, float]
    absolute: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)  # x, y, z, w quaternion
    reached_sent: bool = False


class SetpointFollowerNode:
    def __init__(self):
        rospy.init_node("setpoint_follower", anonymous=False)

        self.model_name = rospy.get_param("~model_name", "iris_px4_sensors")
        self.route_topic = rospy.get_param("~route_topic", "/agent/route_smoothed")
        self.pose_topic = rospy.get_param("~pose_topic", "/mavros/local_position/pose")
        self.setpoint_topic = rospy.get_param("~setpoint_topic", "/mavros/setpoint_position/local")
        self.reached_topic = rospy.get_param("~reached_topic", "/agent/wp_reached")

        self.frame_id = rospy.get_param("~frame_id", "map")
        self.acceptance_radius = float(rospy.get_param("~acceptance_radius", 0.5))
        self.publish_rate = float(rospy.get_param("~publish_rate", 20.0))
        self.takeoff_alt = float(rospy.get_param("~takeoff_alt", 2.0))
        self.min_z = float(rospy.get_param("~min_z", 0.2))
        self.max_target_z = float(rospy.get_param("~max_target_z", self.takeoff_alt + 0.5))
        self.max_xy_step = float(rospy.get_param("~max_xy_step", 10.0))
        self.min_alt_for_control = float(rospy.get_param("~min_alt_for_control", self.takeoff_alt - 0.3))
        self.require_armed = bool(rospy.get_param("~require_armed", True))
        self.require_guided = bool(rospy.get_param("~require_guided", True))
        self.required_mode = rospy.get_param("~required_mode", "GUIDED")
        self.use_3d_distance = bool(rospy.get_param("~use_3d_distance", True))
        self.force_enu = bool(rospy.get_param("~force_enu", False))
        self.force_ned = bool(rospy.get_param("~force_ned", False))

        self.last_pose: Optional[PoseStamped] = None
        self.last_gazebo_pose: Optional[PoseStamped] = None  # NEW: Separate Truth Storage
        self.last_state: Optional[State] = None
        self.target: Optional[Target] = None
        self._ned_frame: Optional[bool] = None
        self._ned_votes = 0
        self._enu_votes = 0
        self._airborne_confirmed = False
        
        # Ground Truth support
        self._gazebo_index: Optional[int] = None
        self._gazebo_sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._gazebo_cb, queue_size=1)

        # Use setpoint_position/local (PoseStamped) - doesn't require global origin
        # Note: PX4 may ignore orientation from PoseStamped, but position works reliably
        self.setpoint_pub = rospy.Publisher('/mavros/setpoint_position/local', PoseStamped, queue_size=1)
        self.reached_pub = rospy.Publisher(self.reached_topic, Empty, queue_size=1)

        self.pose_sub = rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=1)
        self.route_sub = rospy.Subscriber(self.route_topic, Path, self._route_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / max(1.0, self.publish_rate)), self._tick)

        rospy.loginfo(
            "SetpointFollower params: route_topic=%s pose_topic=%s setpoint_topic=%s reached_topic=%s acceptance=%.2f rate=%.1f model=%s",
            self.route_topic,
            self.pose_topic,
            self.setpoint_topic,
            self.reached_topic,
            self.acceptance_radius,
            self.publish_rate,
            self.model_name
        )

    def _pose_cb(self, msg: PoseStamped):
        # Always update MAVROS pose for Relative Control
            
        self.last_pose = msg
        z = float(msg.pose.position.z)
        if self.force_enu:
            self._ned_frame = False
            self._ned_votes = 0
            self._enu_votes = 3
            return

    def _gazebo_cb(self, msg: ModelStates):
        if self._gazebo_index is None:
            try:
                self._gazebo_index = msg.name.index(self.model_name)
                rospy.loginfo("Found Gazebo model '%s' at index %d", self.model_name, self._gazebo_index)
            except ValueError:
                return
        
        # Save Ground Truth separately
        pose = msg.pose[self._gazebo_index]
        ps = PoseStamped()
        ps.header.stamp = rospy.Time.now()
        ps.header.frame_id = "map"  # Gazebo is global map
        ps.pose = pose
        
        self.last_gazebo_pose = ps
        # No need to overwrite last_pose, we use last_gazebo_pose for logic checks

        # No need for NED heuristics here, Gazebo is ENU.

    def _state_cb(self, msg: State):
        self.last_state = msg

    def _route_cb(self, msg: Path):
        if not msg.poses:
            return
        if self.last_pose is None:
            rospy.logwarn_throttle(2.0, "Pose not ready; skipping route")
            return
        # Skip armed/guided checks during prestream (before airborne)
        # This allows setpoint publishing for OFFBOARD mode transition
        if self._airborne_confirmed:
            if self.require_armed and (self.last_state is None or not self.last_state.armed):
                return
            if self.require_guided and (self.last_state is None or self.last_state.mode != self.required_mode):
                return

        offset_point: Point = msg.poses[0].pose.position
        origin = (
            float(self.last_pose.pose.position.x),
            float(self.last_pose.pose.position.y),
            float(self.last_pose.pose.position.z),
        )
        offset = (float(offset_point.x), float(offset_point.y), float(offset_point.z))

        # TRUTH FRAME (GAZEBO): Use for Safety Checks (Altitude)
        # If Gazebo available, use it. Else fallback to MAVROS (origin)
        if self.last_gazebo_pose is not None:
             true_z = float(self.last_gazebo_pose.pose.position.z)
             current_alt = true_z
        else:
             ned = bool(self._ned_frame) if self._ned_frame is not None else False
             current_alt = abs(origin[2]) if ned else origin[2]

        min_safe_alt = self.takeoff_alt - 0.5  # Need to be at least this high for XY movement

        if current_alt < min_safe_alt:
            # Drone too low - only allow vertical movement (climb), no XY
            rospy.logwarn_throttle(1.0, "Drone too low (%.2fm < %.2fm), blocking XY movement", current_alt, min_safe_alt)
            offset = (0.0, 0.0, offset[2])  # Zero out XY, keep Z

        # Clamp XY magnitude to reduce aggressive tilts.
        xy_norm = math.sqrt(offset[0] * offset[0] + offset[1] * offset[1])
        if self.max_xy_step > 0.0 and xy_norm > self.max_xy_step:
            scale = self.max_xy_step / max(1e-6, xy_norm)
            offset = (offset[0] * scale, offset[1] * scale, offset[2])

        absolute = (origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2])

        # Enforce safe Z target: do not climb/descend aggressively during training.
        ned = bool(self._ned_frame) if self._ned_frame is not None else False
        current_alt = abs(origin[2]) if ned else origin[2]
        if current_alt >= self.min_alt_for_control:
            self._airborne_confirmed = True
        safe_min = self.min_z  # Allow descent down to min_z (e.g., 0.5m) even after takeoff
        safe_max = max(safe_min, float(self.max_target_z))
        desired_alt = max(safe_min, min(safe_max, abs(absolute[2]) if ned else absolute[2]))
        if ned:
            absolute = (absolute[0], absolute[1], -desired_alt)
        else:
            absolute = (absolute[0], absolute[1], desired_alt)

        # Get orientation from route (drone should face movement direction)
        route_orientation = msg.poses[0].pose.orientation
        orientation = (
            float(route_orientation.x),
            float(route_orientation.y),
            float(route_orientation.z),
            float(route_orientation.w),
        )
        self.target = Target(origin=origin, offset=offset, absolute=absolute, orientation=orientation, reached_sent=False)
        rospy.loginfo_throttle(
            1.0,
            "New target: origin=(%.2f,%.2f,%.2f) offset=(%.2f,%.2f,%.2f) abs=(%.2f,%.2f,%.2f)",
            origin[0],
            origin[1],
            origin[2],
            offset[0],
            offset[1],
            offset[2],
            absolute[0],
            absolute[1],
            absolute[2],
        )

    def _distance_to_target(self, pose_xyz: Tuple[float, float, float], target_xyz: Tuple[float, float, float]) -> float:
        dx = pose_xyz[0] - target_xyz[0]
        dy = pose_xyz[1] - target_xyz[1]
        dz = pose_xyz[2] - target_xyz[2]
        if self.use_3d_distance:
            return float(math.sqrt(dx * dx + dy * dy + dz * dz))
        return float(math.sqrt(dx * dx + dy * dy))

    def _tick(self, _event):
        if self.last_pose is None or self.target is None:
            return
        # Always publish setpoint when we have a target - needed for OFFBOARD prestream
        # Skip armed/guided checks during prestream phase
        skip_checks = not self._airborne_confirmed
        if not skip_checks:
            if self.require_armed and (self.last_state is None or not self.last_state.armed):
                return
            if self.require_guided and (self.last_state is None or self.last_state.mode != self.required_mode):
                return
        # Update airborne status
        # Use Truth if available, else MAVROS
        if self.last_gazebo_pose is not None:
             true_z = float(self.last_gazebo_pose.pose.position.z)
             if true_z >= self.min_alt_for_control:
                 self._airborne_confirmed = True
        else:
             z = float(self.last_pose.pose.position.z)
             ned = bool(self._ned_frame) if self._ned_frame is not None else False
             alt = abs(z) if ned else z
             if alt >= self.min_alt_for_control:
                 self._airborne_confirmed = True

        # Publish setpoint using PoseStamped (doesn't require global origin)
        setpoint = PoseStamped()
        setpoint.header.stamp = rospy.Time.now()
        setpoint.header.frame_id = self.frame_id

        setpoint.pose.position.x = float(self.target.absolute[0])
        setpoint.pose.position.y = float(self.target.absolute[1])
        setpoint.pose.position.z = float(self.target.absolute[2])

        # Set orientation from target (yaw direction)
        qx, qy, qz, qw = self.target.orientation
        setpoint.pose.orientation.x = qx
        setpoint.pose.orientation.y = qy
        setpoint.pose.orientation.z = qz
        setpoint.pose.orientation.w = qw

        self.setpoint_pub.publish(setpoint)

        current = (
            float(self.last_pose.pose.position.x),
            float(self.last_pose.pose.position.y),
            float(self.last_pose.pose.position.z),
        )
        dist = self._distance_to_target(current, self.target.absolute)
        if dist <= self.acceptance_radius and not self.target.reached_sent:
            self.target.reached_sent = True
            self.reached_pub.publish(Empty())


if __name__ == "__main__":
    try:
        node = SetpointFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
