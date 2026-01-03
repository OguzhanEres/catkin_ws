#!/usr/bin/env python3
"""
Setpoint follower node (ROS 2).

Consumes a smoothed Path (/agent/route_smoothed) expressed as local ENU offsets
relative to the vehicle position at receipt time, and drives the vehicle in
GUIDED by continuously publishing a position setpoint to MAVROS:
  /mavros/setpoint_position/local (geometry_msgs/PoseStamped)

Publishes a per-step "reached" event (std_msgs/Empty) on /agent/wp_reached when
the active target is within acceptance_radius.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from std_msgs.msg import Empty

from drl_ros2 import rospy_compat as rospy

try:
    from mavros_msgs.msg import State
except Exception:  # pragma: no cover
    State = object  # type: ignore


@dataclass
class Target:
    origin: Tuple[float, float, float]
    offset: Tuple[float, float, float]
    absolute: Tuple[float, float, float]
    reached_sent: bool = False


class SetpointFollowerNode:
    def __init__(self):
        rospy.init_node("setpoint_follower", anonymous=False)

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
        self.use_3d_distance = bool(rospy.get_param("~use_3d_distance", True))

        self.last_pose: Optional[PoseStamped] = None
        self.last_state: Optional[State] = None  # type: ignore[assignment]
        self.target: Optional[Target] = None
        self._ned_frame: Optional[bool] = None
        self._ned_votes = 0
        self._enu_votes = 0
        self._airborne_confirmed = False

        self.setpoint_pub = rospy.Publisher(self.setpoint_topic, PoseStamped, queue_size=1)
        self.reached_pub = rospy.Publisher(self.reached_topic, Empty, queue_size=1)

        self.pose_sub = rospy.Subscriber(self.pose_topic, PoseStamped, self._pose_cb, queue_size=1)
        if State is not object:
            self.state_sub = rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=1)
        self.route_sub = rospy.Subscriber(self.route_topic, Path, self._route_cb, queue_size=1)

        self.timer = rospy.Timer(rospy.Duration.from_sec(1.0 / max(1.0, self.publish_rate)), self._tick)

    def _pose_cb(self, msg: PoseStamped):
        self.last_pose = msg
        z = float(msg.pose.position.z)
        if z < -0.5:
            self._ned_votes += 1
            self._enu_votes = max(0, self._enu_votes - 1)
        elif z > 0.5:
            self._enu_votes += 1
            self._ned_votes = max(0, self._ned_votes - 1)
        if self._ned_votes >= 3:
            self._ned_frame = True
        elif self._enu_votes >= 3:
            self._ned_frame = False

    def _state_cb(self, msg: State):
        self.last_state = msg

    def _route_cb(self, msg: Path):
        if not msg.poses:
            return
        if self.last_pose is None:
            rospy.logwarn_throttle(2.0, "Pose not ready; skipping route")
            return
        if self.require_armed and (self.last_state is None or not getattr(self.last_state, "armed", False)):
            return
        if self.require_guided and (self.last_state is None or getattr(self.last_state, "mode", "") != "GUIDED"):
            return

        offset_point: Point = msg.poses[0].pose.position
        origin = (
            float(self.last_pose.pose.position.x),
            float(self.last_pose.pose.position.y),
            float(self.last_pose.pose.position.z),
        )
        offset = (float(offset_point.x), float(offset_point.y), float(offset_point.z))

        xy_norm = math.sqrt(offset[0] * offset[0] + offset[1] * offset[1])
        if self.max_xy_step > 0.0 and xy_norm > self.max_xy_step:
            scale = self.max_xy_step / max(1e-6, xy_norm)
            offset = (offset[0] * scale, offset[1] * scale, offset[2])

        absolute = (origin[0] + offset[0], origin[1] + offset[1], origin[2] + offset[2])

        ned = bool(self._ned_frame) if self._ned_frame is not None else False
        current_alt = abs(origin[2]) if ned else origin[2]
        if current_alt >= self.min_alt_for_control:
            self._airborne_confirmed = True
        safe_min = max(self.min_z, self.takeoff_alt if self._airborne_confirmed else self.min_z)
        safe_max = max(safe_min, float(self.max_target_z))
        desired_alt = max(safe_min, min(safe_max, abs(absolute[2]) if ned else absolute[2]))
        if ned:
            absolute = (absolute[0], absolute[1], -desired_alt)
        else:
            absolute = (absolute[0], absolute[1], desired_alt)

        self.target = Target(origin=origin, offset=offset, absolute=absolute, reached_sent=False)

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
        if self.require_armed and (self.last_state is None or not getattr(self.last_state, "armed", False)):
            return
        if self.require_guided and (self.last_state is None or getattr(self.last_state, "mode", "") != "GUIDED"):
            return
        if not self._airborne_confirmed:
            z = float(self.last_pose.pose.position.z)
            ned = bool(self._ned_frame) if self._ned_frame is not None else False
            alt = abs(z) if ned else z
            if alt >= self.min_alt_for_control:
                self._airborne_confirmed = True
            else:
                return

        sp = PoseStamped()
        sp.header.stamp = rospy.Time.now().to_msg()
        sp.header.frame_id = self.frame_id
        sp.pose.position.x = float(self.target.absolute[0])
        sp.pose.position.y = float(self.target.absolute[1])
        sp.pose.position.z = float(self.target.absolute[2])
        sp.pose.orientation.w = 1.0
        self.setpoint_pub.publish(sp)

        current = (
            float(self.last_pose.pose.position.x),
            float(self.last_pose.pose.position.y),
            float(self.last_pose.pose.position.z),
        )
        dist = self._distance_to_target(current, self.target.absolute)
        if dist <= self.acceptance_radius and not self.target.reached_sent:
            self.target.reached_sent = True
            self.reached_pub.publish(Empty())


def main():
    try:
        _ = SetpointFollowerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.shutdown()


if __name__ == "__main__":
    main()

