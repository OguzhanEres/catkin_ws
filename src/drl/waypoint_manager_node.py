#!/usr/bin/env python3
"""
Waypoint manager node.
Consumes a smoothed Path and sends waypoints sequentially via MAVLink (MAVROS).
"""
from typing import List, Optional, Tuple

import math

import rospy
from geometry_msgs.msg import Point
from nav_msgs.msg import Path
from mavros_msgs.msg import Waypoint, WaypointReached
from mavros_msgs.msg import State
from mavros_msgs.srv import SetMode, WaypointPush, WaypointPushRequest, WaypointClear, WaypointSetCurrent
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import Float32MultiArray


class WaypointManagerNode:
    def __init__(self):
        rospy.init_node("waypoint_manager", anonymous=False)

        self.route_topic = rospy.get_param("~route_topic", "/agent/route_smoothed")
        self.frame_name = rospy.get_param("~frame", "global")
        self.acceptance_radius = float(rospy.get_param("~acceptance_radius", 1.0))
        self.command = int(rospy.get_param("~command", 16))  # MAV_CMD_NAV_WAYPOINT
        self.gps_topic = rospy.get_param("~gps_topic", "/agent/gps_vec")
        self.gps_msg_type = rospy.get_param("~gps_msg_type", "float32_multiarray").strip().lower()
        self.auto_mode_on_push = bool(rospy.get_param("~auto_mode_on_push", True))
        self.auto_mode = rospy.get_param("~auto_mode", "AUTO")
        self.guided_mode = rospy.get_param("~guided_mode", "GUIDED")
        self.takeoff_alt = float(rospy.get_param("~takeoff_alt", 2.0))
        self.takeoff_margin = float(rospy.get_param("~takeoff_margin", 0.3))
        self.always_send_takeoff = bool(rospy.get_param("~always_send_takeoff", True))
        self.include_takeoff_item = bool(rospy.get_param("~include_takeoff_item", True))
        self.min_auto_alt = float(rospy.get_param("~min_auto_alt", self.takeoff_alt - self.takeoff_margin))
        self.push_retries = int(rospy.get_param("~push_retries", 3))
        self.clear_retries = int(rospy.get_param("~clear_retries", 3))
        self.retry_wait = float(rospy.get_param("~retry_wait", 0.3))
        self.clear_on_start = bool(rospy.get_param("~clear_on_start", True))
        self.log_mission_details = bool(rospy.get_param("~log_mission_details", True))

        self.queue: List[Point] = []
        self.pending_queue: Optional[List[Point]] = None
        self.awaiting_reached = False
        self.expected_reached_seq: Optional[int] = None
        self._auto_mode_pending = False

        self.last_state: Optional[State] = None

        self.latest_gps: Optional[Tuple[float, float, float]] = None
        self.origin_gps: Optional[Tuple[float, float, float]] = None
        self.pending_origin_gps: Optional[Tuple[float, float, float]] = None
        self.home_alt: Optional[float] = None
        self._last_pushed_count: int = 0

        self.push_srv = self._wait_service("/mavros/mission/push", WaypointPush)
        self.clear_srv = self._wait_service("/mavros/mission/clear", WaypointClear)
        self.set_mode_srv = self._wait_service("/mavros/set_mode", SetMode)
        self.set_current_srv = self._wait_service("/mavros/mission/set_current", WaypointSetCurrent)

        self.route_sub = rospy.Subscriber(self.route_topic, Path, self._route_cb, queue_size=1)
        self.reached_sub = rospy.Subscriber("/mavros/mission/reached", WaypointReached, self._reached_cb, queue_size=1)
        if self.gps_msg_type in ("navsatfix", "nav_sat_fix", "nav_sat_fix_msg"):
            self.gps_sub = rospy.Subscriber(self.gps_topic, NavSatFix, self._gps_fix_cb, queue_size=1)
        else:
            self.gps_sub = rospy.Subscriber(self.gps_topic, Float32MultiArray, self._gps_cb, queue_size=1)
        self.state_sub = rospy.Subscriber("/mavros/state", State, self._state_cb, queue_size=1)

        rospy.loginfo(
            "WaypointManager params: route_topic=%s frame=%s acceptance=%.2f auto_mode=%s guided_mode=%s takeoff_alt=%.2f",
            self.route_topic,
            self.frame_name,
            self.acceptance_radius,
            self.auto_mode,
            self.guided_mode,
            self.takeoff_alt,
        )
        if self.clear_on_start:
            self._set_mode(self.guided_mode)
            self._clear_mission()

    def _state_cb(self, msg: State):
        self.last_state = msg
        if self._auto_mode_pending and msg.armed and self._can_start_auto():
            self._auto_mode_pending = False
            if msg.mode != self.auto_mode:
                self._set_mode(self.auto_mode)

    def _wait_service(self, name, srv_type):
        rospy.loginfo("Waiting for MAVROS service: %s", name)
        rospy.wait_for_service(name)
        return rospy.ServiceProxy(name, srv_type)

    def _route_cb(self, msg: Path):
        points = [pose.pose.position for pose in msg.poses]
        if not points:
            return
        if self.log_mission_details:
            first = points[0]
            rospy.loginfo_throttle(
                1.0,
                "Route received (%d poses). First ENU offset: x=%.2f y=%.2f z=%.2f",
                len(points),
                first.x,
                first.y,
                first.z,
            )
        if self._needs_gps() and self.latest_gps is None:
            rospy.logwarn_throttle(2.0, "GPS not ready; skipping route")
            return
        if self.awaiting_reached:
            self.pending_queue = points
            if self._needs_gps():
                self.pending_origin_gps = self.latest_gps
            return
        self.queue = points
        if self._needs_gps():
            self.origin_gps = self.latest_gps
            if self.home_alt is None and self.origin_gps is not None:
                self.home_alt = self.origin_gps[2]
        self.awaiting_reached = False
        self._send_next()

    def _reached_cb(self, msg: WaypointReached):
        if not self.awaiting_reached:
            return
        if self.expected_reached_seq is not None and not self._is_expected_seq(int(msg.wp_seq), int(self.expected_reached_seq)):
            rospy.loginfo_throttle(
                1.0,
                "Reached wp_seq=%d but waiting for wp_seq=%d; ignoring",
                int(msg.wp_seq),
                int(self.expected_reached_seq),
            )
            return
        if self.queue:
            self.queue.pop(0)
        self.awaiting_reached = False
        self.expected_reached_seq = None
        if not self.queue and self.pending_queue:
            self.queue = self.pending_queue
            self.pending_queue = None
            if self._needs_gps():
                self.origin_gps = self.pending_origin_gps or self.latest_gps
                self.pending_origin_gps = None
                if self.home_alt is None and self.origin_gps is not None:
                    self.home_alt = self.origin_gps[2]
        self._send_next()

    def _send_next(self):
        if self.awaiting_reached or not self.queue:
            return
        if self._needs_gps() and self.origin_gps is None:
            rospy.logwarn_throttle(2.0, "GPS origin missing; cannot send waypoint")
            return
        if not self._set_mode(self.guided_mode):
            rospy.logwarn("Failed to switch to %s before mission update", self.guided_mode)
        if not self._clear_mission():
            return
        wps = self._build_mission(self.queue[0])
        if self.log_mission_details:
            summary = ", ".join([f"(cmd={wp.command},frame={wp.frame},alt={wp.z_alt:.2f})" for wp in wps])
            rospy.loginfo("Pushing mission: %s", summary)
        if not self._push_waypoints(wps):
            return
        self._last_pushed_count = len(wps)
        self.awaiting_reached = True
        self.expected_reached_seq = 1 if len(wps) >= 2 else 0
        self._set_current_retry(0)
        if self.auto_mode_on_push:
            if self.last_state is not None and self.last_state.armed and self._can_start_auto():
                self._set_mode(self.auto_mode)
            else:
                self._auto_mode_pending = True

    def _set_current_retry(self, seq: int):
        for attempt in range(1, 6):
            self._set_current(seq)
            rospy.sleep(0.2)
            if self.expected_reached_seq is None:
                return
            # No direct feedback from set_current; rely on retries to overcome FCU busy windows.
            if attempt == 1:
                continue

    def _set_current(self, seq: int):
        try:
            resp = self.set_current_srv(wp_seq=seq)
            if not resp.success:
                rospy.logwarn("Set current wp rejected: %d", seq)
        except rospy.ServiceException as exc:
            rospy.logwarn("Set current wp failed: %s", exc)

    def _set_auto_mode(self):
        try:
            resp = self.set_mode_srv(base_mode=0, custom_mode=self.auto_mode)
            if not resp.mode_sent:
                rospy.logwarn("Set mode rejected: %s", self.auto_mode)
        except rospy.ServiceException as exc:
            rospy.logwarn("Set mode failed: %s", exc)

    def _set_mode(self, mode: str) -> bool:
        try:
            resp = self.set_mode_srv(base_mode=0, custom_mode=mode)
            if not resp.mode_sent:
                rospy.logwarn("Set mode rejected: %s", mode)
                return False
            return True
        except rospy.ServiceException as exc:
            rospy.logwarn("Set mode failed: %s", exc)
            return False

    def _clear_mission(self) -> bool:
        for attempt in range(1, self.clear_retries + 1):
            try:
                resp = self.clear_srv()
                if resp.success:
                    return True
                rospy.logwarn("Waypoint clear rejected (attempt %d)", attempt)
            except rospy.ServiceException as exc:
                rospy.logwarn("Waypoint clear failed (attempt %d): %s", attempt, exc)
            rospy.sleep(self.retry_wait)
        return False

    def _push_waypoints(self, waypoints: List[Waypoint]) -> bool:
        req = WaypointPushRequest(start_index=0, waypoints=waypoints)
        for attempt in range(1, self.push_retries + 1):
            try:
                resp = self.push_srv(req)
                if resp.success and resp.wp_transfered == len(waypoints):
                    return True
                rospy.logwarn("Waypoint push rejected (attempt %d)", attempt)
            except rospy.ServiceException as exc:
                rospy.logwarn("Waypoint push failed (attempt %d): %s", attempt, exc)
            rospy.sleep(self.retry_wait)
        return False

    def _build_waypoint(self, point: Point) -> Waypoint:
        frame = self._frame_id()
        x, y, z = self._convert_point(point, frame)
        wp = Waypoint()
        wp.frame = frame
        wp.command = self.command
        wp.is_current = True
        wp.autocontinue = True
        wp.param1 = 0.0
        wp.param2 = self.acceptance_radius
        wp.param3 = 0.0
        wp.param4 = 0.0
        wp.x_lat = x
        wp.y_long = y
        wp.z_alt = z
        return wp

    def _build_takeoff(self) -> Optional[Waypoint]:
        frame = self._frame_id()
        if frame not in (0, 3):
            return None
        if self.latest_gps is None:
            return None
        if not self.include_takeoff_item:
            return None
        if not self.always_send_takeoff:
            if self.home_alt is None:
                self.home_alt = self.latest_gps[2]
            rel_alt = float(self.latest_gps[2] - float(self.home_alt))
            if rel_alt >= (self.takeoff_alt - self.takeoff_margin):
                return None

        wp = Waypoint()
        wp.frame = frame
        wp.command = 22  # MAV_CMD_NAV_TAKEOFF
        wp.is_current = True
        wp.autocontinue = True
        wp.param1 = 0.0
        wp.param2 = 0.0
        wp.param3 = 0.0
        wp.param4 = 0.0
        wp.x_lat = float(self.latest_gps[0])
        wp.y_long = float(self.latest_gps[1])
        wp.z_alt = float(self.takeoff_alt) if frame == 3 else float(self.home_alt + self.takeoff_alt)
        return wp

    def _build_mission(self, first_point: Point) -> List[Waypoint]:
        wps: List[Waypoint] = []
        takeoff = self._build_takeoff()
        if takeoff is not None:
            wps.append(takeoff)
        nav = self._build_waypoint(first_point)
        if wps:
            nav.is_current = False
        else:
            nav.is_current = True
        wps.append(nav)
        return wps

    def _can_start_auto(self) -> bool:
        if self.last_state is None:
            return False
        if not self.last_state.armed:
            return False
        if self.frame_name != "global_rel_alt":
            return True
        if self.latest_gps is None:
            return False
        if self.home_alt is None:
            return False
        rel_alt = float(self.latest_gps[2] - float(self.home_alt))
        return rel_alt >= float(self.min_auto_alt)

    def _gps_cb(self, msg: Float32MultiArray):
        if len(msg.data) < 3:
            return
        self.latest_gps = (float(msg.data[0]), float(msg.data[1]), float(msg.data[2]))

    def _gps_fix_cb(self, msg: NavSatFix):
        self.latest_gps = (float(msg.latitude), float(msg.longitude), float(msg.altitude))

    def _is_expected_seq(self, wp_seq: int, expected: int) -> bool:
        if wp_seq == expected:
            return True
        # Some stacks report 1-based seq even for a single-item mission.
        if self._last_pushed_count == 1 and expected == 0 and wp_seq == 1:
            return True
        return False

    def _needs_gps(self) -> bool:
        return self.frame_name in ("global", "global_rel_alt")

    def _frame_id(self) -> int:
        frame_map = {
            "global": 0,
            "local_ned": 1,
            "global_rel_alt": 3,
            "local_enu": 4,
        }
        return frame_map.get(self.frame_name, 1)

    def _convert_point(self, point: Point, frame: int):
        if frame in (0, 3):
            if self.origin_gps is None:
                return 0.0, 0.0, 0.0
            lat, lon, alt = self._enu_offset_to_global(point, self.origin_gps)
            if frame == 3:
                if self.home_alt is None:
                    self.home_alt = self.origin_gps[2]
                home_alt = float(self.home_alt)
                if self.latest_gps is not None:
                    current_rel_alt = float(self.latest_gps[2]) - home_alt
                else:
                    current_rel_alt = float(self.origin_gps[2]) - home_alt
                alt = current_rel_alt + float(point.z)
                alt = max(alt, float(self.takeoff_alt))
            return lat, lon, alt
        if frame == 1:
            return float(point.y), float(point.x), float(-point.z)
        return float(point.x), float(point.y), float(point.z)

    @staticmethod
    def _enu_offset_to_global(point: Point, origin: Tuple[float, float, float]):
        lat0, lon0, alt0 = origin
        lat0_rad = math.radians(lat0)
        earth_radius = 6378137.0
        d_lat = (point.y / earth_radius) * (180.0 / math.pi)
        d_lon = (point.x / (earth_radius * math.cos(lat0_rad))) * (180.0 / math.pi)
        return lat0 + d_lat, lon0 + d_lon, alt0 + point.z


if __name__ == "__main__":
    try:
        node = WaypointManagerNode()
        rospy.loginfo("Waypoint manager node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
