#!/usr/bin/env python3
"""
Safety filter node to supervise DRL commands using geometric avoidance (APF/IFDS style).
- If obstacle within threshold, override with escape command.
- Otherwise, pass DRL command through.
"""
from typing import Optional

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


def replace_inf_with_max(ranges: np.ndarray, max_range: float) -> np.ndarray:
    clean = np.copy(ranges)
    clean[np.isinf(clean)] = max_range
    return clean


def find_escape_command(scan: LaserScan, avoid_dist: float) -> Twist:
    ranges = np.array(scan.ranges, dtype=np.float32)
    ranges = replace_inf_with_max(ranges, scan.range_max)
    min_idx = int(np.argmin(ranges))
    min_val = float(ranges[min_idx])
    angle = scan.angle_min + min_idx * scan.angle_increment

    cmd = Twist()
    if min_val < avoid_dist:
        # TODO: Replace with full APF/IFDS; this is a simple heuristic for Gazebo tests.
        turn_dir = -1.0 if angle > 0 else 1.0
        cmd.linear.x = max(0.0, min_val - 0.1) * 0.2
        cmd.angular.z = turn_dir * 0.8
    else:
        cmd.linear.x = 0.0
        cmd.angular.z = 0.0
    return cmd


class SafetyFilterNode:
    def __init__(self):
        rospy.init_node("safety_filter", anonymous=False)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self._scan_cb, queue_size=1)
        self.cmd_sub = rospy.Subscriber("/cmd_vel_raw", Twist, self._cmd_cb, queue_size=1)
        self.safe_pub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel_unstamped", Twist, queue_size=1)

        self.avoid_dist = rospy.get_param("~avoid_distance", 0.5)

        self.latest_scan: Optional[LaserScan] = None
        self.latest_cmd: Optional[Twist] = None

    def _scan_cb(self, msg: LaserScan):
        self.latest_scan = msg

    def _cmd_cb(self, msg: Twist):
        self.latest_cmd = msg
        self._filter_and_publish()

    def _filter_and_publish(self):
        if self.latest_cmd is None:
            return
        if self.latest_scan is None:
            self.safe_pub.publish(self.latest_cmd)
            return

        ranges = np.array(self.latest_scan.ranges, dtype=np.float32)
        ranges = replace_inf_with_max(ranges, self.latest_scan.range_max)
        min_range = float(np.min(ranges))

        if min_range < self.avoid_dist:
            safe_cmd = find_escape_command(self.latest_scan, self.avoid_dist)
            rospy.logwarn_throttle(2.0, f"Override DRL cmd; obstacle at {min_range:.2f} m")
            self.safe_pub.publish(safe_cmd)
        else:
            self.safe_pub.publish(self.latest_cmd)


if __name__ == "__main__":
    try:
        node = SafetyFilterNode()
        rospy.loginfo("Safety filter node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
