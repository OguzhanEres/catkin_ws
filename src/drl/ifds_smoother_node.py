#!/usr/bin/env python3
"""
IFDS route smoothing node.
Applies incremental direction smoothing to a Path.
"""
from typing import List

import rospy
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path


class IFDSSmootherNode:
    def __init__(self):
        rospy.init_node("ifds_smoother", anonymous=False)
        self.alpha = float(rospy.get_param("~alpha", 0.35))
        self.route_in = rospy.get_param("~route_in", "/agent/route_raw")
        self.route_out = rospy.get_param("~route_out", "/agent/route_smoothed")
        self.sub = rospy.Subscriber(self.route_in, Path, self._route_cb, queue_size=1)
        self.pub = rospy.Publisher(self.route_out, Path, queue_size=1)

    def _route_cb(self, msg: Path):
        points = [pose.pose.position for pose in msg.poses]
        smoothed = self._ifds_smooth(points)

        out = Path()
        out.header = msg.header
        for i, pt in enumerate(smoothed):
            pose = PoseStamped()
            pose.header = out.header
            pose.pose.position = pt
            # Preserve orientation from input (use first pose's orientation for yaw control)
            if msg.poses:
                pose.pose.orientation = msg.poses[0].pose.orientation
            else:
                pose.pose.orientation.w = 1.0
            out.poses.append(pose)
        self.pub.publish(out)

    def _ifds_smooth(self, points: List[Point]) -> List[Point]:
        if len(points) < 2:
            return points

        smoothed = [Point(points[0].x, points[0].y, points[0].z)]
        prev_dx = points[1].x - points[0].x
        prev_dy = points[1].y - points[0].y
        prev_dz = points[1].z - points[0].z

        for i in range(1, len(points)):
            cur_dx = points[i].x - points[i - 1].x
            cur_dy = points[i].y - points[i - 1].y
            cur_dz = points[i].z - points[i - 1].z

            new_dx = prev_dx + self.alpha * (cur_dx - prev_dx)
            new_dy = prev_dy + self.alpha * (cur_dy - prev_dy)
            new_dz = prev_dz + self.alpha * (cur_dz - prev_dz)

            last = smoothed[-1]
            smoothed.append(Point(last.x + new_dx, last.y + new_dy, last.z + new_dz))

            prev_dx, prev_dy, prev_dz = new_dx, new_dy, new_dz

        return smoothed


if __name__ == "__main__":
    try:
        node = IFDSSmootherNode()
        rospy.loginfo("IFDS smoother node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
