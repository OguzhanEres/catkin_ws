#!/usr/bin/env python3
"""
IFDS route smoothing node (ROS 2).
Applies incremental direction smoothing to a Path.
"""

from typing import List

from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path

from drl_ros2 import rospy_compat as rospy


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
        for pt in smoothed:
            pose = PoseStamped()
            pose.header = out.header
            pose.pose.position = pt
            pose.pose.orientation.w = 1.0
            out.poses.append(pose)
        self.pub.publish(out)

    def _ifds_smooth(self, points: List[Point]) -> List[Point]:
        if len(points) < 2:
            return points

        smoothed = [Point(x=points[0].x, y=points[0].y, z=points[0].z)]
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
            smoothed.append(Point(x=last.x + new_dx, y=last.y + new_dy, z=last.z + new_dz))

            prev_dx, prev_dy, prev_dz = new_dx, new_dy, new_dz

        return smoothed


def main():
    try:
        _ = IFDSSmootherNode()
        rospy.loginfo("IFDS smoother node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.shutdown()


if __name__ == "__main__":
    main()

