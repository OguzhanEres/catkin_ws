#!/usr/bin/env python3
"""
Bridge Gazebo model pose into MAVROS vision topics so PX4 EKF gains a stable
heading/position source (LOCAL_POSITION_NED becomes valid before OFFBOARD).
"""
from typing import Optional

import rospy
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import PoseStamped, TwistStamped
from nav_msgs.msg import Odometry


class GazeboVisionBridge:
    def __init__(self):
        rospy.init_node("gazebo_vision_bridge", anonymous=False)

        self.model_name = rospy.get_param("~model_name", "iris_px4_sensors")
        self.frame_id = rospy.get_param("~frame_id", "map")
        self.pose_topic = rospy.get_param("~pose_topic", "/mavros/vision_pose/pose")
        self.twist_topic = rospy.get_param("~twist_topic", "/mavros/vision_speed/speed_twist")
        self.publish_twist = bool(rospy.get_param("~publish_twist", True))
        self.publish_odom = bool(rospy.get_param("~publish_odom", True))
        self.odom_topic = rospy.get_param("~odom_topic", "/mavros/odometry/in")

        self._gazebo_index: Optional[int] = None
        self._logged_ready = False

        self.pose_pub = rospy.Publisher(self.pose_topic, PoseStamped, queue_size=1)
        self.twist_pub = rospy.Publisher(self.twist_topic, TwistStamped, queue_size=1) if self.publish_twist else None
        self.odom_pub = rospy.Publisher(self.odom_topic, Odometry, queue_size=1) if self.publish_odom else None

        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._states_cb, queue_size=1)

    def _states_cb(self, msg: ModelStates):
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
                rospy.logwarn_throttle(5.0, "Model %s not found in gazebo/model_states", self.model_name)
                return
            self._gazebo_index = idx

        now = rospy.Time.now()
        if hasattr(msg, "header") and getattr(msg, "header", None) is not None:
            stamp = getattr(msg.header, "stamp", rospy.Time(0))
            if stamp and stamp != rospy.Time(0):
                now = stamp
        pose = msg.pose[idx]
        twist = msg.twist[idx]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = now
        pose_msg.header.frame_id = self.frame_id
        pose_msg.pose = pose
        self.pose_pub.publish(pose_msg)

        if self.twist_pub is not None:
            twist_msg = TwistStamped()
            twist_msg.header.stamp = now
            twist_msg.header.frame_id = self.frame_id
            twist_msg.twist = twist
            self.twist_pub.publish(twist_msg)

        if self.odom_pub is not None:
            odom_msg = Odometry()
            odom_msg.header.stamp = now
            odom_msg.header.frame_id = self.frame_id
            odom_msg.child_frame_id = self.model_name
            odom_msg.pose.pose = pose
            odom_msg.twist.twist = twist
            self.odom_pub.publish(odom_msg)

        if not self._logged_ready:
            self._logged_ready = True
            rospy.loginfo(
                "Publishing Gazebo pose as vision: model=%s -> %s (twist:%s odom:%s)",
                self.model_name,
                self.pose_topic,
                "on" if self.publish_twist else "off",
                "on" if self.publish_odom else "off",
            )


if __name__ == "__main__":
    try:
        GazeboVisionBridge()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
