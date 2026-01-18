#!/usr/bin/env python3
"""
Converts PoseStamped messages to TF transforms for RViz visualization.
"""
import rospy
import tf2_ros
from geometry_msgs.msg import PoseStamped, TransformStamped


class PoseToTF:
    def __init__(self):
        rospy.init_node('pose_to_tf', anonymous=True)

        self.parent_frame = rospy.get_param("~parent_frame", "map")
        self.child_frame = rospy.get_param("~child_frame", "base_link")
        pose_topic = rospy.get_param("~pose_topic", "/mavros/local_position/pose")

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()
        rospy.Subscriber(pose_topic, PoseStamped, self._pose_cb)

        rospy.loginfo("PoseToTF: %s -> %s from %s", self.parent_frame, self.child_frame, pose_topic)

    def _pose_cb(self, msg: PoseStamped):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp if msg.header.stamp.secs > 0 else rospy.Time.now()
        t.header.frame_id = self.parent_frame
        t.child_frame_id = self.child_frame
        t.transform.translation.x = msg.pose.position.x
        t.transform.translation.y = msg.pose.position.y
        t.transform.translation.z = msg.pose.position.z
        t.transform.rotation = msg.pose.orientation
        self.tf_broadcaster.sendTransform(t)


if __name__ == '__main__':
    try:
        node = PoseToTF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
