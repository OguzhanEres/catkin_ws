#!/usr/bin/env python3
"""
Gazebo target marker updater.

Spawns a simple red cylinder model and keeps it at the latest setpoint
(/mavros/setpoint_position/local). Useful to see the active target in Gazebo.
"""
import os
from typing import Optional

import rospy
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SpawnModel, SpawnModelRequest, GetModelState


class TargetMarkerNode:
    def __init__(self):
        rospy.init_node("target_marker", anonymous=False)

        self.model_name = rospy.get_param("~model_name", "target_marker_red")
        self.model_path = rospy.get_param(
            "~model_path",
            os.path.expanduser("~/Desktop/catkin_ws/src/ardupilot_city_sim/models/target_marker_red/model.sdf"),
        )
        self.setpoint_topic = rospy.get_param("~setpoint_topic", "/mavros/setpoint_position/local")
        self.frame_id = rospy.get_param("~frame_id", "map")

        self._spawned = False
        self._last_pose: Optional[PoseStamped] = None

        self.state_pub = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=1)
        self.setpoint_sub = rospy.Subscriber(self.setpoint_topic, PoseStamped, self._setpoint_cb, queue_size=1)
        self._spawn_timer = rospy.Timer(rospy.Duration.from_sec(2.0), self._spawn_timer_cb, oneshot=False)

    def _spawn_timer_cb(self, _evt):
        if not self._spawned:
            self._spawn_model_if_needed()
    def _spawn_model_if_needed(self):
        try:
            rospy.wait_for_service("/gazebo/spawn_sdf_model", timeout=10.0)
        except rospy.ROSException:
            rospy.logwarn("spawn_sdf_model service unavailable; marker not spawned")
            return

        if not os.path.isfile(self.model_path):
            rospy.logwarn("Marker SDF not found: %s", self.model_path)
            return
        with open(self.model_path, "r", encoding="utf-8") as fh:
            sdf_xml = fh.read()
        try:
            spawn = rospy.ServiceProxy("/gazebo/spawn_sdf_model", SpawnModel)
            req = SpawnModelRequest()
            req.model_name = self.model_name
            req.model_xml = sdf_xml
            req.robot_namespace = ""
            req.reference_frame = self.frame_id
            req.initial_pose.position.z = 0.0
            resp = spawn(req)
            if resp.success or "already exists" in (resp.status_message or ""):
                rospy.loginfo("Target marker ready: %s", self.model_name)
                self._spawned = True
            else:
                rospy.logwarn("Spawn marker failed: %s", resp.status_message)
        except rospy.ServiceException as exc:
            rospy.logwarn("Spawn marker service failed: %s", exc)

    def _setpoint_cb(self, msg: PoseStamped):
        if not self._spawned:
            return
        self._last_pose = msg
        state = ModelState()
        state.model_name = self.model_name
        state.pose = msg.pose
        state.reference_frame = self.frame_id
        self.state_pub.publish(state)


if __name__ == "__main__":
    try:
        TargetMarkerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
