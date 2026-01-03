#!/usr/bin/env python3
"""
Mission manager to set GUIDED, arm, and climb to a target altitude via MavROS.
"""
import rospy
from mavros_msgs.srv import CommandBool, CommandBoolRequest
from mavros_msgs.srv import SetMode, SetModeRequest
from mavros_msgs.srv import CommandTOL, CommandTOLRequest


class MissionManager:
    def __init__(self):
        rospy.init_node("mission_manager", anonymous=False)
        self.takeoff_alt = rospy.get_param("~takeoff_alt", 2.0)

        rospy.wait_for_service("/mavros/set_mode")
        rospy.wait_for_service("/mavros/cmd/arming")
        rospy.wait_for_service("/mavros/cmd/takeoff")

        self.set_mode_srv = rospy.ServiceProxy("/mavros/set_mode", SetMode)
        self.arm_srv = rospy.ServiceProxy("/mavros/cmd/arming", CommandBool)
        self.takeoff_srv = rospy.ServiceProxy("/mavros/cmd/takeoff", CommandTOL)

    def run(self):
        if not self._set_guided():
            return
        if not self._arm():
            return
        self._takeoff()

    def _set_guided(self) -> bool:
        req = SetModeRequest()
        req.custom_mode = "GUIDED"
        resp = self.set_mode_srv(req)
        if not resp.mode_sent:
            rospy.logerr("Failed to set GUIDED mode")
            return False
        rospy.loginfo("GUIDED mode set")
        return True

    def _arm(self) -> bool:
        req = CommandBoolRequest()
        req.value = True
        resp = self.arm_srv(req)
        if not resp.success:
            rospy.logerr("Arming failed")
            return False
        rospy.loginfo("Vehicle armed")
        return True

    def _takeoff(self):
        req = CommandTOLRequest()
        req.altitude = self.takeoff_alt
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = 0.0
        resp = self.takeoff_srv(req)
        if not resp.success:
            rospy.logerr("Takeoff command failed")
            return
        rospy.loginfo(f"Takeoff to {self.takeoff_alt:.2f} m commanded")


def main():
    mgr = MissionManager()
    mgr.run()
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
