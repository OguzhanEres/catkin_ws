#!/usr/bin/env python3
"""
PX4 OFFBOARD mode arm and takeoff script.
Streams setpoints, switches to OFFBOARD, and arms the drone.
"""
import rospy
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
import sys


class PX4ArmTakeoff:
    def __init__(self):
        rospy.init_node('px4_arm_takeoff', anonymous=True)

        self.takeoff_alt = rospy.get_param('~takeoff_alt', 2.0)
        self.prestream_duration = rospy.get_param('~prestream_duration', 3.0)
        self.arm_timeout = rospy.get_param('~arm_timeout', 30.0)
        self.setpoint_rate = rospy.get_param('~setpoint_rate', 20.0)

        self.current_state = None
        self.local_pos = None

        # Publishers
        self.setpoint_pub = rospy.Publisher(
            '/mavros/setpoint_position/local',
            PoseStamped,
            queue_size=10
        )

        # Subscribers
        self.state_sub = rospy.Subscriber(
            '/mavros/state',
            State,
            self.state_cb
        )
        self.pose_sub = rospy.Subscriber(
            '/mavros/local_position/pose',
            PoseStamped,
            self.pose_cb
        )

        # Wait for services
        rospy.loginfo("Waiting for MAVROS services...")
        rospy.wait_for_service('/mavros/cmd/arming', timeout=30)
        rospy.wait_for_service('/mavros/set_mode', timeout=30)

        self.arming_client = rospy.ServiceProxy('/mavros/cmd/arming', CommandBool)
        self.set_mode_client = rospy.ServiceProxy('/mavros/set_mode', SetMode)

        rospy.loginfo("PX4 Arm/Takeoff node initialized")

    def state_cb(self, msg):
        self.current_state = msg

    def pose_cb(self, msg):
        self.local_pos = msg

    def wait_for_connection(self, timeout=30.0):
        """Wait for MAVROS connection to FCU"""
        rospy.loginfo("Waiting for FCU connection...")
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.current_state is not None and self.current_state.connected:
                rospy.loginfo("FCU connected!")
                return True
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logerr("FCU connection timeout!")
                return False
            rate.sleep()
        return False

    def publish_setpoint(self, x=0, y=0, z=2.0):
        """Publish a single setpoint"""
        msg = PoseStamped()
        msg.header.stamp = rospy.Time.now()
        msg.header.frame_id = 'map'
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = z
        msg.pose.orientation.w = 1.0
        self.setpoint_pub.publish(msg)

    def prestream_setpoints(self):
        """Stream setpoints before switching to OFFBOARD"""
        rospy.loginfo(f"Pre-streaming setpoints for {self.prestream_duration}s...")
        rate = rospy.Rate(self.setpoint_rate)
        start = rospy.Time.now()

        while not rospy.is_shutdown():
            self.publish_setpoint(z=self.takeoff_alt)
            if (rospy.Time.now() - start).to_sec() > self.prestream_duration:
                break
            rate.sleep()

        rospy.loginfo("Pre-streaming complete")

    def set_offboard_mode(self):
        """Switch to OFFBOARD mode"""
        rospy.loginfo("Switching to OFFBOARD mode...")
        try:
            resp = self.set_mode_client(base_mode=0, custom_mode='OFFBOARD')
            if resp.mode_sent:
                rospy.loginfo("OFFBOARD mode set!")
                return True
            else:
                rospy.logwarn("Failed to set OFFBOARD mode")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Set mode service failed: {e}")
            return False

    def arm(self):
        """Arm the drone"""
        rospy.loginfo("Arming drone...")
        try:
            resp = self.arming_client(value=True)
            if resp.success:
                rospy.loginfo("Drone armed!")
                return True
            else:
                rospy.logwarn("Arming rejected")
                return False
        except rospy.ServiceException as e:
            rospy.logerr(f"Arming service failed: {e}")
            return False

    def wait_for_arm(self, timeout=10.0):
        """Wait until drone is armed"""
        rate = rospy.Rate(10)
        start = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.current_state is not None and self.current_state.armed:
                return True
            if (rospy.Time.now() - start).to_sec() > timeout:
                return False
            rate.sleep()
        return False

    def run(self):
        """Main execution"""
        # Wait for connection
        if not self.wait_for_connection():
            return False

        # Pre-stream setpoints (required for OFFBOARD)
        self.prestream_setpoints()

        # Switch to OFFBOARD mode
        rate = rospy.Rate(self.setpoint_rate)

        # Keep publishing setpoints while trying to arm
        start = rospy.Time.now()
        armed = False
        offboard_set = False

        while not rospy.is_shutdown() and (rospy.Time.now() - start).to_sec() < self.arm_timeout:
            # Always publish setpoints
            self.publish_setpoint(z=self.takeoff_alt)

            if self.current_state is None:
                rate.sleep()
                continue

            # Set OFFBOARD mode if not set
            if not offboard_set and self.current_state.mode != "OFFBOARD":
                if self.set_offboard_mode():
                    offboard_set = True
                    rospy.sleep(0.5)

            # Arm if not armed and in OFFBOARD mode
            if not armed and self.current_state.mode == "OFFBOARD" and not self.current_state.armed:
                if self.arm():
                    armed = True

            # Check if we're armed and in OFFBOARD
            if self.current_state.armed and self.current_state.mode == "OFFBOARD":
                rospy.loginfo("SUCCESS: Armed and in OFFBOARD mode!")
                # Continue publishing setpoints for a bit
                for _ in range(int(self.setpoint_rate * 2)):
                    self.publish_setpoint(z=self.takeoff_alt)
                    rate.sleep()
                return True

            rate.sleep()

        rospy.logerr("Failed to arm and enter OFFBOARD mode")
        return False


def main():
    try:
        node = PX4ArmTakeoff()
        success = node.run()
        if success:
            rospy.loginfo("PX4 arm/takeoff complete, exiting")
        else:
            rospy.logerr("PX4 arm/takeoff failed")
            sys.exit(1)
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
