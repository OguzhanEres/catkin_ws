#!/usr/bin/env python3

import argparse

from std_msgs.msg import String

from drl_ros2 import rospy_compat as rospy


def main():
    parser = argparse.ArgumentParser(description="Publish a training trigger mode.")
    parser.add_argument("--mode", choices=["exploration", "tracking"], required=True)
    args = parser.parse_args()

    rospy.init_node("training_trigger", anonymous=True)
    pub = rospy.Publisher("/agent/mode", String, queue_size=1, latch=True)
    msg = String(data=args.mode)

    for _ in range(3):
        pub.publish(msg)
        rospy.sleep(0.1)

    rospy.loginfo("Agent mode published: %s", args.mode)
    rospy.shutdown()


if __name__ == "__main__":
    main()

