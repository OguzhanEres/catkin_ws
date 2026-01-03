#!/usr/bin/env python3
"""
PPO+LSTM trainer node (ROS 2) - placeholder.

The ROS1 trainer (`src/drl/ppo_lstm_trainer_node.py`) is tightly coupled to
rospy + MAVROS + Gazebo classic services. A full ROS 2 port needs:
- rclpy conversion (node/pubs/subs/services/timers)
- Gazebo service/topic API adaptation (ModelState vs EntityState)
- MAVROS2 service/topic verification for your ROS 2 distro

This file exists so the ROS 2 package structure is complete and can be built.
"""


def main() -> None:
    raise SystemExit(
        "drl_ros2: PPO+LSTM trainer is not ported to ROS 2 yet. "
        "Use the ROS1 workspace for training, or ask me to port the trainer next."
    )


if __name__ == "__main__":
    main()

