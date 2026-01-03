#!/usr/bin/env python3
import math
import rospy
from gazebo_msgs.srv import SetModelState
from gazebo_msgs.msg import ModelState


class LinearObstacle:
    def __init__(self, name, axis, origin, amplitude, speed):
        self.name = name
        self.axis = axis
        self.origin = origin
        self.amplitude = amplitude
        self.speed = speed
        self.offset = 0.0
        self.direction = 1.0

    def step(self, dt):
        if dt <= 0.0:
            return
        self.offset += self.direction * self.speed * dt
        if self.offset > self.amplitude:
            self.offset = self.amplitude
            self.direction = -1.0
        elif self.offset < -self.amplitude:
            self.offset = -self.amplitude
            self.direction = 1.0

    def pose_xyz(self):
        x, y, z = self.origin
        if self.axis == "x":
            x += self.offset
        elif self.axis == "z":
            z += self.offset
        return x, y, z


def main():
    rospy.init_node("dynamic_obstacle_controller")

    obstacles = [
        LinearObstacle("dyn_box_1", "x", (0.0, -5.0, 1.0), 3.0, 0.6),
        LinearObstacle("dyn_box_2", "z", (12.0, 0.0, 1.5), 1.0, 0.4),
        LinearObstacle("dyn_box_3", "x", (-12.0, 10.0, 1.0), 2.5, 0.5),
        LinearObstacle("dyn_box_4", "z", (8.0, 15.0, 1.5), 1.0, 0.4),
    ]

    rospy.wait_for_service("/gazebo/set_model_state")
    set_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)

    rate = rospy.Rate(20)
    last_time = rospy.Time.now()

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        dt = (now - last_time).to_sec()
        last_time = now

        for obs in obstacles:
            obs.step(dt)
            x, y, z = obs.pose_xyz()
            state = ModelState()
            state.model_name = obs.name
            state.pose.position.x = x
            state.pose.position.y = y
            state.pose.position.z = z
            state.pose.orientation.w = 1.0
            try:
                set_state(state)
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(2.0, "SetModelState failed: %s", exc)

        rate.sleep()


if __name__ == "__main__":
    main()
