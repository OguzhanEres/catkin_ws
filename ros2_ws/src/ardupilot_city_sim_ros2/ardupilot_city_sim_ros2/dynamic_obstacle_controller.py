#!/usr/bin/env python3

from drl_ros2 import rospy_compat as rospy

try:
    from gazebo_msgs.srv import SetEntityState
    from gazebo_msgs.msg import EntityState

    _SET_SRV = ("/gazebo/set_entity_state", SetEntityState, "state", EntityState)
except Exception:  # pragma: no cover
    _SET_SRV = None

try:
    from gazebo_msgs.srv import SetModelState
    from gazebo_msgs.msg import ModelState

    _SET_SRV_FALLBACK = ("/gazebo/set_model_state", SetModelState, "model_state", ModelState)
except Exception:  # pragma: no cover
    _SET_SRV_FALLBACK = None


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

    srv_spec = _SET_SRV or _SET_SRV_FALLBACK
    if srv_spec is None:
        rospy.logwarn("Gazebo set-state service type not available; exiting")
        rospy.shutdown()
        return

    srv_name, srv_type, field, msg_type = srv_spec
    set_state = rospy.ServiceProxy(srv_name, srv_type)

    rate = rospy.Rate(20)
    last_time = rospy.Time.now()

    while not rospy.is_shutdown():
        now = rospy.Time.now()
        dt = (now - last_time).to_sec()
        last_time = now

        for obs in obstacles:
            obs.step(dt)
            x, y, z = obs.pose_xyz()
            state = msg_type()
            if field == "state":
                state.name = obs.name
            else:
                state.model_name = obs.name
            state.pose.position.x = float(x)
            state.pose.position.y = float(y)
            state.pose.position.z = float(z)
            state.pose.orientation.w = 1.0
            try:
                set_state(**{field: state})
            except rospy.ServiceException as exc:
                rospy.logwarn_throttle(2.0, "Set state failed: %s", exc)

        rate.sleep()

    rospy.shutdown()


if __name__ == "__main__":
    main()

