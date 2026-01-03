# ROS 2 Workspace (colcon) - `ros2_ws`

This repository currently contains a ROS 1 (Noetic) `catkin_ws`.
To make it usable with ROS 2, a parallel ROS 2 workspace is provided here:

- `ros2_ws/src/drl_ros2`
- `ros2_ws/src/ardupilot_city_sim_ros2`

## Build

```bash
source /opt/ros/<distro>/setup.bash
cd ~/catkin_ws/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Status

- `drl_ros2`: `sensor_vectorizer`, `ifds_smoother`, `setpoint_follower` are ported.
- `drl_ros2`: `ppo_lstm_trainer` is a placeholder (full port still needed).
- `ardupilot_city_sim_ros2`: Gazebo launch + dynamic obstacle controller scaffolding is included.

## Run (ROS 2)

### Gazebo city sim

```bash
source /opt/ros/<distro>/setup.bash
source ~/catkin_ws/ros2_ws/install/setup.bash
ros2 launch ardupilot_city_sim_ros2 city_sim_gazebo.launch.py gui:=true
```

### DRL helper nodes (no trainer)

```bash
source /opt/ros/<distro>/setup.bash
source ~/catkin_ws/ros2_ws/install/setup.bash
ros2 launch drl_ros2 drl_train.launch.py
```

Note: `mode/epochs/step_size/takeoff_alt` arguments are kept for parity with the ROS1 launch,
but only `takeoff_alt` affects the currently-portable nodes.

### Training (ROS 1 / Noetic)

Training is still run from the ROS 1 workspace as documented in `src/README.md`.
