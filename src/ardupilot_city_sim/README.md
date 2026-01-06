# PX4 City Simulation (Gazebo 11 + ROS Noetic)

This package prepares a Gazebo 11 city-style simulation using the official
PX4 Iris quadcopter model, with a 180-degree 2D LiDAR and a forward camera.

This repo also contains a `drl` package that can run PPO+LSTM training using a
setpoint-based control loop (with IFDS smoothing). For a full usage guide see:
`/home/oguz/Desktop/catkin_ws/src/README.md`

## Requirements
- ROS Noetic + Gazebo 11
- PX4 SITL in `~/PX4-Autopilot`
- PX4 Gazebo plugins built in `~/PX4-Autopilot/build/px4_sitl_default/build_gazebo`

## Quick Start
1) Source ROS and the workspace:
```
source /opt/ros/noetic/setup.bash
source ~/Desktop/catkin_ws/devel/setup.bash
```
2) Launch Gazebo with the city world:
```
roslaunch ardupilot_city_sim city_sim_gazebo.launch
```
3) Start PX4 SITL (SITL only, Gazebo already running):
```
cd ~/PX4-Autopilot
export PX4_SIM_MODEL=iris
export PX4_SIM_HOST_ADDR=127.0.0.1
./build/px4_sitl_default/bin/px4 ./build/px4_sitl_default/etc -s etc/init.d-posix/rcS -t ./test_data
```
4) Connect MAVROS (optional, for external software):
```
roslaunch ardupilot_city_sim mavros_connect.launch fcu_url:=udp://:14540@127.0.0.1:14580
```
5) Start the GUI:
```
rosrun ardupilot_city_sim sim_control_gui.py
```
6) (Optional) Start the DRL infra stack (inference only):
```
roslaunch drl drl_infra.launch
```
7) (Optional) Start PPO+LSTM training:
```
roslaunch drl drl_train.launch mode:=exploration epochs:=10 step_size:=1.0 takeoff_alt:=1.0 min_alt_for_control:=0.0
```

## Full Stack (Gazebo + MAVROS + DRL Infra)
```
roslaunch ardupilot_city_sim full_stack.launch
```
Note: PX4 SITL is started separately (GUI or terminal).

## Training Notes
- Target marker is placed at (0, 18, 1) in the Gazebo world.
- GUI training buttons launch `drl_train.launch` with the entered epoch count.

## ROS Topics
- LiDAR (180 deg): `/scan`
- Camera: `/front_camera/image_raw`, `/front_camera/camera_info`
- Agent mode trigger: `/agent/mode` (std_msgs/String)

## Notes
- Dynamic obstacles move linearly and deterministically.
- The training trigger node only publishes a mode string; it does not train.
