# ArduPilot City Simulation (Gazebo 11 + ROS Noetic)

This package prepares a Gazebo 11 city-style simulation using the official
ArduPilot Iris quadcopter model, with a 180-degree 2D LiDAR and a forward camera.

This repo also contains a `drl` package that can run PPO+LSTM training using a
setpoint-based control loop (with IFDS smoothing). For a full usage guide see:
`/home/oguz/catkin_ws/src/README.md`

## Requirements
- ROS Noetic + Gazebo 11
- ArduPilot SITL in `~/ardupilot`
- `ardupilot_gazebo` built in `~/ardupilot_gazebo` (libArduPilotPlugin)

## Quick Start
1) Source ROS and the workspace:
```
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
```
2) Launch Gazebo with the city world:
```
roslaunch ardupilot_city_sim city_sim_gazebo.launch
```
3) Start ArduPilot SITL (official Gazebo Iris model):
```
cd ~/ardupilot/ArduCopter
../Tools/autotest/sim_vehicle.py -v ArduCopter -f gazebo-iris -I0 --out=127.0.0.1:14550
```
4) Connect MAVROS (optional, for external software):
```
roslaunch ardupilot_city_sim mavros_connect.launch
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
roslaunch drl drl_train.launch mode:=exploration epochs:=10
```

## Full Stack (Gazebo + MAVROS + DRL Infra)
```
roslaunch ardupilot_city_sim full_stack.launch
```
Note: ArduPilot SITL is started separately (GUI or terminal).

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
