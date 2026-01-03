cmake_minimum_required(VERSION 3.0.2)
project(drl)

find_package(catkin REQUIRED COMPONENTS
  rospy
  roscpp
  std_msgs
  sensor_msgs
  geometry_msgs
  nav_msgs
  tf2
  tf2_ros
  urdf
  xacro
  gazebo_ros
  gazebo_msgs
  cv_bridge
  image_transport
  mavros
  mavros_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    roscpp
    std_msgs
    sensor_msgs
    geometry_msgs
    nav_msgs
    tf2
    tf2_ros
    urdf
    xacro
    gazebo_ros
    gazebo_msgs
    cv_bridge
    image_transport
    mavros
    mavros_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# Install Python scripts
catkin_install_python(PROGRAMS
  drl_agent_node.py
  sensor_vectorizer_node.py
  ppo_lstm_agent_node.py
  ppo_lstm_trainer_node.py
  ifds_smoother_node.py
  waypoint_manager_node.py
  setpoint_follower_node.py
  mission_manager.py
  tracker_node.py
  safety_filter_node.py
  mission_control_dashboard.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install URDF files
install(DIRECTORY urdf/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/urdf
)

# Install world files
install(DIRECTORY worlds/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/worlds
)

# Install model files
install(DIRECTORY models/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/models
)
