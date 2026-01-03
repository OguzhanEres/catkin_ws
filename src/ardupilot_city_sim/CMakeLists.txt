cmake_minimum_required(VERSION 3.0.2)
project(ardupilot_city_sim)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  gazebo_msgs
)

catkin_package()

catkin_install_python(PROGRAMS
  scripts/sim_control_gui.py
  scripts/dynamic_obstacle_controller.py
  scripts/training_trigger.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch worlds models
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
