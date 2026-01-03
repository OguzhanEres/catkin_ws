from setuptools import find_packages, setup

package_name = "ardupilot_city_sim_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/city_sim_gazebo.launch.py"]),
        ("share/" + package_name + "/config", ["config/ardupilot_override.parm"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="oguz",
    maintainer_email="oguz@example.com",
    description="ROS 2 scaffolding for ArduPilot + Gazebo city simulation environment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "training_trigger = ardupilot_city_sim_ros2.training_trigger:main",
            "dynamic_obstacle_controller = ardupilot_city_sim_ros2.dynamic_obstacle_controller:main",
            "sim_control_gui = ardupilot_city_sim_ros2.sim_control_gui:main",
        ],
    },
)

