from setuptools import find_packages, setup

package_name = "drl_ros2"

setup(
    name=package_name,
    version="1.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", ["launch/drl_train.launch.py"]),
        ("share/" + package_name + "/models", ["models/.gitkeep"]),
        ("share/" + package_name + "/worlds", ["worlds/.gitkeep"]),
        ("share/" + package_name + "/urdf", ["urdf/.gitkeep"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Oguz",
    maintainer_email="oguz@example.com",
    description="ROS 2 port of the DRL UAV navigation package.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "ifds_smoother = drl_ros2.ifds_smoother_node:main",
            "sensor_vectorizer = drl_ros2.sensor_vectorizer_node:main",
            "setpoint_follower = drl_ros2.setpoint_follower_node:main",
            "ppo_lstm_trainer = drl_ros2.ppo_lstm_trainer_node:main",
        ],
    },
)

