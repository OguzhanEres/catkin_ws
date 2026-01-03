from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    gui = LaunchConfiguration("gui")
    use_sim_time = LaunchConfiguration("use_sim_time")
    pkg_share = FindPackageShare("ardupilot_city_sim_ros2")
    world_file = PathJoinSubstitution([pkg_share, "worlds", "city_sim.world"])

    model_path = [PathJoinSubstitution([pkg_share, "models"]), TextSubstitution(text=":"), EnvironmentVariable("GAZEBO_MODEL_PATH")]
    resource_path = [
        PathJoinSubstitution([pkg_share, "worlds"]),
        TextSubstitution(text=":"),
        EnvironmentVariable("GAZEBO_RESOURCE_PATH"),
    ]

    gzserver = ExecuteProcess(
        cmd=["gzserver", "--verbose", "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so", world_file],
        output="screen",
        condition=UnlessCondition(gui),
    )
    gazebo = ExecuteProcess(
        cmd=["gazebo", "--verbose", "-s", "libgazebo_ros_init.so", "-s", "libgazebo_ros_factory.so", world_file],
        output="screen",
        condition=IfCondition(gui),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("gui", default_value="true"),
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            SetEnvironmentVariable(name="GAZEBO_MODEL_PATH", value=model_path),
            SetEnvironmentVariable(name="GAZEBO_RESOURCE_PATH", value=resource_path),
            gzserver,
            gazebo,
            Node(
                package="ardupilot_city_sim_ros2",
                executable="dynamic_obstacle_controller",
                name="dynamic_obstacle_controller",
                output="screen",
                parameters=[{"use_sim_time": use_sim_time}],
            ),
        ]
    )

