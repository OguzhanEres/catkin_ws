from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    mode = LaunchConfiguration("mode")
    epochs = LaunchConfiguration("epochs")
    step_size = LaunchConfiguration("step_size")
    takeoff_alt = LaunchConfiguration("takeoff_alt")

    return LaunchDescription(
        [
            DeclareLaunchArgument("mode", default_value="exploration"),
            DeclareLaunchArgument("epochs", default_value="10"),
            DeclareLaunchArgument("step_size", default_value="6.0"),
            DeclareLaunchArgument("takeoff_alt", default_value="2.0"),
            Node(
                package="drl_ros2",
                executable="sensor_vectorizer",
                name="sensor_vectorizer",
                output="screen",
                parameters=[
                    {
                        "lidar_topic": "/scan",
                        "camera_topic": "/front_camera/image_raw",
                        "imu_topic": "/mavros/imu/data",
                        "gps_topic": "/mavros/global_position/global",
                        "lidar_bins": 180,
                        "lidar_fov_deg": 180.0,
                        "lidar_front_center_deg": 0.0,
                        "camera_width": 32,
                        "camera_height": 24,
                    }
                ],
            ),
            Node(
                package="drl_ros2",
                executable="ifds_smoother",
                name="ifds_smoother",
                output="screen",
                parameters=[{"alpha": 0.35, "route_in": "/agent/route_raw", "route_out": "/agent/route_smoothed"}],
            ),
            Node(
                package="drl_ros2",
                executable="setpoint_follower",
                name="setpoint_follower",
                output="screen",
                parameters=[
                    {
                        "route_topic": "/agent/route_smoothed",
                        "acceptance_radius": 0.5,
                        "pose_topic": "/mavros/local_position/pose",
                        "setpoint_topic": "/mavros/setpoint_position/local",
                        "reached_topic": "/agent/wp_reached",
                        "publish_rate": 20.0,
                        "frame_id": "map",
                        "takeoff_alt": takeoff_alt,
                        "max_target_z": takeoff_alt,
                        "max_xy_step": 3.0,
                        "min_alt_for_control": 1.7,
                        "require_armed": True,
                        "require_guided": True,
                    }
                ],
            ),
        ]
    )
