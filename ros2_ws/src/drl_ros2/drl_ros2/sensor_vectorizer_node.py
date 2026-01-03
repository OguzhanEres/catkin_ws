#!/usr/bin/env python3
"""
Sensor vectorizer node (ROS 2).
Converts LiDAR, camera, IMU, gyro, and GPS inputs into fixed-size vectors.
"""

import cv2
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, Imu, LaserScan, NavSatFix
from std_msgs.msg import Float32MultiArray

from drl_ros2 import rospy_compat as rospy


class SensorVectorizerNode:
    def __init__(self):
        rospy.init_node("sensor_vectorizer", anonymous=False)

        self.lidar_topic = rospy.get_param("~lidar_topic", "/scan")
        self.camera_topic = rospy.get_param("~camera_topic", "/front_camera/image_raw")
        self.imu_topic = rospy.get_param("~imu_topic", "/mavros/imu/data")
        self.gps_topic = rospy.get_param("~gps_topic", "/mavros/global_position/global")

        self.lidar_bins = int(rospy.get_param("~lidar_bins", 180))
        self.lidar_fov_deg = float(rospy.get_param("~lidar_fov_deg", 180.0))
        self.lidar_front_center_deg = float(rospy.get_param("~lidar_front_center_deg", 0.0))
        self.cam_w = int(rospy.get_param("~camera_width", 32))
        self.cam_h = int(rospy.get_param("~camera_height", 24))

        self.bridge = CvBridge()

        self.lidar_pub = rospy.Publisher("/agent/lidar_vec", Float32MultiArray, queue_size=1)
        self.camera_pub = rospy.Publisher("/agent/camera_vec", Float32MultiArray, queue_size=1)
        self.imu_pub = rospy.Publisher("/agent/imu_vec", Float32MultiArray, queue_size=1)
        self.gyro_pub = rospy.Publisher("/agent/gyro_vec", Float32MultiArray, queue_size=1)
        self.gps_pub = rospy.Publisher("/agent/gps_vec", Float32MultiArray, queue_size=1)

        self.scan_sub = rospy.Subscriber(self.lidar_topic, LaserScan, self._lidar_cb, queue_size=1)
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self._camera_cb, queue_size=1)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=1)
        self.gps_sub = rospy.Subscriber(self.gps_topic, NavSatFix, self._gps_cb, queue_size=1)

    def _lidar_cb(self, msg: LaserScan):
        ranges = np.array(msg.ranges, dtype=np.float32)
        ranges = np.nan_to_num(ranges, nan=msg.range_max, posinf=msg.range_max, neginf=0.0)
        if ranges.size == 0:
            return
        if msg.angle_increment != 0.0 and self.lidar_fov_deg > 0.0:
            angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
            half_fov = np.deg2rad(self.lidar_fov_deg) * 0.5
            center = np.deg2rad(self.lidar_front_center_deg)
            mask = (angles >= center - half_fov) & (angles <= center + half_fov)
            if mask.any():
                ranges = ranges[mask]
        idx = np.linspace(0, ranges.size - 1, self.lidar_bins).astype(np.int32)
        sliced = ranges[idx]
        denom = max(1e-6, float(msg.range_max))
        vec = (sliced / denom).astype(np.float32)
        self.lidar_pub.publish(Float32MultiArray(data=vec.tolist()))

    def _camera_cb(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as exc:
            rospy.logwarn_throttle(2.0, "Camera conversion failed: %s", exc)
            return
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (self.cam_w, self.cam_h), interpolation=cv2.INTER_AREA)
        vec = (resized.astype(np.float32).reshape(-1) / 255.0).astype(np.float32)
        self.camera_pub.publish(Float32MultiArray(data=vec.tolist()))

    def _imu_cb(self, msg: Imu):
        ori = msg.orientation
        acc = msg.linear_acceleration
        gyro = msg.angular_velocity
        imu_vec = np.array([ori.x, ori.y, ori.z, ori.w, acc.x, acc.y, acc.z], dtype=np.float32)
        gyro_vec = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)
        self.imu_pub.publish(Float32MultiArray(data=imu_vec.tolist()))
        self.gyro_pub.publish(Float32MultiArray(data=gyro_vec.tolist()))

    def _gps_cb(self, msg: NavSatFix):
        gps_vec = np.array([msg.latitude, msg.longitude, msg.altitude], dtype=np.float32)
        self.gps_pub.publish(Float32MultiArray(data=gps_vec.tolist()))


def main():
    try:
        _ = SensorVectorizerNode()
        rospy.loginfo("Sensor vectorizer node started")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        rospy.shutdown()


if __name__ == "__main__":
    main()

