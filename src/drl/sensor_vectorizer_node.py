#!/usr/bin/env python3
"""
Sensor vectorizer node.
Converts LiDAR, camera, IMU, gyro, and GPS inputs into fixed-size vectors.

Enhanced with:
- Pitch-aware ground effect filtering
- Min-pooling downsampling for robustness
- Configurable narrow FOV (default 60°)
- Range_max synchronization via ROS param
"""
import cv2
import math
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, Imu, LaserScan, NavSatFix
from std_msgs.msg import Float32MultiArray

# Try to import tf.transformations, fallback to manual quaternion conversion
try:
    from tf.transformations import euler_from_quaternion
    _HAS_TF = True
except ImportError:
    _HAS_TF = False
    rospy.logwarn("tf.transformations not available, using manual quaternion->euler")


def quaternion_to_euler_manual(x, y, z, w):
    """Manual quaternion to euler conversion (roll, pitch, yaw)."""
    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


class SensorVectorizerNode:
    def __init__(self):
        rospy.init_node("sensor_vectorizer", anonymous=False)

        # === TOPIC PARAMETERS ===
        self.lidar_topic = rospy.get_param("~lidar_topic", "/scan")
        self.camera_topic = rospy.get_param("~camera_topic", "/front_camera/image_raw")
        self.imu_topic = rospy.get_param("~imu_topic", "/mavros/imu/data")
        self.gps_topic = rospy.get_param("~gps_topic", "/mavros/global_position/global")

        # === LIDAR PROCESSING PARAMETERS ===
        self.lidar_bins = int(rospy.get_param("~lidar_bins", 180))
        # Narrow FOV for virtual gimbal effect (±30° = 60° total)
        self.lidar_fov_deg = float(rospy.get_param("~lidar_fov_deg", 60.0))
        self.lidar_front_center_deg = float(rospy.get_param("~lidar_front_center_deg", 0.0))

        # === GROUND EFFECT FILTER PARAMETERS ===
        self.pitch_ground_deg = float(rospy.get_param("~pitch_ground_deg", 15.0))
        self.ground_min_range_m = float(rospy.get_param("~ground_min_range_m", 0.50))
        self.use_dynamic_ground_filter = rospy.get_param("~use_dynamic_ground_filter", False)
        self.ground_margin = float(rospy.get_param("~ground_margin", 1.15))

        # === CAMERA PARAMETERS ===
        self.cam_w = int(rospy.get_param("~camera_width", 32))
        self.cam_h = int(rospy.get_param("~camera_height", 24))

        # === DEBUG ===
        self.debug_interval = int(rospy.get_param("~debug_interval", 100))  # Log every N frames
        self._frame_count = 0

        # === STATE ===
        self._current_pitch_rad = 0.0
        self._current_altitude = None  # For dynamic ground filter
        self._last_range_max = 10.0  # Default, updated from LaserScan

        self.bridge = CvBridge()

        # === PUBLISHERS ===
        self.lidar_pub = rospy.Publisher("/agent/lidar_vec", Float32MultiArray, queue_size=1)
        self.camera_pub = rospy.Publisher("/agent/camera_vec", Float32MultiArray, queue_size=1)
        self.imu_pub = rospy.Publisher("/agent/imu_vec", Float32MultiArray, queue_size=1)
        self.gyro_pub = rospy.Publisher("/agent/gyro_vec", Float32MultiArray, queue_size=1)
        self.gps_pub = rospy.Publisher("/agent/gps_vec", Float32MultiArray, queue_size=1)

        # === SUBSCRIBERS ===
        self.scan_sub = rospy.Subscriber(self.lidar_topic, LaserScan, self._lidar_cb, queue_size=1)
        self.image_sub = rospy.Subscriber(self.camera_topic, Image, self._camera_cb, queue_size=1)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self._imu_cb, queue_size=1)
        self.gps_sub = rospy.Subscriber(self.gps_topic, NavSatFix, self._gps_cb, queue_size=1)

        # Optional: Subscribe to local position for altitude (dynamic ground filter)
        try:
            from geometry_msgs.msg import PoseStamped
            self.pose_sub = rospy.Subscriber(
                "/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=1
            )
        except Exception:
            rospy.logwarn("Could not subscribe to local_position/pose for altitude")

        rospy.loginfo(
            "SensorVectorizer initialized: FOV=%.0f°, bins=%d, ground_filter=%s",
            self.lidar_fov_deg, self.lidar_bins,
            "dynamic" if self.use_dynamic_ground_filter else "static"
        )

    def _pose_cb(self, msg):
        """Extract altitude for dynamic ground filter."""
        self._current_altitude = msg.pose.position.z

    def _imu_cb(self, msg: Imu):
        """
        Process IMU data:
        1. Extract pitch from quaternion (for ground effect filter)
        2. Publish IMU vector (orientation + acceleration)
        3. Publish gyro vector
        """
        ori = msg.orientation
        acc = msg.linear_acceleration
        gyro = msg.angular_velocity

        # --- Extract pitch from quaternion ---
        if _HAS_TF:
            q = [ori.x, ori.y, ori.z, ori.w]
            roll, pitch, yaw = euler_from_quaternion(q)
        else:
            roll, pitch, yaw = quaternion_to_euler_manual(ori.x, ori.y, ori.z, ori.w)

        self._current_pitch_rad = pitch

        # --- Publish IMU vector (unchanged format for NN compatibility) ---
        imu_vec = np.array(
            [ori.x, ori.y, ori.z, ori.w, acc.x, acc.y, acc.z],
            dtype=np.float32,
        )
        gyro_vec = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)
        self.imu_pub.publish(Float32MultiArray(data=imu_vec.tolist()))
        self.gyro_pub.publish(Float32MultiArray(data=gyro_vec.tolist()))

    def _lidar_cb(self, msg: LaserScan):
        """
        Process LiDAR data with robust pipeline:
        1. Clean inf/nan values
        2. Apply FOV mask (narrow front view)
        3. Apply pitch-aware ground effect filter
        4. Min-pooling downsampling
        5. Normalize to [0, 1]
        6. Publish range_max to ROS param
        """
        ranges = np.array(msg.ranges, dtype=np.float32)
        range_max = float(msg.range_max)
        self._last_range_max = range_max

        # === 1. CLEAN INF/NAN VALUES ===
        # All invalid readings become range_max (open space assumption)
        ranges = np.nan_to_num(ranges, nan=range_max, posinf=range_max, neginf=range_max)

        if ranges.size == 0:
            return

        # === 2. FOV MASK (narrow front view) ===
        if msg.angle_increment != 0.0 and self.lidar_fov_deg > 0.0:
            angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
            half_fov = np.deg2rad(self.lidar_fov_deg) * 0.5
            center = np.deg2rad(self.lidar_front_center_deg)
            mask = (angles >= center - half_fov) & (angles <= center + half_fov)
            if mask.any():
                ranges = ranges[mask]
            else:
                rospy.logwarn_throttle(5.0, "FOV mask resulted in empty ranges, using full scan")

        if ranges.size == 0:
            return

        # === 3. PITCH-AWARE GROUND EFFECT FILTER ===
        pitch_deg = abs(math.degrees(self._current_pitch_rad))

        if pitch_deg > self.pitch_ground_deg:
            # Drone is pitching significantly - ground may appear as obstacle
            if self.use_dynamic_ground_filter and self._current_altitude is not None:
                # Dynamic filter: Calculate expected ground distance
                pitch_rad = abs(self._current_pitch_rad)
                if pitch_rad > 0.01:  # Avoid division by zero
                    # d_ground = altitude / tan(|pitch|)
                    d_ground = self._current_altitude / math.tan(pitch_rad)
                    ground_threshold = d_ground * self.ground_margin
                    # Replace readings below threshold with range_max
                    ranges = np.where(ranges < ground_threshold, range_max, ranges)
            else:
                # Static filter: Use fixed minimum range threshold
                ranges = np.where(ranges < self.ground_min_range_m, range_max, ranges)

        # === 4. MIN-POOLING DOWNSAMPLING ===
        # Split ranges into lidar_bins segments and take min of each
        # This is more robust than single-point sampling and won't miss narrow obstacles
        pooled = self._min_pool_ranges(ranges, self.lidar_bins, range_max)

        # === 5. NORMALIZE TO [0, 1] ===
        denom = max(1e-6, range_max)
        vec = (pooled / denom).astype(np.float32)

        # === 6. PUBLISH RANGE_MAX TO ROS PARAM ===
        # This allows trainer to denormalize correctly
        try:
            rospy.set_param("/agent/lidar_range_max", range_max)
        except Exception:
            pass  # Non-critical

        # === 7. PUBLISH LIDAR VECTOR ===
        self.lidar_pub.publish(Float32MultiArray(data=vec.tolist()))

        # === DEBUG LOGGING ===
        self._frame_count += 1
        if self.debug_interval > 0 and self._frame_count % self.debug_interval == 0:
            min_raw = float(np.min(ranges)) if ranges.size > 0 else -1
            rospy.logdebug(
                "LiDAR[%d]: pitch=%.1f°, min_range=%.2fm, range_max=%.1fm, fov=%.0f°",
                self._frame_count, pitch_deg, min_raw, range_max, self.lidar_fov_deg
            )

    def _min_pool_ranges(self, ranges: np.ndarray, num_bins: int, default_val: float) -> np.ndarray:
        """
        Min-pooling: Split ranges into num_bins segments, take min of each.
        This is more robust than single-point sampling:
        - Won't miss narrow obstacles between sample points
        - More resistant to single-point noise
        """
        n = len(ranges)
        if n == 0:
            return np.full(num_bins, default_val, dtype=np.float32)

        if n <= num_bins:
            # Not enough points to pool, pad with default
            result = np.full(num_bins, default_val, dtype=np.float32)
            result[:n] = ranges
            return result

        # Split into segments and take min of each
        pooled = np.zeros(num_bins, dtype=np.float32)
        segment_size = n / num_bins

        for i in range(num_bins):
            start_idx = int(i * segment_size)
            end_idx = int((i + 1) * segment_size)
            if end_idx > start_idx:
                pooled[i] = np.min(ranges[start_idx:end_idx])
            else:
                pooled[i] = default_val

        return pooled

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

    def _gps_cb(self, msg: NavSatFix):
        gps_vec = np.array([msg.latitude, msg.longitude, msg.altitude], dtype=np.float32)
        self.gps_pub.publish(Float32MultiArray(data=gps_vec.tolist()))


if __name__ == "__main__":
    try:
        node = SensorVectorizerNode()
        rospy.loginfo("Sensor vectorizer node started (enhanced LiDAR processing)")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
