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
        # Narrow FOV for virtual gimbal effect (±30° = 60° total side-to-side)
        self.lidar_fov_deg = float(rospy.get_param("~lidar_fov_deg", 60.0))
        self.lidar_front_center_deg = float(rospy.get_param("~lidar_front_center_deg", 0.0))
        # Normalization denominator (soft limit)
        self.lidar_max_range = float(rospy.get_param("~lidar_max_range", 10.0))

        # === GROUND EFFECT FILTER PARAMETERS ===
        self.pitch_ground_deg = float(rospy.get_param("~pitch_ground_deg", 15.0))
        # Safer threshold: 0.35m avoids eating walls but filters ground noise
        self.ground_min_range_m = float(rospy.get_param("~ground_min_range_m", 0.35))
        self.use_dynamic_ground_filter = rospy.get_param("~use_dynamic_ground_filter", False)
        self.ground_margin = float(rospy.get_param("~ground_margin", 1.15))

        # === CAMERA PARAMETERS ===
        self.cam_w = int(rospy.get_param("~camera_width", 32))
        self.cam_h = int(rospy.get_param("~camera_height", 24))

        # === DEBUG & SYNC ===
        self.debug_interval = int(rospy.get_param("~debug_interval", 100))
        self._frame_count = 0
        # Throttle range_max param updates to avoid overloading ROS master
        self.range_sync_hz = float(rospy.get_param("~range_sync_hz", 1.0))
        self._last_range_sync_time = 0.0

        # === STATE ===
        self._current_pitch_rad = 0.0
        self._current_altitude = None
        
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

        try:
            from geometry_msgs.msg import PoseStamped
            self.pose_sub = rospy.Subscriber(
                "/mavros/local_position/pose", PoseStamped, self._pose_cb, queue_size=1
            )
        except Exception:
            pass

        rospy.loginfo(
            "SensorVectorizer Ready: FOV=%.0f°, PitchThr=%.0f°, GroundMin=%.2fm, MaxRange=%.1fm",
            self.lidar_fov_deg, self.pitch_ground_deg, self.ground_min_range_m, self.lidar_max_range
        )

    def _pose_cb(self, msg):
        self._current_altitude = msg.pose.position.z

    def _imu_cb(self, msg: Imu):
        ori = msg.orientation
        acc = msg.linear_acceleration
        gyro = msg.angular_velocity

        # Extract pitch
        if _HAS_TF:
            q = [ori.x, ori.y, ori.z, ori.w]
            roll, pitch, yaw = euler_from_quaternion(q)
        else:
            roll, pitch, yaw = quaternion_to_euler_manual(ori.x, ori.y, ori.z, ori.w)
        
        self._current_pitch_rad = pitch

        # Publish IMU vec
        imu_vec = np.array(
            [ori.x, ori.y, ori.z, ori.w, acc.x, acc.y, acc.z],
            dtype=np.float32,
        )
        gyro_vec = np.array([gyro.x, gyro.y, gyro.z], dtype=np.float32)
        self.imu_pub.publish(Float32MultiArray(data=imu_vec.tolist()))
        self.gyro_pub.publish(Float32MultiArray(data=gyro_vec.tolist()))

    def _lidar_cb(self, msg: LaserScan):
        """
        Robust LiDAR Pipeline:
        1. Clean NaN/Inf -> msg.range_max (physical max)
        2. FOV Mask -> Mask outside ±FOV/2
        3. Ground Filter -> Mask short readings if pitching
        4. Min-Pooling -> Downsample
        5. Normalize -> Use self.lidar_max_range (logical max)
        """
        # Physical limits from sensor
        phys_range_max = float(msg.range_max)
        ranges = np.array(msg.ranges, dtype=np.float32)

        # 1. CLEAN INVALID VALUES
        # Replace nan/inf with physical range_max (implies open space or bad read)
        ranges = np.nan_to_num(ranges, nan=phys_range_max, posinf=phys_range_max, neginf=phys_range_max)

        if ranges.size == 0:
            return

        # 2. FOV MASK
        # Calculate angles for each ray
        if msg.angle_increment != 0.0 and self.lidar_fov_deg > 0.0:
            angles = msg.angle_min + np.arange(ranges.size, dtype=np.float32) * msg.angle_increment
            
            # Wrap angles to [-pi, pi] if needed (though usually LaserScan is already wrapped)
            angles = (angles + np.pi) % (2 * np.pi) - np.pi
            
            center_rad = np.deg2rad(self.lidar_front_center_deg)
            half_fov_rad = np.deg2rad(self.lidar_fov_deg) * 0.5
            
            # Create mask: |angle - center| <= half_fov
            # Handle wrap-around diff for safety
            diff = np.abs(angles - center_rad)
            diff = np.minimum(diff, 2*np.pi - diff)
            mask = diff <= half_fov_rad
            
            if mask.any():
                ranges = ranges[mask]
            else:
                # Fallback: if mask empty (shouldn't happen with normal params), keep full scan
                pass

        if ranges.size == 0:
            return

        # 3. PITCH-AWARE GROUND FILTER
        # If pitching down/up significantly, short readings are likely ground
        pitch_deg = abs(math.degrees(self._current_pitch_rad))
        
        if pitch_deg > self.pitch_ground_deg:
            # Mask ranges shorter than safety threshold to physical max (open space)
            # This suppresses false positives from ground, while keeping walls (usually further)
            # CAUTION: Very close walls might be masked, but better than false collision loop.
            ranges = np.where(ranges < self.ground_min_range_m, phys_range_max, ranges)

        # 4. MIN-POOLING DOWNSAMPLE
        # More robust than single-point sampling
        pooled = self._min_pool_ranges(ranges, self.lidar_bins, phys_range_max)

        # 5. NORMALIZE
        # Clip to [0, logical_max] and divide by logical_max
        # logical_max (self.lidar_max_range) might be 10.0m even if sensor sees 30m
        clipped = np.clip(pooled, 0.0, self.lidar_max_range)
        vec = (clipped / max(1e-6, self.lidar_max_range)).astype(np.float32)

        # 6. SYNC PARAM (Throttled)
        # Inform trainer about the range_max used for normalization
        now = rospy.get_time()
        if now - self._last_range_sync_time > (1.0 / self.range_sync_hz):
            try:
                rospy.set_param("/agent/lidar_range_max", self.lidar_max_range)
                self._last_range_sync_time = now
            except Exception:
                pass

        # Publish
        self.lidar_pub.publish(Float32MultiArray(data=vec.tolist()))

        # Debug
        self._frame_count += 1
        if self.debug_interval > 0 and self._frame_count % self.debug_interval == 0:
             rospy.logdebug(
                "LiDAR: pitch=%.1f°, raw_min=%.2fm, vec_min=%.2f, valid_rays=%d",
                pitch_deg, np.min(ranges) if len(ranges) else -1, np.min(vec), len(ranges)
            )

    def _min_pool_ranges(self, ranges: np.ndarray, num_bins: int, default_val: float) -> np.ndarray:
        n = len(ranges)
        if n == 0:
            return np.full(num_bins, default_val, dtype=np.float32)
        if n <= num_bins:
            res = np.full(num_bins, default_val, dtype=np.float32)
            res[:n] = ranges
            return res
            
        pooled = np.zeros(num_bins, dtype=np.float32)
        # Use np.array_split for more even distribution than integer math
        chunks = np.array_split(ranges, num_bins)
        
        for i, chunk in enumerate(chunks):
            if chunk.size > 0:
                pooled[i] = np.min(chunk)
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
