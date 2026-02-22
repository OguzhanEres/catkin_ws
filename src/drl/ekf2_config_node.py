#!/usr/bin/env python3
"""
PX4 EKF2 Parameter Configuration for Vision-Based Navigation.

This node sets EKF2 parameters to use vision pose data (from gazebo_vision_bridge)
as the primary position/heading source. This is CRITICAL for sim2real transfer
as the same configuration works with real VIO sensors (Intel RealSense T265, ZED2).

Must run BEFORE takeoff to ensure EKF initializes with correct sources.
"""
import rospy
from mavros_msgs.srv import ParamSet, ParamGet
from mavros_msgs.msg import ParamValue


# PX4 EKF2 Parameters for Vision-Based Navigation
# Reference: https://docs.px4.io/main/en/advanced_config/tuning_the_ecl_ekf.html
EKF2_VISION_PARAMS = [
    # EKF2_EV_CTRL: External Vision Control Bitmask
    # Bit 0 (1): Horizontal position fusion
    # Bit 1 (2): Vertical position fusion 
    # Bit 2 (4): Velocity fusion
    # Bit 3 (8): Yaw fusion
    # Value 15 = All (1+2+4+8) - Use vision for everything
    ("EKF2_EV_CTRL", 15, "Vision control: pos+vel+yaw fusion"),
    
    # EKF2_HGT_REF: Height Reference Source
    # 0 = Barometer, 1 = GPS, 2 = Range sensor, 3 = Vision
    ("EKF2_HGT_REF", 3, "Use vision for height reference"),
    
    # EKF2_EV_DELAY: Vision measurement delay (ms)
    # Gazebo has minimal delay
    ("EKF2_EV_DELAY", 0, "Vision delay compensation (ms)"),
    
    # EKF2_EV_POS_X/Y/Z: Vision sensor position offsets (body frame)
    ("EKF2_EV_POS_X", 0.0, "Vision sensor X offset"),
    ("EKF2_EV_POS_Y", 0.0, "Vision sensor Y offset"),
    ("EKF2_EV_POS_Z", 0.0, "Vision sensor Z offset"),
    
    # EKF2_EV_NOISE_MD: Vision noise mode
    # 0 = Fixed noise, 1 = Vision message noise
    ("EKF2_EV_NOISE_MD", 0, "Use fixed vision noise"),
    
    # Disable GPS and Barometer as primary sources
    ("EKF2_GPS_CTRL", 0, "Disable GPS fusion"),
    ("EKF2_BARO_CTRL", 0, "Disable barometer fusion"),
    
    # Relaxed innovation gates for simulation
    ("EKF2_EVP_GATE", 10.0, "Vision position innovation gate"),
    ("EKF2_EVV_GATE", 10.0, "Vision velocity innovation gate"),
]


class EKF2ConfigNode:
    def __init__(self):
        rospy.init_node("ekf2_config", anonymous=False)
        
        self.timeout = rospy.get_param("~timeout", 30.0)
        self.retry_count = rospy.get_param("~retry_count", 5)
        
        rospy.loginfo("EKF2 Configuration Node starting...")
        
        # Wait for MAVROS param service
        try:
            rospy.wait_for_service("/mavros/param/set", timeout=self.timeout)
            rospy.wait_for_service("/mavros/param/get", timeout=self.timeout)
        except rospy.ROSException:
            rospy.logerr("MAVROS param services not available!")
            return
        
        self.param_set = rospy.ServiceProxy("/mavros/param/set", ParamSet)
        self.param_get = rospy.ServiceProxy("/mavros/param/get", ParamGet)
        
        # Apply parameters
        success_count = self._apply_params()
        rospy.loginfo("EKF2 Configuration: Applied %d/%d parameters", 
                     success_count, len(EKF2_VISION_PARAMS))
        
        # Verify critical params
        self._verify_params()
        
    def _set_param(self, name: str, value, description: str) -> bool:
        """Set a single parameter with retry logic."""
        for attempt in range(self.retry_count):
            try:
                pv = ParamValue()
                if isinstance(value, float):
                    pv.real = value
                    pv.integer = 0
                else:
                    pv.integer = int(value)
                    pv.real = float(value)
                
                resp = self.param_set(param_id=name, value=pv)
                if resp.success:
                    rospy.loginfo("  ✓ %s = %s (%s)", name, value, description)
                    return True
                    
            except rospy.ServiceException as e:
                rospy.logdebug("Attempt %d failed for %s: %s", attempt + 1, name, e)
                rospy.sleep(0.5)
        
        rospy.logwarn("  ✗ Failed to set %s", name)
        return False
    
    def _apply_params(self) -> int:
        """Apply all EKF2 vision parameters."""
        rospy.loginfo("Setting EKF2 parameters for Vision-Based Navigation:")
        success_count = 0
        
        for name, value, desc in EKF2_VISION_PARAMS:
            if self._set_param(name, value, desc):
                success_count += 1
                
        return success_count
    
    def _verify_params(self):
        """Verify critical parameters are set correctly."""
        rospy.loginfo("Verifying EKF2 configuration...")
        
        critical_params = [
            ("EKF2_EV_CTRL", 15),
            ("EKF2_HGT_REF", 3),
        ]
        
        all_ok = True
        for name, expected in critical_params:
            try:
                resp = self.param_get(param_id=name)
                if resp.success:
                    actual = resp.value.integer if resp.value.integer != 0 else int(resp.value.real)
                    if actual == expected:
                        rospy.loginfo("  ✓ %s = %d (OK)", name, actual)
                    else:
                        rospy.logwarn("  ✗ %s = %d (expected %d)", name, actual, expected)
                        all_ok = False
                else:
                    rospy.logwarn("  ? %s: Could not read", name)
                    all_ok = False
            except rospy.ServiceException:
                rospy.logwarn("  ? %s: Service error", name)
                all_ok = False
        
        if all_ok:
            rospy.loginfo("EKF2 is configured for Vision-Based Navigation! ✓")
        else:
            rospy.logwarn("EKF2 configuration incomplete. Vision data may not be used.")
            rospy.logwarn("You may need to restart PX4 SITL for params to take effect.")


if __name__ == "__main__":
    try:
        EKF2ConfigNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
