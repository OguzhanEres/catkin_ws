#!/usr/bin/env python3
"""
SITL-specific parameter tuning for DRL training.
These parameters relax hardware checks that are problematic in simulation
but should remain enabled for real flight.

Usage:
    from drl_utils.sitl_params import apply_sitl_params, force_arm
    apply_sitl_params()  # Call once after MAVROS connection
    force_arm()  # Force arm ignoring pre-arm checks
"""
import rospy
from mavros_msgs.srv import ParamSet, ParamGet, CommandBool, CommandLong
from mavros_msgs.msg import ParamValue, State
import time


# Parameters to set for simulation training
# Format: (param_name, integer_value, real_value, description)
# Note: ArduPilot 4.x uses different param names than 3.x
SITL_PARAMS = [
    # CRITICAL: Disable ALL arming checks for simulation
    ("ARMING_CHECK", 0, 0.0, "Disable all arming checks"),
    
    # INS (Inertial Navigation) settings for Gazebo
    ("INS_ACCEL_FILTER", 10, 10.0, "Lower accel filter for sim stability"),
    ("INS_ACC_BODYFIX", 2, 2.0, "Use body-fixed accel"),
    ("INS_FAST_SAMPLE", 0, 0.0, "Disable fast sampling in sim"),
    ("INS_GYR_CAL", 0, 0.0, "Disable gyro cal on startup"),
    ("INS_TRIM_OPTION", 0, 0.0, "Disable trim in sim"),
    ("INS_GYRO_RATE", 0, 0.0, "Disable gyro rate check"),
    ("INS_HNTCH_ENABLE", 0, 0.0, "Disable harmonic notch filter"),
    ("INS_HNTC2_ENABLE", 0, 0.0, "Disable second harmonic notch filter"),
    
    # Scheduler - lower rate for Gazebo compatibility
    ("SCHED_LOOP_RATE", 50, 50.0, "Lower loop rate for Gazebo"),
    
    # EKF3 settings - ArduPilot 4.x uses EK3_ prefix
    ("EK3_CHECK_SCALE", 200, 200.0, "Very relaxed EKF3 check scaling"),
    ("EK3_ACC_P_NSE", 1, 1.0, "Higher accel process noise"),
    ("EK3_POSNE_M_NSE", 1, 1.0, "Higher position measurement noise"),
    ("EK3_GPS_CHECK", 0, 0.0, "Disable EKF3 GPS checks"),
    ("EK3_IMU_MASK", 3, 3.0, "Use both IMUs"),
    ("EK3_MAG_CAL", 0, 0.0, "Disable mag calibration in EKF"),
    ("EK3_MAG_M_NSE", 0, 0.5, "Higher mag measurement noise"),
    ("EK3_YAW_M_NSE", 0, 0.5, "Higher yaw measurement noise"),
    
    # Compass settings for simulation - disable anomaly detection
    ("COMPASS_USE", 1, 1.0, "Use compass (keep enabled)"),
    ("COMPASS_LEARN", 0, 0.0, "Disable compass learning in sim"),
    ("COMPASS_ENABLE", 1, 1.0, "Enable compass"),
    ("COMPASS_AUTODEC", 0, 0.0, "Disable auto declination"),
    ("COMPASS_MOTCT", 0, 0.0, "Disable motor compensation"),
    
    # Safety and failsafe - disable for sim
    ("BRD_SAFETY_DEFLT", 0, 0.0, "Disable safety switch default"),
    ("FS_THR_ENABLE", 0, 0.0, "Disable throttle failsafe"),
    ("FS_GCS_ENABLE", 0, 0.0, "Disable GCS failsafe"),
    ("FS_EKF_ACTION", 0, 0.0, "Disable EKF failsafe action"),
    
    # Position source settings
    ("EK3_SRC1_POSXY", 3, 3.0, "Use GPS for position"),
    ("EK3_SRC1_VELXY", 3, 3.0, "Use GPS for velocity"),
    ("EK3_SRC1_POSZ", 1, 1.0, "Use baro for altitude"),
    ("EK3_SRC1_YAW", 1, 1.0, "Use compass for yaw"),
]


def apply_sitl_params(timeout: float = 30.0) -> bool:
    """
    Apply simulation-specific parameters via MAVROS.
    
    Args:
        timeout: Max seconds to wait for param service
        
    Returns:
        True if critical params (ARMING_CHECK) set successfully
    """
    rospy.loginfo("Applying SITL parameters for DRL training...")
    
    try:
        rospy.wait_for_service("/mavros/param/set", timeout=timeout)
    except rospy.ROSException:
        rospy.logwarn("MAVROS param service not available")
        return False
    
    param_set = rospy.ServiceProxy("/mavros/param/set", ParamSet)
    success_count = 0
    critical_success = False
    
    for param_name, int_val, real_val, desc in SITL_PARAMS:
        try:
            pv = ParamValue()
            pv.integer = int_val
            pv.real = real_val
            
            resp = param_set(param_id=param_name, value=pv)
            if resp.success:
                rospy.logdebug(f"Set {param_name} = {int_val} ({desc})")
                success_count += 1
                if param_name == "ARMING_CHECK":
                    critical_success = True
                    rospy.loginfo("ARMING_CHECK set to 0 - pre-arm checks disabled")
            else:
                # Try alternative: some params need float only
                pv2 = ParamValue()
                pv2.integer = 0
                pv2.real = float(int_val)
                resp2 = param_set(param_id=param_name, value=pv2)
                if resp2.success:
                    success_count += 1
                    if param_name == "ARMING_CHECK":
                        critical_success = True
                else:
                    rospy.logdebug(f"Param {param_name} not available (may not exist in this ArduPilot version)")
        except rospy.ServiceException as e:
            rospy.logdebug(f"Service call failed for {param_name}: {e}")
    
    rospy.loginfo(f"Applied {success_count}/{len(SITL_PARAMS)} SITL parameters")
    
    # Verify ARMING_CHECK is actually 0
    if not critical_success:
        rospy.logwarn("ARMING_CHECK could not be set via param service")
        rospy.logwarn("Make sure SITL was started with: -w --add-param-file=drl_sitl.parm")
    
    return critical_success


def get_param(param_name: str) -> float:
    """Get a parameter value from MAVROS."""
    try:
        rospy.wait_for_service("/mavros/param/get", timeout=5.0)
        param_get = rospy.ServiceProxy("/mavros/param/get", ParamGet)
        resp = param_get(param_id=param_name)
        if resp.success:
            return resp.value.real if resp.value.real != 0 else float(resp.value.integer)
    except Exception as e:
        rospy.logwarn(f"Failed to get {param_name}: {e}")
    return 0.0


def verify_sitl_mode() -> bool:
    """Check if we're running in SITL by checking a known param."""
    # SIM_SPEEDUP only exists in SITL
    try:
        rospy.wait_for_service("/mavros/param/get", timeout=5.0)
        param_get = rospy.ServiceProxy("/mavros/param/get", ParamGet)
        resp = param_get(param_id="SIM_SPEEDUP")
        return resp.success
    except Exception:
        return False


def force_arm(timeout: float = 10.0) -> bool:
    """
    Force arm the vehicle using MAVLink COMMAND_LONG with force flag.
    This bypasses pre-arm checks when ARMING_CHECK=0 isn't working.
    
    Uses MAV_CMD_COMPONENT_ARM_DISARM (400) with param2=21196 (magic number to force)
    
    Returns:
        True if arming succeeded
    """
    rospy.loginfo("Attempting force arm via COMMAND_LONG...")
    
    try:
        rospy.wait_for_service("/mavros/cmd/command", timeout=timeout)
        cmd_long = rospy.ServiceProxy("/mavros/cmd/command", CommandLong)
        
        # MAV_CMD_COMPONENT_ARM_DISARM = 400
        # param1 = 1 (arm)
        # param2 = 21196 (force arm magic number, bypasses checks)
        resp = cmd_long(
            broadcast=False,
            command=400,  # MAV_CMD_COMPONENT_ARM_DISARM
            confirmation=0,
            param1=1.0,   # 1 = arm
            param2=21196.0,  # Force arm magic number
            param3=0.0,
            param4=0.0,
            param5=0.0,
            param6=0.0,
            param7=0.0
        )
        
        if resp.success:
            rospy.loginfo("Force arm command accepted!")
            return True
        else:
            rospy.logwarn(f"Force arm failed with result: {resp.result}")
            return False
            
    except rospy.ServiceException as e:
        rospy.logerr(f"Force arm service call failed: {e}")
        return False


def wait_for_ekf_origin(timeout: float = 60.0) -> bool:
    """
    Wait for EKF to have a valid origin (fixes 'Need Position Estimate' error).
    
    Returns:
        True if EKF origin is set
    """
    rospy.loginfo("Waiting for EKF origin to be set...")
    start = time.time()
    
    while time.time() - start < timeout:
        try:
            state = rospy.wait_for_message("/mavros/state", State, timeout=2.0)
            # Check if system status indicates EKF is healthy
            # system_status 3 = MAV_STATE_STANDBY (ready), 4 = MAV_STATE_ACTIVE
            if state.system_status >= 3:
                rospy.loginfo("EKF origin appears to be set (system standby/active)")
                return True
        except rospy.ROSException:
            pass
        rospy.sleep(1.0)
    
    rospy.logwarn("Timeout waiting for EKF origin")
    return False
