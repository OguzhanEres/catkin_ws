#!/bin/bash

# PX4 SITL Configuration Script
# Sets necessary parameters for DRL training environment via MAVROS

echo "[INFO] Starting PX4 Parameter Configuration..."

# 1. Wait for ROS master
echo "[INFO] Waiting for ROS master..."
until rostopic list > /dev/null 2>&1; do
    sleep 1
done

# 2. Wait for MAVROS param service
echo "[INFO] Waiting for MAVROS param service..."
until rosservice list | grep -q "/mavros/param/set"; do
    sleep 1
done

# 3. Wait for FCU connection
echo "[INFO] Waiting for FCU connection..."
connected=false
while [ "$connected" = false ]; do
    if rostopic echo -n 1 /mavros/state | grep -q "connected: True"; then
        connected=true
        echo "[INFO] FCU Connected!"
    else
        echo "[WARN] FCU not connected yet. Waiting..."
        sleep 2
    fi
done

# Function to set param with retry
set_param() {
    local param=$1
    local value=$2
    echo "Setting $param = $value"
    
    # Try loop
    for i in {1..5}; do
        if rosrun mavros mavparam set "$param" "$value"; then
            return 0
        else
            echo "[WARN] Failed to set $param (attempt $i/5). Retrying..."
            sleep 1
        fi
    done
    echo "[ERR] Could not set $param after 5 attempts."
}

echo "----------------------------------------"
echo "Setting PX4 parameters..."
echo "----------------------------------------"

# Circuit Breakers
set_param CBRK_SUPPLY_CHK 894281
set_param CBRK_IO_SAFETY 22027

# Power / Battery
set_param BAT1_N_CELLS 0
set_param BAT1_SOURCE 0
set_param COM_POWER_COUNT 0

# Arming Checks
set_param COM_ARM_WO_GPS 1
set_param COM_ARM_MAG_ANG 180
set_param COM_ARM_CHK_ESCS 0
set_param SYS_HAS_BARO 0

# Failsafes
set_param COM_OF_LOSS_T 60.0
set_param COM_OBL_RC_ACT 0
set_param NAV_RCL_ACT 0
set_param NAV_DLL_ACT 0

# EKF2 Configuration
set_param EKF2_EV_CTRL 7
set_param EKF2_HGT_REF 3
set_param EKF2_EV_DELAY 0
set_param EKF2_EV_POS_X 0
set_param EKF2_EV_POS_Y 0
set_param EKF2_EV_POS_Z 0

echo "----------------------------------------"
echo "[INFO] Parameter configuration complete."
echo "[INFO] Please restart PX4 (e.g., 'reboot' in px4 console) for changes to fully take effect."
