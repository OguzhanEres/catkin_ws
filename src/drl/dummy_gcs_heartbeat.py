#!/usr/bin/env python3
"""
Dummy GCS heartbeat sender for PX4 SITL.
Sends MAVLink 2.0 HEARTBEAT messages to prevent datalink failsafe.

Usage:
    python3 dummy_gcs_heartbeat.py [--port 18570] [--rate 1.0]
"""
import argparse
import socket
import struct
import time


def crc_accumulate(byte, crc):
    """Accumulate CRC for MAVLink"""
    tmp = byte ^ (crc & 0xff)
    tmp ^= (tmp << 4) & 0xff
    return ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xffff


def crc_calculate(data, crc_extra):
    """Calculate MAVLink CRC"""
    crc = 0xffff
    for byte in data:
        crc = crc_accumulate(byte, crc)
    crc = crc_accumulate(crc_extra, crc)
    return crc


def mavlink2_heartbeat(seq):
    """
    Build a MAVLink 2.0 HEARTBEAT message.
    
    System ID: 255 (GCS)
    Component ID: 190 (MAV_COMP_ID_MISSIONPLANNER)
    Message ID: 0 (HEARTBEAT)
    """
    # MAVLink 2.0 header
    magic = 0xFD
    payload_len = 9
    incompat_flags = 0
    compat_flags = 0
    sysid = 255  # GCS system ID
    compid = 190  # MAV_COMP_ID_MISSIONPLANNER
    msgid_low = 0  # HEARTBEAT message ID (low byte)
    msgid_mid = 0
    msgid_high = 0
    
    # HEARTBEAT payload
    # uint32_t custom_mode, uint8_t type, uint8_t autopilot, 
    # uint8_t base_mode, uint8_t system_status, uint8_t mavlink_version
    custom_mode = 0
    mav_type = 6  # MAV_TYPE_GCS
    autopilot = 8  # MAV_AUTOPILOT_INVALID
    base_mode = 0
    system_status = 0  # MAV_STATE_UNINIT
    mavlink_version = 3  # MAVLink version
    
    payload = struct.pack('<IBBBBB',
        custom_mode,
        mav_type,
        autopilot,
        base_mode,
        system_status,
        mavlink_version
    )
    
    # Build message without CRC
    header = struct.pack('<BBBBBBBBB',
        magic,
        payload_len,
        incompat_flags,
        compat_flags,
        seq & 0xFF,
        sysid,
        compid,
        msgid_low,
        msgid_mid
    )
    
    # CRC calculation (exclude magic byte, include CRC_EXTRA for HEARTBEAT=50)
    crc_data = header[1:] + payload
    crc = crc_calculate(crc_data, 50)  # CRC_EXTRA for HEARTBEAT is 50
    
    return header + payload + struct.pack('<H', crc)


def main():
    parser = argparse.ArgumentParser(description='Send GCS heartbeat to PX4 SITL')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Target host (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=18570,
                        help='Target UDP port (default: 18570 for PX4 SITL GCS link)')
    parser.add_argument('--rate', type=float, default=1.0,
                        help='Heartbeat rate in Hz (default: 1.0)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Print status messages')
    args = parser.parse_args()
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    target = (args.host, args.port)
    interval = 1.0 / args.rate
    seq = 0
    
    print(f"Sending GCS heartbeat to {target[0]}:{target[1]} at {args.rate} Hz")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            msg = mavlink2_heartbeat(seq)
            sock.sendto(msg, target)
            if args.verbose:
                print(f"Sent heartbeat #{seq}")
            seq = (seq + 1) & 0xFF
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped")
    finally:
        sock.close()


if __name__ == '__main__':
    main()
