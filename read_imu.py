#!/usr/bin/env python3
"""
Read and display IMU sensor outputs from the Unitree Go2 MuJoCo simulation.
Sensors: imu_accel (accelerometer), imu_gyro (gyroscope), imu_quat (orientation quaternion)
"""

import mujoco
import numpy as np
import os
import sys
import time


def print_imu(data: mujoco.MjData, step: int):
    accel = data.sensor("imu_accel").data.copy()
    gyro  = data.sensor("imu_gyro").data.copy()
    quat  = data.sensor("imu_quat").data.copy()

    # Convert quaternion (w, x, y, z) to roll/pitch/yaw
    w, x, y, z = quat
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    pitch = np.arcsin(np.clip(2*(w*y - z*x), -1.0, 1.0))
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

    print(f"--- Step {step:6d} | t={data.time:.3f}s ---")
    print(f"  Accel  [m/s²]  x={accel[0]:+7.3f}  y={accel[1]:+7.3f}  z={accel[2]:+7.3f}")
    print(f"  Gyro   [rad/s] x={gyro[0]:+7.3f}  y={gyro[1]:+7.3f}  z={gyro[2]:+7.3f}")
    print(f"  Quat   [w,x,y,z] {quat[0]:+.4f} {quat[1]:+.4f} {quat[2]:+.4f} {quat[3]:+.4f}")
    print(f"  RPY    [deg]   r={np.degrees(roll):+7.2f}  p={np.degrees(pitch):+7.2f}  y={np.degrees(yaw):+7.2f}")


def main():
    controllers_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'controllers')
    sys.path.insert(0, controllers_dir)

    import imu_bridge

    print("Waiting for go2_walk.py to publish state...")
    PRINT_INTERVAL = 0.5   # seconds of sim time between console prints
    next_print     = 0.0

    # Wait until go2_walk.py starts publishing
    while True:
        result = imu_bridge.get_state()
        if result is not None:
            break
        time.sleep(0.01)

    print("Connected — IMU readings printed every 0.5 s of sim time\n")

    # Main read loop
    while True:
        result = imu_bridge.get_state()
        if result is not None:
            model, data, step, sim_time = result
        if result is not None and sim_time >= next_print:
            print_imu(data, step)
            next_print += PRINT_INTERVAL
        time.sleep(0.01)

if __name__ == "__main__":
    main()
