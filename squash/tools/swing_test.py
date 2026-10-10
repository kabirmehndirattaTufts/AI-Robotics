"""
Live swing counter. Uses SWING_THRESHOLD (and SWING_MAX) from config.py.

    python tools/swing_test.py

Swing the motor. Each counted swing prints its power (0 = barely, 1 = hard).
Press Ctrl+C to stop.
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from lelib import doubleMotor

THRESHOLD = config.SWING_THRESHOLD
SWING_MAX = getattr(config, "SWING_MAX", 11000)
GRAVITY = 1000.0
COLLECT_S = 0.3      # after the first reading over threshold, keep the biggest one for this long
COOLDOWN_S = 0.6     # ignore new swings right after one is counted
GYRO_DEADBAND = getattr(config, "STROKE_GYRO_DEADBAND", 200)  # |gyroY| below this = unsure


def acc_dev(imu):
    ax, ay, az = imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ
    return abs(math.sqrt(ax * ax + ay * ay + az * az) - GRAVITY)


def stroke_from_gyro(gy):
    """gyroY at the swing peak: positive = forehand, negative = backhand (from your recordings)."""
    if gy > GYRO_DEADBAND:
        return "forehand"
    if gy < -GYRO_DEADBAND:
        return "backhand"
    return "unsure"


def power(peak):
    return max(0.0, min(1.0, (peak - THRESHOLD) / (SWING_MAX - THRESHOLD)))


def main():
    if THRESHOLD is None:
        print("Set SWING_THRESHOLD in config.py first.")
        return
    dm = doubleMotor()
    print("Connecting...")
    dm.connect(card_serial=config.CARD_SERIAL, card_color=config.CARD_COLOR)
    print(f"Connected. Threshold {THRESHOLD}. Swing away (Ctrl+C to stop).")
    time.sleep(1.0)

    count = 0
    collecting_since = None
    peak = 0.0
    peak_gy = 0.0
    last_swing_time = -1e9
    try:
        while True:
            imu = dm.imu_device
            value = acc_dev(imu)
            now = time.time()
            if collecting_since is None:
                if value >= THRESHOLD and now - last_swing_time > COOLDOWN_S:
                    collecting_since = now
                    peak = value
                    peak_gy = imu.gyroscopeY
            else:
                if value > peak:
                    peak = value
                    peak_gy = imu.gyroscopeY
                if now - collecting_since >= COLLECT_S:
                    count += 1
                    last_swing_time = now
                    bar = "#" * int(power(peak) * 20)
                    stroke = stroke_from_gyro(peak_gy)
                    print(f"SWING {count:3d}   {stroke:9}   peak {peak:6.0f}   power {power(peak):.2f}  {bar}   (gyroY {peak_gy:5.0f})")
                    collecting_since = None
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            dm.disconnect()
        except Exception:
            pass
        print(f"\nDone. Counted {count} swings.")


if __name__ == "__main__":
    main()