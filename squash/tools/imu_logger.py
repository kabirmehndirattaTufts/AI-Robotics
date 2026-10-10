"""
M0: see what the Double Motor IMU exposes, then record labeled swing data.

    python tools/imu_logger.py discover
        Connects, lists every numeric field on the IMU object, and prints them
        live for 8 seconds. Move/tilt/swing the motor and watch what changes.

    python tools/imu_logger.py record <label> <seconds>
        Logs all numeric IMU fields to data/imu_<label>_<time>.csv at up to
        100 Hz, and reports how often the values really update.
        Labels to use: swing, still, other (walking, waving, etc.)
"""
import csv
import os
import sys
import time

# lelib.py lives in the parent folder (squash/)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import legoeducation as le
from lelib import doubleMotor

CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0997"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def numeric_fields(obj):
    """Names of public, non-callable, numeric attributes on obj."""
    names = []
    for name in dir(obj):
        if name.startswith("_"):
            continue
        try:
            value = getattr(obj, name)
        except Exception:
            continue
        if callable(value) or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            names.append(name)
    return names


def snapshot(obj, names):
    row = []
    for n in names:
        try:
            row.append(float(getattr(obj, n)))
        except Exception:
            row.append(float("nan"))
    return row


def connect():
    dm = doubleMotor()
    print("Connecting to Double Motor (green card 0997)...")
    dm.connect(card_serial=CARD_SERIAL, card_color=CARD_COLOR)
    print("Connected.")
    time.sleep(1.0)  # let the first IMU notifications arrive
    return dm


def discover(dm):
    imu = dm.imu_device
    names = numeric_fields(imu)
    print("\nNumeric IMU fields found:")
    for n in names:
        print("  ", n)
    print("\nLive values for 8 s. Tilt it, shake it, swing it.\n")
    t_end = time.time() + 8
    while time.time() < t_end:
        vals = snapshot(dm.imu_device, names)
        print("  ".join(f"{n}={v:8.2f}" for n, v in zip(names, vals)))
        time.sleep(0.25)


def record(dm, label, seconds):
    imu = dm.imu_device
    names = numeric_fields(imu)
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"imu_{label}_{time.strftime('%H%M%S')}.csv")

    for k in (3, 2, 1):
        print(f"Starting in {k}...")
        time.sleep(1)
    print(f"GO: recording '{label}' for {seconds} s")

    rows = []
    t0 = time.time()
    last = None
    changes = 0
    while time.time() - t0 < seconds:
        vals = snapshot(dm.imu_device, names)
        if last is not None and vals != last:
            changes += 1
        last = vals
        rows.append([time.time() - t0] + vals)
        time.sleep(0.01)

    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t"] + names + ["label"])
        for r in rows:
            w.writerow(r + [label])

    print(f"Saved {len(rows)} rows to {path}")
    print(f"Value changes: {changes} in {seconds} s, about {changes / seconds:.0f} real updates per second")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("discover", "record"):
        print(__doc__)
        return
    dm = connect()
    try:
        if sys.argv[1] == "discover":
            discover(dm)
        else:
            label = sys.argv[2] if len(sys.argv) > 2 else "swing"
            seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 10
            record(dm, label, seconds)
    finally:
        try:
            dm.disconnect()
        except Exception:
            pass
        print("Disconnected.")


if __name__ == "__main__":
    main()
