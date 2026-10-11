"""
Find out which motor commands actually move the Double Motor, and how long
each Bluetooth call takes.

    python tools/motor_check.py

Hold the motor in your hand (or rest your palm on it). Tap the green card
first. It runs each test below with a 2 s pause in between; note which ones
you can feel or see move, then paste the output and your notes.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from lelib import doubleMotor


def timed(label, fn, *args, **kwargs):
    t0 = time.time()
    try:
        fn(*args, **kwargs)
        print(f"      {label}: call took {(time.time() - t0) * 1000:.0f} ms")
    except Exception as e:
        print(f"      {label}: ERROR {type(e).__name__}: {e}")


def pause(s=2.0):
    time.sleep(s)


def main():
    dm = doubleMotor()
    print("Connecting...")
    dm.connect(card_serial=config.CARD_SERIAL, card_color=config.CARD_COLOR)
    print("Connected.\n")
    time.sleep(1.0)
    try:
        print("A: run(50) forward for 1 s")
        timed("run(50)", dm.run, 50)
        time.sleep(1.0)
        timed("stop()", dm.stop)
        pause()

        print("B: run(-50) backward for 1 s")
        timed("run(-50)", dm.run, -50)
        time.sleep(1.0)
        timed("stop()", dm.stop)
        pause()

        print("C: run(100) forward for 1 s (full speed)")
        timed("run(100)", dm.run, 100)
        time.sleep(1.0)
        timed("stop()", dm.stop)
        pause()

        print("D: left motor only, 1 s")
        timed("run_left()", dm.run_left)
        time.sleep(1.0)
        timed("stop()", dm.stop)
        pause()

        print("E: right motor only, 1 s")
        timed("run_right()", dm.run_right)
        time.sleep(1.0)
        timed("stop()", dm.stop)
        pause()

        for pulse in (0.08, 0.15, 0.25):
            print(f"F: wiggle at speed 100, {pulse * 1000:.0f} ms per direction, 3 cycles")
            start = time.time()
            for _ in range(3):
                dm.run(100)
                time.sleep(pulse)
                dm.run(-100)
                time.sleep(pulse)
            dm.stop()
            print(f"      took {time.time() - start:.2f} s (ideal {pulse * 6:.2f} s)")
            pause()

        print("\nDone.")
    finally:
        try:
            dm.stop()
            dm.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    main()
