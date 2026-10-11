"""
Try the haptic wiggle.

    python tools/haptics_test.py          dry run: prints what it would do, no hardware needed
    python tools/haptics_test.py --live   real motor (tap the green card first)

Plays a soft, a medium and a hard hit, then fires two hits back to back to
show that the second one is ignored while the first is still playing.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config
from haptics import Haptics

LIVE = "--live" in sys.argv


def main():
    motor = None
    if LIVE:
        from lelib import doubleMotor
        motor = doubleMotor()
        print("Connecting...")
        motor.connect(card_serial=config.CARD_SERIAL, card_color=config.CARD_COLOR)
        print("Connected.")
        time.sleep(1.0)

    h = Haptics(
        motor,
        speed_min=getattr(config, "HAPTIC_SPEED_MIN", 25),
        speed_max=getattr(config, "HAPTIC_SPEED_MAX", 60),
        pulse_s=getattr(config, "HAPTIC_PULSE_S", 0.08),
        cycles=getattr(config, "HAPTIC_CYCLES", 3),
        dry_run=not LIVE,
    )
    print(f"Mode: {'LIVE' if LIVE else 'DRY RUN'}   one wiggle takes {h.duration:.2f} s\n")

    try:
        for name, power in (("soft", 0.1), ("medium", 0.5), ("hard", 1.0)):
            print(f"{name} hit (power {power}):")
            t0 = time.time()
            h.hit(power)
            while h.busy:
                time.sleep(0.01)
            print(f"  took {time.time() - t0:.2f} s\n")
            time.sleep(1.0)

        print("Two hits back to back:")
        first = h.hit(0.5)
        second = h.hit(0.5)
        print(f"  first started: {first}, second started: {second} (second should be False)")
        while h.busy:
            time.sleep(0.01)
    finally:
        if LIVE:
            try:
                motor.stop()
                motor.disconnect()
            except Exception:
                pass
        print("\nDone.")


if __name__ == "__main__":
    main()
