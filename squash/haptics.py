"""
Haptic feedback for the squash game.

On a hit, both motors spin back and forth a few times (about half a second),
so you feel the hit through the Double Motor in your hand.

It runs in a background thread so the game loop never waits on Bluetooth.
If a wiggle is already playing, a new hit is ignored (no overlapping buzz).

Uses lelib's doubleMotor.run(speed) (positive = forward, negative = backward)
and stop().
"""
import threading
import time


class Haptics:
    def __init__(self, motor=None, speed_min=25, speed_max=60,
                 pulse_s=0.08, cycles=3, dry_run=False, log=print):
        """
        motor     connected lelib doubleMotor (None is fine for dry_run)
        speed_min wiggle speed at power 0 (soft hit)
        speed_max wiggle speed at power 1 (hard hit)
        pulse_s   seconds per direction
        cycles    forward+back pairs, so total time = cycles * 2 * pulse_s
        dry_run   print what would happen instead of driving the motor
        """
        self.motor = motor
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.pulse_s = pulse_s
        self.cycles = cycles
        self.dry_run = dry_run
        self.log = log
        self._lock = threading.Lock()
        self._busy = False

    @property
    def duration(self):
        """Seconds one wiggle takes."""
        return self.cycles * 2 * self.pulse_s

    @property
    def busy(self):
        return self._busy

    def speed_for(self, power):
        power = max(0.0, min(1.0, power))
        return self.speed_min + (self.speed_max - self.speed_min) * power

    def hit(self, power=0.5):
        """Start a wiggle. Returns True if started, False if one is already playing."""
        with self._lock:
            if self._busy:
                return False
            self._busy = True
        threading.Thread(target=self._play, args=(self.speed_for(power),), daemon=True).start()
        return True

    # --- internals ----------------------------------------------------------
    def _drive(self, speed):
        if self.dry_run:
            self.log(f"    motor run({speed:+.0f})")
        else:
            self.motor.run(speed)

    def _stop(self):
        if self.dry_run:
            self.log("    motor stop()")
        else:
            self.motor.stop()

    def _play(self, speed):
        try:
            start = time.time()
            phase = 0
            for _ in range(self.cycles):
                for sign in (+1, -1):
                    self._drive(sign * speed)
                    phase += 1
                    # schedule against the clock so slow Bluetooth calls don't stretch the total
                    wait = start + phase * self.pulse_s - time.time()
                    if wait > 0:
                        time.sleep(wait)
        except Exception as e:
            self.log(f"haptics error: {e}")
        finally:
            try:
                self._stop()
            except Exception as e:
                self.log(f"haptics stop error: {e}")
            self._busy = False
