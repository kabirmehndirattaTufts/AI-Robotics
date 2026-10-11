"""
Swing detector that runs in its own thread, so a slow Bluetooth read never
stalls the camera loop.

    det = SwingDetector(motor, threshold=2250, swing_max=12500)
    det.start()
    ...
    for ev in det.pop_events():     # call every frame
        ev["t_onset"]   when the swing started (time.time())
        ev["power"]     0..1
        ev["stroke"]    'forehand' / 'backhand' / 'unsure'
    det.mute(1.0)                   # ignore swings for 1 s (while the haptic wiggle shakes the motor)
    det.stop()

Same logic as tools/swing_test.py: the first reading over the threshold starts a
short collection window, and the biggest reading in it is the swing's peak.
"""
import math
import queue
import threading
import time

GRAVITY = 1000.0


class SwingDetector:
    def __init__(self, motor, threshold, swing_max, collect_s=0.15,
                 cooldown_s=0.6, gyro_deadband=200):
        self.motor = motor
        self.threshold = threshold
        self.swing_max = swing_max
        self.collect_s = collect_s
        self.cooldown_s = cooldown_s
        self.gyro_deadband = gyro_deadband
        self._events = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._mute_until = 0.0
        self.last_error = None

    # --- control ---------------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def mute(self, seconds):
        self._mute_until = max(self._mute_until, time.time() + seconds)

    def inject(self, power, stroke="unsure"):
        """Fake a swing (keyboard testing without the motor)."""
        t = time.time()
        self._events.put({"t_onset": t, "t_detect": t, "power": power, "peak": 0.0, "stroke": stroke})

    def pop_events(self):
        out = []
        while True:
            try:
                out.append(self._events.get_nowait())
            except queue.Empty:
                return out

    # --- helpers ---------------------------------------------------------------
    @staticmethod
    def acc_dev(imu):
        ax, ay, az = imu.accelerometerX, imu.accelerometerY, imu.accelerometerZ
        return abs(math.sqrt(ax * ax + ay * ay + az * az) - GRAVITY)

    def stroke_from_gyro(self, gy):
        if gy > self.gyro_deadband:
            return "forehand"
        if gy < -self.gyro_deadband:
            return "backhand"
        return "unsure"

    def power_of(self, peak):
        return max(0.0, min(1.0, (peak - self.threshold) / (self.swing_max - self.threshold)))

    # --- the thread ------------------------------------------------------------
    def _run(self):
        collecting_since = None
        peak = 0.0
        peak_gy = 0.0
        last_swing = -1e9
        while not self._stop.is_set():
            try:
                imu = self.motor.imu_device          # always re-read, never cache
                value = self.acc_dev(imu)
                gy = imu.gyroscopeY
            except Exception as e:                   # a bad read should not kill the thread
                self.last_error = e
                time.sleep(0.05)
                continue
            now = time.time()
            if collecting_since is None:
                if (value >= self.threshold and now - last_swing > self.cooldown_s
                        and now >= self._mute_until):
                    collecting_since = now
                    peak, peak_gy = value, gy
            else:
                if value > peak:
                    peak, peak_gy = value, gy
                if now - collecting_since >= self.collect_s:
                    last_swing = now
                    self._events.put({
                        "t_onset": collecting_since, "t_detect": now,
                        "power": self.power_of(peak), "peak": peak,
                        "stroke": self.stroke_from_gyro(peak_gy),
                    })
                    collecting_since = None
            time.sleep(0.01)
