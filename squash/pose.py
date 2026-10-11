"""
Pose gate for the squash game: is your swinging arm raised?

    lift = (hip_y - wrist_y) / (hip_y - shoulder_y)

  lift 0 = wrist level with your hip, 1 = wrist level with your shoulder,
  above 1 = hand above the shoulder. It is relative to your own body, so it works at any
  distance from the camera.

The gate opens when lift goes above open_at and closes when it drops below close_at
(two different numbers so it doesn't flicker). A swing is only detected by the IMU a
moment AFTER it happens, by which time your arm may already be coming down, so
armed() answers "was the gate open at any point in the last remember_s seconds?"

Left/right: MediaPipe labels body parts from the person's point of view. We feed it
the unmirrored camera image. If the ring lands on the wrong wrist, set POSE_SWAP_LR = True.
"""
import os
import time
import urllib.request

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pose_landmarker_lite.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task")

# MediaPipe pose landmark indices
SIDE = {
    "right": {"shoulder": 12, "hip": 24, "wrist": 16},
    "left": {"shoulder": 11, "hip": 23, "wrist": 15},
}


class ArmGate:
    """Pure math: shoulder/hip/wrist points in, lift and an open/closed gate out."""

    def __init__(self, open_at=0.30, close_at=0.20, smooth=0.5,
                 min_visibility=0.5, remember_s=0.6):
        self.open_at = open_at
        self.close_at = close_at
        self.smooth = smooth
        self.min_visibility = min_visibility
        self.remember_s = remember_s
        self.lift = None
        self.open = False
        self.last_open = -1e9

    def _ok(self, p):
        return p is not None and getattr(p, "visibility", 1.0) >= self.min_visibility

    def update(self, shoulder, hip, wrist, now):
        if not (self._ok(shoulder) and self._ok(hip) and self._ok(wrist)):
            self.lift = None
            self.open = False
            return None
        torso = hip.y - shoulder.y            # image y grows downward
        if torso < 0.05:                      # body too small or upside down: don't trust it
            self.lift = None
            self.open = False
            return None
        raw = (hip.y - wrist.y) / torso
        self.lift = raw if self.lift is None else self.lift + self.smooth * (raw - self.lift)
        if not self.open and self.lift >= self.open_at:
            self.open = True
        elif self.open and self.lift <= self.close_at:
            self.open = False
        if self.open:
            self.last_open = now
        return self.lift

    def armed(self, now):
        """True if the gate is open now or was open within the last remember_s seconds."""
        return now - self.last_open <= self.remember_s


class PoseTracker:
    """MediaPipe PoseLandmarker + ArmGate."""

    def __init__(self, hand="right", swap_lr=False, gate=None, width=640,
                 model_path=MODEL_PATH, landmarker=None):
        self.hand = hand
        side = hand
        if swap_lr:
            side = "left" if hand == "right" else "right"
        self.idx = SIDE[side]
        self.gate = gate or ArmGate()
        self.width = width
        self._last_ts = 0
        self.points = None       # (shoulder, hip, wrist) normalized (x, y) for drawing

        if landmarker is not None:
            self._landmarker = landmarker           # for tests
        else:
            import mediapipe as mp                  # noqa: F401  (imported here so tests can skip it)
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python import vision as mp_vision
            if not os.path.exists(model_path):
                print(f"Downloading pose model to {model_path} ...")
                urllib.request.urlretrieve(MODEL_URL, model_path)
                print("Model download complete.")
            options = mp_vision.PoseLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.5,
            )
            self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def update(self, frame_bgr, now=None):
        """Returns (lift or None, gate_open, armed)."""
        import mediapipe as mp
        now = time.time() if now is None else now
        h, w = frame_bgr.shape[:2]
        if w > self.width:
            frame_bgr = cv2.resize(frame_bgr, (self.width, int(h * self.width / w)))
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts = int(now * 1000)
        if ts <= self._last_ts:                      # MediaPipe needs strictly increasing timestamps
            ts = self._last_ts + 1
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)

        self.points = None
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]
            s, hp, wr = lm[self.idx["shoulder"]], lm[self.idx["hip"]], lm[self.idx["wrist"]]
            self.gate.update(s, hp, wr, now)
            self.points = ((s.x, s.y), (hp.x, hp.y), (wr.x, wr.y))
        else:
            self.gate.update(None, None, None, now)
        return self.gate.lift, self.gate.open, self.gate.armed(now)
