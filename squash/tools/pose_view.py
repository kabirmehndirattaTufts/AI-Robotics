"""
M3: see the pose gate (is your swinging arm raised?).

    python tools/pose_view.py            pose only
    python tools/pose_view.py --tags     pose + AprilTags together (checks the speed of both)

Stand where you will play. You should see a ring on your SWING WRIST (right hand),
a line from your shoulder to your hip, and a bar showing how high the wrist is:
bottom of the bar = hip level, the "1" mark = shoulder level. The big word says
GATE OPEN when your arm is raised high enough.

If the ring is on your LEFT wrist, set POSE_SWAP_LR = True in config.py.

Keys: q = quit.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

import config
from pose import ArmGate, PoseTracker
from vision import TagTracker, open_camera

WITH_TAGS = "--tags" in sys.argv


def draw_bar(frame, gate):
    h = frame.shape[0]
    x0, x1 = 20, 50
    top, bot = int(h * 0.15), int(h * 0.75)           # lift 1.5 at the top, -0.25 at the bottom
    span = 1.75

    def y_of(v):
        return int(bot - (v + 0.25) / span * (bot - top))

    cv2.rectangle(frame, (x0, top), (x1, bot), (60, 60, 60), -1)
    for v, name in ((0.0, "hip"), (1.0, "shoulder")):
        cv2.line(frame, (x0 - 6, y_of(v)), (x1 + 6, y_of(v)), (200, 200, 200), 1)
        cv2.putText(frame, name, (x1 + 10, y_of(v) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    cv2.line(frame, (x0 - 6, y_of(gate.open_at)), (x1 + 6, y_of(gate.open_at)), (0, 220, 255), 2)
    if gate.lift is not None:
        v = max(-0.25, min(1.5, gate.lift))
        cv2.rectangle(frame, (x0, y_of(v)), (x1, bot), (80, 220, 80) if gate.open else (160, 160, 160), -1)


def main():
    gate = ArmGate(
        open_at=getattr(config, "GATE_OPEN_AT", 0.30),
        close_at=getattr(config, "GATE_CLOSE_AT", 0.20),
        remember_s=getattr(config, "GATE_REMEMBER_S", 0.6),
    )
    pose = PoseTracker(
        hand=getattr(config, "SWING_HAND", "right"),
        swap_lr=getattr(config, "POSE_SWAP_LR", False),
        gate=gate,
    )
    tags = None
    if WITH_TAGS:
        tags = TagTracker(
            tag_size_m=getattr(config, "TAG_SIZE_MM", 100) / 1000.0,
            left_id=getattr(config, "TAG_LEFT_ID", 0),
            right_id=getattr(config, "TAG_RIGHT_ID", 1),
        )
    cap = open_camera(getattr(config, "CAMERA_INDEX", None))
    print("Press q in the window to quit.")

    fps, last = 0.0, time.time()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = time.time()
            lift, is_open, armed = pose.update(frame, now)
            status = ""
            if tags is not None:
                pos, status = tags.update(frame, now)

            h, w = frame.shape[:2]
            if pose.points:
                (sx, sy), (hx, hy), (wx, wy) = pose.points
                s, hp, wr = (int(sx * w), int(sy * h)), (int(hx * w), int(hy * h)), (int(wx * w), int(wy * h))
                cv2.line(frame, s, hp, (255, 200, 0), 2)
                cv2.circle(frame, wr, 18, (0, 255, 0) if is_open else (0, 140, 255), 3)
            draw_bar(frame, gate)

            text = "GATE OPEN" if is_open else ("armed (just open)" if armed else "gate closed")
            color = (80, 230, 80) if is_open else ((0, 200, 255) if armed else (170, 170, 170))
            cv2.putText(frame, text, (90, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            cv2.putText(frame, f"lift {gate.lift:.2f}" if gate.lift is not None else "lift --",
                        (90, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (240, 240, 240), 2)
            dt = now - last
            last = now
            fps = 0.9 * fps + 0.1 * (1.0 / dt if dt > 0 else 0)
            cv2.putText(frame, f"{fps:.0f} fps" + (f"   tags: {status}" if tags else ""),
                        (90, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2)
            cv2.imshow("Pose gate", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
