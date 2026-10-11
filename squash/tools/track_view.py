"""
Test view for tag tracking + pose assist (no game, no motor).

    python tools/track_view.py

Court window: the big dot is the position the game would use.
   green  = both tags seen        cyan = one tag seen
   orange = POSE (tags blurred out; side-to-side from your body; needs POSE_ASSIST = True)
   yellow = held (tags just lost) red = lost
   small red dot = raw, unsmoothed tag position
Camera window: tag outlines, your shoulder line and swing wrist.
Bottom of the court window: "tags seen X%" = share of camera frames in the last 3 s where at least one
of your tags was found. Compare this number between settings while you move the same way each time.

Switches (config.py):  TAG_BLUR_TOLERANT, POSE_ASSIST, POSE_ASSIST_MAX_S
Keys: q = quit
"""
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

import config
from pose import ArmGate, PoseTracker
from vision import TagTracker, open_camera
from vision_thread import VisionThread

PX_PER_M = 200
MARGIN = 50
COURT_W = getattr(config, "COURT_WIDTH", 2.0)
COURT_D = getattr(config, "COURT_DEPTH", 3.0)
COLORS = {"both": (80, 220, 80), "left": (220, 200, 80), "right": (220, 200, 80),
          "pose": (0, 150, 255), "held": (0, 220, 255), "lost": (60, 60, 220)}


def to_px(x, z):
    return int(MARGIN + (x + COURT_W / 2.0) * PX_PER_M), int(MARGIN + z * PX_PER_M)


def draw_court(pos, status, raw, seen_pct, fps, assist):
    W = int(COURT_W * PX_PER_M) + 2 * MARGIN
    H = int(COURT_D * PX_PER_M) + 2 * MARGIN + 80
    img = np.full((H, W, 3), 35, np.uint8)
    x0, y0 = to_px(-COURT_W / 2, 0)
    x1, y1 = to_px(COURT_W / 2, COURT_D)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), 5)
    if raw is not None:
        rx, ry = to_px(max(-COURT_W / 2, min(COURT_W / 2, raw[0])), max(0.0, min(COURT_D, raw[1])))
        cv2.circle(img, (rx, ry), 4, (60, 60, 255), -1)
    if pos is not None:
        px, py = to_px(max(-COURT_W / 2, min(COURT_W / 2, pos[0])), max(0.0, min(COURT_D, pos[1])))
        cv2.circle(img, (px, py), 10, COLORS.get(status, (200, 200, 200)), -1)
    cv2.putText(img, f"source: {status}", (10, H - 62), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS.get(status, (230, 230, 230)), 2)
    cv2.putText(img, f"tags seen {seen_pct:3.0f}%   {fps:.0f} cam fps", (10, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(img, f"pose assist {'ON' if assist else 'off'}   blur tolerant {'ON' if getattr(config, 'TAG_BLUR_TOLERANT', False) else 'off'}",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


def draw_camera(frame, s):
    for tag_id, pts in s["detections"]:
        p = pts.astype(int)
        cv2.polylines(frame, [p], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tag_id}", (p[0][0], p[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if s["points"] is not None:
        h, w = frame.shape[:2]
        sh, hp, wr = (tuple(int(v * k) for v, k in zip(pt, (w, h))) for pt in s["points"])
        cv2.line(frame, sh, hp, (200, 200, 200), 2)
        cv2.circle(frame, wr, 20, (80, 220, 80), 3)
    return frame


def main():
    tracker = TagTracker(
        tag_size_m=getattr(config, "TAG_SIZE_MM", 100) / 1000.0,
        left_id=getattr(config, "TAG_LEFT_ID", 0),
        right_id=getattr(config, "TAG_RIGHT_ID", 1),
        spacing_m=getattr(config, "TAG_SPACING_M", 0.30),
        hold_s=getattr(config, "TAG_HOLD_S", 0.3),
        distance_scale=getattr(config, "DISTANCE_SCALE", 1.0),
        detect_width=getattr(config, "TAG_DETECT_WIDTH", 960),
        smooth_tau=getattr(config, "TAG_SMOOTH_TAU_S", 0.0),
        blur_tolerant=getattr(config, "TAG_BLUR_TOLERANT", False),
    )
    gate = ArmGate(open_at=getattr(config, "GATE_OPEN_AT", 0.30),
                   close_at=getattr(config, "GATE_CLOSE_AT", 0.20),
                   remember_s=getattr(config, "GATE_REMEMBER_S", 0.6))
    pose = PoseTracker(hand=getattr(config, "SWING_HAND", "right"),
                       swap_lr=getattr(config, "POSE_SWAP_LR", False), gate=gate)
    cap = open_camera(getattr(config, "CAMERA_INDEX", None),
                      backend=getattr(config, "CAMERA_BACKEND", "MSMF"),
                      mjpg=getattr(config, "CAMERA_MJPG", True))
    assist = getattr(config, "POSE_ASSIST", False)
    vision = VisionThread(cap, tracker, pose, pose_assist=assist,
                          assist_max_s=getattr(config, "POSE_ASSIST_MAX_S", 1.5))
    vision.start()
    print("Running. q in a window to quit.")

    seen = deque()           # (time, 1 if a tag was found in that camera frame else 0)
    last_id = -1
    try:
        while True:
            s = vision.snapshot()
            if s["error"]:
                print(s["error"])
                break
            now = time.time()
            if s["frame_id"] != last_id:
                last_id = s["frame_id"]
                seen.append((now, 1 if any(i in (tracker.left_id, tracker.right_id) for i, _ in s["detections"]) else 0))
            while seen and now - seen[0][0] > 3.0:
                seen.popleft()
            pct = 100.0 * sum(v for _, v in seen) / len(seen) if seen else 0.0

            cv2.imshow("Court", draw_court(s["position"], s["tag_status"], s["raw"], pct, vision.cam_fps, assist))
            if s["frame"] is not None:
                cv2.imshow("Camera", draw_camera(s["frame"].copy(), s))
            if cv2.waitKey(15) & 0xFF == ord("q"):
                break
    finally:
        vision.stop()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
