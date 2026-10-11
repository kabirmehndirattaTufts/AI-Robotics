"""
M1: your AprilTags -> a dot on a top-down court.

    python tools/tag_view.py

Two windows:
  Camera   what the camera sees, with detected tags outlined
  Court    top-down view. The front wall (the screen) is the thick line at the top.
           Walk toward / away from the screen and left / right; the dot should follow.
           Stepping to YOUR right should move the dot to the right.

Keys: q = quit.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

import config
from vision import TagTracker, open_camera

PX_PER_M = 200
MARGIN = 50
COURT_W = getattr(config, "COURT_WIDTH", 2.0)
COURT_D = getattr(config, "COURT_DEPTH", 3.0)
REACH = getattr(config, "REACH_RADIUS", 0.5)


def to_px(court_x, court_z):
    px = MARGIN + (court_x + COURT_W / 2.0) * PX_PER_M
    py = MARGIN + court_z * PX_PER_M
    return int(px), int(py)


def draw_court(position, status, tracker):
    W = int(COURT_W * PX_PER_M) + 2 * MARGIN
    H = int(COURT_D * PX_PER_M) + 2 * MARGIN + 60
    img = np.full((H, W, 3), 35, np.uint8)
    x0, y0 = to_px(-COURT_W / 2, 0)
    x1, y1 = to_px(COURT_W / 2, COURT_D)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), 5)                # front wall = the screen
    cv2.putText(img, "WALL / SCREEN", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    shown = position if position is not None else tracker.last_known
    if shown is not None:
        inside = abs(shown[0]) <= COURT_W / 2 and 0 <= shown[1] <= COURT_D
        color = {"both": (80, 220, 80), "left": (80, 200, 220), "right": (80, 200, 220),
                 "held": (60, 160, 255), "lost": (90, 90, 220)}[status]
        if not inside:
            color = (60, 60, 255)
        px, py = to_px(*shown)
        px = int(np.clip(px, 5, W - 5))
        py = int(np.clip(py, 5, H - 65))
        cv2.circle(img, (px, py), int(REACH * PX_PER_M), (70, 70, 70), 1)   # reach radius
        cv2.circle(img, (px, py), 9, color, -1)
        txt = f"x {shown[0]:+.2f} m   z {shown[1]:.2f} m"
    else:
        txt = "no position yet"
    cv2.putText(img, txt, (10, H - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    cv2.putText(img, f"tags: {status}   spacing {tracker.spacing * 100:.0f} cm",
                (10, H - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1)
    return img


def draw_camera(frame, tracker):
    for tag_id, pts in tracker.detections:
        p = pts.astype(int)
        cv2.polylines(frame, [p], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tag_id}", (p[0][0], p[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame


def main():
    tracker = TagTracker(
        tag_size_m=getattr(config, "TAG_SIZE_MM", 100) / 1000.0,
        left_id=getattr(config, "TAG_LEFT_ID", 0),
        right_id=getattr(config, "TAG_RIGHT_ID", 1),
        spacing_m=getattr(config, "TAG_SPACING_M", 0.30),
        hold_s=getattr(config, "TAG_HOLD_S", 0.3),
        distance_scale=getattr(config, "DISTANCE_SCALE", 1.0),
    )
    cap = open_camera(getattr(config, "CAMERA_INDEX", None))
    ok, frame = cap.read()
    print(f"Camera frame size: {frame.shape[1]} x {frame.shape[0]}   (1280 x 720 recommended)")
    print("Press q in a window to quit.")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera read failed.")
                break
            position, status = tracker.update(frame)
            cv2.imshow("Camera", draw_camera(frame, tracker))
            cv2.imshow("Court", draw_court(position, status, tracker))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
