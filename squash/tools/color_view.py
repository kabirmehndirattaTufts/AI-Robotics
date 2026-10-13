"""
Teach the computer your two paddle colors by clicking on them, then see both masks at once.

Run:  python tools/color_view.py

Step 1: hold the ORANGE face to the camera, CLICK on the orange in the picture.
Step 2: hold the BLUE face to the camera, CLICK on the blue.
Step 3: the right window shows both masks together (orange pixels painted orange, blue pixels painted blue; only the biggest blob of each counts, gray = ignored).

Keys:  = / -  widen / narrow the hue buffer (both colors)     r  start over     q  quit
Clicking again at any time re-picks the color of the current stage; after step 3, press 1 or 2
and then click to re-pick orange or blue.
The result is saved to color_calib.json, which the game will read later.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import config
from paddle import PaddleColor
from vision import open_camera

ORANGE_BGR, BLUE_BGR = (0, 140, 255), (255, 120, 0)
state = {"stage": "orange", "click": None, "buffer": 8}
samples = {}


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click"] = (x, y)


def put(img, text, y, scale=0.7, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.rectangle(img, (8, y - th - 6), (16 + tw, y + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


def main():
    pc = PaddleColor()
    pc.reset()                                   # start fresh every time this tool runs
    cap = open_camera(getattr(config, "CAMERA_INDEX", None), backend=getattr(config, "CAMERA_BACKEND", "MSMF"),
                      mjpg=getattr(config, "CAMERA_MJPG", True))
    cv2.namedWindow("Camera")
    cv2.setMouseCallback("Camera", on_mouse)
    marks = {}
    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cv2.flip(frame, 1)               # mirror, like a mirror (clicks use this same picture)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        c = state["click"]
        state["click"] = None
        if c is not None and state["stage"] == "pick_skin":
            sk = pc.set_skin(frame, c[0], c[1])
            marks["skin"] = c
            print(f"skin: clicked (h,s,v) = {sk}; those colors are now ignored for orange")
            state["stage"] = "both"
        elif c is not None and state["stage"] in ("orange", "blue", "pick_orange", "pick_blue"):
            name = "orange" if state["stage"] in ("orange", "pick_orange") else "blue"
            res = pc.calibrate_click(frame, c[0], c[1], name, hue_buffer=state["buffer"])
            if res is None:
                print(f"That spot is gray or dark. Click on the colorful part of the {name} face.")
            else:
                _, samples[name] = res
                marks[name] = c
                print(f"{name}: clicked hue {samples[name][0]}, range {pc.ranges[name]}")
                state["stage"] = "blue" if name == "orange" and "blue" not in samples else "both"

        cam = frame.copy()
        for name, col in (("orange", ORANGE_BGR), ("blue", BLUE_BGR), ("skin", (180, 180, 180))):
            if name in marks:
                cv2.circle(cam, marks[name], 14, col, 3)
                cv2.circle(cam, marks[name], 16, (255, 255, 255), 1)

        if state["stage"] == "orange":
            put(cam, "STEP 1: hold the ORANGE face up, then CLICK on the orange", 34, 0.8, ORANGE_BGR)
        elif state["stage"] == "blue":
            put(cam, "STEP 2: hold the BLUE face up, then CLICK on the blue", 34, 0.8, BLUE_BGR)
        elif state["stage"] == "pick_skin":
            put(cam, "click on your SKIN (hand / arm) so orange ignores it", 34, 0.8)
        elif state["stage"].startswith("pick"):
            put(cam, f"click on the {state['stage'][5:]} to re-pick it", 34, 0.8)
        else:
            put(cam, "STEP 3: masks on the right.  1/2 re-pick orange/blue,  s = click skin to ignore", 34, 0.8)
        put(cam, f"hue buffer +-{state['buffer']}  (= wider, - narrower)   r restart   q quit", frame.shape[0] - 14, 0.55)
        cv2.imshow("Camera", cam)

        if "orange" in samples or "blue" in samples:
            out = np.zeros_like(frame)
            lines = []
            for name, col in (("orange", ORANGE_BGR), ("blue", BLUE_BGR)):
                if name in samples:
                    raw = pc.raw_mask(hsv, name)
                    out[raw > 0] = (70, 70, 70)          # dark gray = right color but NOT the biggest blob (ignored)
                    m = pc.mask(hsv, name)
                    out[m > 0] = col
                    lines.append(f"{name} {100.0 * np.count_nonzero(m) / m.size:.1f}%")
            put(out, "   ".join(lines), 34, 0.8)
            cv2.imshow("Masks (orange + blue together)", out)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("r"):
            pc.reset(); samples.clear(); marks.clear()
            state["stage"] = "orange"
        if k == ord("1") and state["stage"] == "both":
            state["stage"] = "pick_orange"
        if k == ord("s") and state["stage"] == "both":
            state["stage"] = "pick_skin"
        if k == ord("2") and state["stage"] == "both":
            state["stage"] = "pick_blue"
        if k in (ord("="), ord("+"), ord("-")):
            state["buffer"] = max(2, min(30, state["buffer"] + (2 if k != ord("-") else -2)))
            for name, s in samples.items():
                pc.set_from_sample(s, name, state["buffer"])
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()