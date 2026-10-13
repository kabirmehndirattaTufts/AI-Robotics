"""
Which face of the paddle (the 3D printed enclosure) is the camera seeing: ORANGE or BLUE?

Method: a color mask. Take a box around your swing wrist (the pose tracker gives the wrist position),
convert it to HSV, keep the pixels inside the orange range and the pixels inside the blue range, and
compare how much of the box each one fills.

    pc = PaddleColor()
    face, fracs, box = pc.classify(frame_bgr, wrist_norm=(0.62, 0.41), now=time.time())
    face   'orange', 'blue' or None (not sure)
    fracs  {'orange': 0.12, 'blue': 0.00}   share of the box that matched each color
    pc.recent(now)   the last sure answer if it is newer than max_age seconds

The color ranges start from DEFAULTS and are overridden by color_calib.json, which tools/color_view.py
writes when you press o (orange) or b (blue) while holding that face to the camera.
"""
import json
import os
import time

import cv2
import numpy as np

CALIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "color_calib.json")

# OpenCV hue runs 0..179. Orange is about 8..22, blue about 95..130.
DEFAULTS = {
    "orange": {"h": [5, 22], "s_min": 150, "v_min": 100},
    "blue": {"h": [95, 130], "s_min": 120, "v_min": 50},
}
KERNEL = np.ones((3, 3), np.uint8)


def load_ranges(path=CALIB_PATH):
    ranges = json.loads(json.dumps(DEFAULTS))          # deep copy
    try:
        with open(path) as f:
            saved = json.load(f)
        for name in ranges:
            if name in saved:
                ranges[name].update(saved[name])
    except (OSError, ValueError):
        pass
    return ranges


class PaddleColor:
    def __init__(self, roi_frac=0.22, min_frac=0.04, dominance=1.5, calib_path=CALIB_PATH, ranges=None):
        """
        roi_frac   half-size of the box around the wrist, as a share of the frame WIDTH (0.22 = 282 px at 1280,
               so a 564 px square: big enough for a ping-pong-sized paddle held in the hand)
        min_frac   the winning color must fill at least this share of the box
        dominance  and must beat the other color by at least this factor
        """
        self.roi_frac = roi_frac
        self.min_frac = min_frac
        self.dominance = dominance
        self.calib_path = calib_path
        self.ranges = ranges or load_ranges(calib_path)
        self.skin = None            # (h, s, v) of a clicked skin spot; those colors are never counted as the paddle
        try:
            with open(calib_path) as f:
                sk = json.load(f).get("skin")
            if sk:
                self.skin = tuple(sk["sample"])
        except (OSError, ValueError, KeyError):
            pass
        self.min_blob_px = 300      # the paddle face has to be at least this many pixels
        self.last_face = None
        self.last_t = -1e9

    # --- geometry ------------------------------------------------------------------
    def box(self, frame_shape, wrist_norm):
        h, w = frame_shape[:2]
        half = max(20, int(self.roi_frac * w))
        cx, cy = int(wrist_norm[0] * w), int(wrist_norm[1] * h)
        return max(0, cx - half), max(0, cy - half), min(w, cx + half), min(h, cy + half)

    # --- masks -----------------------------------------------------------------------
    def raw_mask(self, hsv, name):
        """Every pixel in the color range, minus pixels that look like the clicked skin color."""
        r = self.ranges[name]
        lo = np.array([r["h"][0], r["s_min"], r["v_min"]], np.uint8)
        hi = np.array([r["h"][1], 255, 255], np.uint8)
        m = cv2.inRange(hsv, lo, hi)
        if self.skin is not None and name == "orange":
            h, s, v = self.skin
            slo = np.array([max(0, h - 6), max(0, s - 40), max(0, v - 60)], np.uint8)
            shi = np.array([min(179, h + 6), min(255, s + 40), min(255, v + 60)], np.uint8)
            m[cv2.inRange(hsv, slo, shi) > 0] = 0
        return m

    def mask(self, hsv, name, min_px=None):
        """Only the single biggest blob of the color (the paddle face); small patches elsewhere are dropped."""
        m = cv2.morphologyEx(self.raw_mask(hsv, name), cv2.MORPH_OPEN, KERNEL)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))   # fill glare holes in the face
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n <= 1:
            return np.zeros_like(m)
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[biggest, cv2.CC_STAT_AREA] < (self.min_blob_px if min_px is None else min_px):
            return np.zeros_like(m)
        return (labels == biggest).astype(np.uint8) * 255

    def set_skin(self, frame_bgr, x, y, radius=10, save=True):
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = max(0, x - radius), max(0, y - radius), min(w, x + radius + 1), min(h, y + radius + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        self.skin = tuple(int(np.median(hsv[:, i])) for i in range(3))
        if save:
            try:
                with open(self.calib_path) as f:
                    saved = json.load(f)
            except (OSError, ValueError):
                saved = {}
            saved["skin"] = {"sample": list(self.skin)}
            with open(self.calib_path, "w") as f:
                json.dump(saved, f, indent=2)
        return self.skin

    def classify(self, frame_bgr, wrist_norm, now=None):
        now = time.time() if now is None else now
        x0, y0, x1, y1 = self.box(frame_bgr.shape, wrist_norm)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None, {"orange": 0.0, "blue": 0.0}, (x0, y0, x1, y1)
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        area = float(hsv.shape[0] * hsv.shape[1])
        fracs = {n: float(np.count_nonzero(self.mask(hsv, n))) / area for n in ("orange", "blue")}
        face = None
        a, b = fracs["orange"], fracs["blue"]
        if a >= self.min_frac and a >= self.dominance * b:
            face = "orange"
        elif b >= self.min_frac and b >= self.dominance * a:
            face = "blue"
        if face is not None:
            self.last_face, self.last_t = face, now
        return face, fracs, (x0, y0, x1, y1)

    def classify_frame(self, frame_bgr, width=320, min_area_frac=0.01):
        """
        Whole-picture version used by the game: shrink the frame, find the biggest orange blob and the
        biggest blue blob, and answer 'orange', 'blue' or None (neither is big enough, or they tie).
        Returns (face, {'orange': area_frac, 'blue': area_frac}).
        """
        h, w = frame_bgr.shape[:2]
        if w > width:
            frame_bgr = cv2.resize(frame_bgr, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        area = float(hsv.shape[0] * hsv.shape[1])
        small = max(20, int(0.002 * area))
        fr = {n: float(np.count_nonzero(self.mask(hsv, n, min_px=small))) / area for n in ("orange", "blue")}
        a, b = fr["orange"], fr["blue"]
        face = None
        if a >= min_area_frac and a >= self.dominance * b:
            face = "orange"
        elif b >= min_area_frac and b >= self.dominance * a:
            face = "blue"
        return face, fr

    def recent(self, now, max_age=0.5):
        """The last sure answer, if it is fresh enough (a swing is only noticed a moment after it starts)."""
        return self.last_face if now - self.last_t <= max_age else None

    # --- calibration -------------------------------------------------------------------
    def calibrate(self, frame_bgr, wrist_norm, name):
        """Auto mode: hold that face inside the wrist square, then call this. Returns the new range or None."""
        return self.calibrate_box(frame_bgr, self.box(frame_bgr.shape, wrist_norm), name, drawn=False)

    def calibrate_box(self, frame_bgr, box, name, drawn=True):
        """
        Learn a color from the pixels inside `box` (x0, y0, x1, y1).
        drawn=True : you drew the box around just the colored face, so every colorful pixel counts.
        drawn=False: the box is the whole wrist square (skin, background...), so only the most
                     saturated third is used.
        Returns the new range, or None if the box has too little color.
        """
        x0, y0, x1, y1 = box
        if x1 - x0 < 8 or y1 - y0 < 8:
            return None
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hsv = hsv[(hsv[:, 1] >= 60 if drawn else hsv[:, 1] >= 90) & (hsv[:, 2] >= 40)]   # colorful pixels only
        if len(hsv) < 200:
            return None
        if not drawn:
            order = np.argsort(hsv[:, 1])[::-1][: max(100, len(hsv) // 3)]
            hsv = hsv[order]
        lo_h, hi_h = np.percentile(hsv[:, 0], [5, 95])
        lo_s = np.percentile(hsv[:, 1], 5)
        lo_v = np.percentile(hsv[:, 2], 5)
        new = {"h": [max(0, int(lo_h) - 4), min(179, int(hi_h) + 4)],
               "s_min": max(60, int(0.8 * lo_s)), "v_min": max(30, int(0.8 * lo_v))}
        self.ranges[name] = new
        try:
            with open(self.calib_path) as f:
                saved = json.load(f)
        except (OSError, ValueError):
            saved = {}
        saved[name] = new
        with open(self.calib_path, "w") as f:
            json.dump(saved, f, indent=2)
        return new

    def calibrate_click(self, frame_bgr, x, y, name, radius=10, hue_buffer=8, sv_keep=0.6, save=True):
        """
        Learn a color from ONE click: take the middle (median) color of a small patch around (x, y),
        then allow hue +-hue_buffer and anything at least sv_keep (60%) as saturated / bright as the click.
        Returns (new_range, sample_hsv) or None if the click is on a gray / dark spot.
        """
        h, w = frame_bgr.shape[:2]
        x0, y0, x1, y1 = max(0, x - radius), max(0, y - radius), min(w, x + radius + 1), min(h, y + radius + 1)
        if x1 - x0 < 3 or y1 - y0 < 3:
            return None
        hsv = cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV).reshape(-1, 3)
        hh, ss, vv = (int(np.median(hsv[:, i])) for i in range(3))
        if ss < 50 or vv < 40:
            return None
        sample = (hh, ss, vv)
        return self.set_from_sample(sample, name, hue_buffer, sv_keep, save), sample

    def set_from_sample(self, sample, name, hue_buffer=8, sv_keep=0.6, save=True):
        hh, ss, vv = sample
        new = {"h": [max(0, hh - hue_buffer), min(179, hh + hue_buffer)],
               "s_min": max(50, int(sv_keep * ss)), "v_min": max(30, int(sv_keep * vv))}
        self.ranges[name] = new
        if save:
            try:
                with open(self.calib_path) as f:
                    saved = json.load(f)
            except (OSError, ValueError):
                saved = {}
            saved[name] = new
            with open(self.calib_path, "w") as f:
                json.dump(saved, f, indent=2)
        return new

    def reset(self):
        self.skin = None
        self.ranges = json.loads(json.dumps(DEFAULTS))
        try:
            os.remove(self.calib_path)
        except OSError:
            pass