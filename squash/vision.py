"""
Vision for the squash game.

TagTracker finds your two AprilTags (family 36h11) in a camera frame and turns
them into a player position on the court:

    court_x  meters left/right of the camera's center line. Positive = YOUR right.
    court_z  meters from the camera/screen (the front wall) toward you.

Tags: left_id is worn on your LEFT side, right_id on your RIGHT side.
A camera does not mirror: when you face it, your left shows up on the right
side of the image. This file takes care of that.

If only one tag is visible, your center is estimated from the known spacing
between the tags. The spacing is learned whenever both tags are seen.
If both vanish, the last position is held for hold_s seconds, then you are "lost".
"""
import math
import time

import cv2
import numpy as np


class TagTracker:
    def __init__(self, tag_size_m=0.10, left_id=0, right_id=1, spacing_m=0.30,
                 hold_s=0.3, smooth=0.4, distance_scale=1.0, detect_width=960,
                 adaptive_smooth=False, remember_offset=False, smooth_tau=0.0,
                 blur_tolerant=False):
        """
        tag_size_m      side of the BLACK square in meters (100 mm = 0.10)
        spacing_m       starting guess for the distance between your two tags
        hold_s          keep the last position this long when tags disappear
        smooth          0..1, higher = follows faster, lower = smoother
        distance_scale  multiply all distances by this (calibrate against a tape measure)
        """
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)
        params = cv2.aruco.DetectorParameters()
        if blur_tolerant:
            # defaults assume a sharp picture; these let smeared tags still decode (tested: ~9 px -> ~21 px of smear)
            params.adaptiveThreshWinSizeMax = 53
            params.polygonalApproxAccuracyRate = 0.05
            params.maxErroneousBitsInBorderRate = 0.6
            params.errorCorrectionRate = 0.9
            params.perspectiveRemoveIgnoredMarginPerCell = 0.25
            params.minOtsuStdDev = 3.0
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        h = tag_size_m / 2.0
        # corner order OpenCV returns: top-left, top-right, bottom-right, bottom-left
        self.obj = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)
        self.left_id = left_id
        self.right_id = right_id
        self.spacing = spacing_m
        self.hold_s = hold_s
        self.smooth = smooth
        self.distance_scale = distance_scale
        self.adaptive_smooth = adaptive_smooth   # light smoothing when moving fast, heavy when still
        self.remember_offset = remember_offset   # one-tag fallback uses the last measured center offset
        self.smooth_tau = smooth_tau             # seconds; >0 = smoothing by time instead of per frame
        self._last_update = None
        self.detect_width = detect_width   # find tags on a smaller copy of the frame (faster)

        self._off_l = None         # last center-minus-left-tag vector seen while both tags were visible
        self._off_r = None
        self.raw = None            # latest unsmoothed (court_x, court_z), for debugging
        self.pos = None            # smoothed (court_x, court_z)
        self.last_seen = -1e9
        self.last_known = None     # last good position, for display
        self.detections = []       # list of (id, corners) from the latest frame

    # --- camera model: uncalibrated guess (focal length = image width) -------
    @staticmethod
    def _camera_matrix(w, h):
        f = float(w)
        return np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1]], dtype=np.float64)

    def _pose(self, corners, K):
        pts = corners.reshape(4, 2).astype(np.float32)
        ok, _rvec, tvec = cv2.solvePnP(self.obj, pts, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        if not ok:
            return None
        t = tvec.flatten()
        return np.array([float(t[0]), float(t[1]), float(t[2])]) * self.distance_scale

    def update(self, frame, now=None):
        """
        Process one BGR frame. Returns (position, status):
          position  (court_x, court_z) in meters, or None if lost
          status    'both', 'left', 'right', 'held', or 'lost'
        """
        now = time.time() if now is None else now
        h, w = frame.shape[:2]
        K = self._camera_matrix(w, h)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame

        shrink = 1.0
        if self.detect_width and w > self.detect_width:
            shrink = self.detect_width / float(w)
            gray = cv2.resize(gray, (self.detect_width, int(round(h * shrink))), interpolation=cv2.INTER_AREA)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if shrink != 1.0 and ids is not None:
            corners = [c / shrink for c in corners]            # back to full-frame pixels
        poses = {}
        self.detections = []
        if ids is not None:
            for c, i in zip(corners, ids.flatten()):
                i = int(i)
                self.detections.append((i, c.reshape(4, 2)))
                if i in (self.left_id, self.right_id):
                    p = self._pose(c, K)
                    if p is not None:
                        poses[i] = p

        tl = poses.get(self.left_id)
        tr = poses.get(self.right_id)
        measured = None
        status = "lost"
        if tl is not None and tr is not None:
            cam_x = (tl[0] + tr[0]) / 2.0
            z = (tl[2] + tr[2]) / 2.0
            sp = float(np.linalg.norm(tl - tr))
            if 0.10 < sp < 0.80:                         # ignore nonsense readings
                self.spacing = 0.9 * self.spacing + 0.1 * sp
            measured, status = (cam_x, z), "both"
            self._off_l = (cam_x - tl[0], z - tl[2])           # remember where the center is relative to each tag
            self._off_r = (cam_x - tr[0], z - tr[2])
        elif tl is not None:
            # your left tag appears right of center in the image, so center is to its left (smaller x)
            if self.remember_offset and self._off_l is not None:
                measured = (tl[0] + self._off_l[0], tl[2] + self._off_l[1])
            else:
                measured = (tl[0] - self.spacing / 2.0, tl[2])
            status = "left"
        elif tr is not None:
            if self.remember_offset and self._off_r is not None:
                measured = (tr[0] + self._off_r[0], tr[2] + self._off_r[1])
            else:
                measured = (tr[0] + self.spacing / 2.0, tr[2])
            status = "right"

        if measured is not None:
            court = (float(-measured[0]), float(measured[1]))          # mirror x so stepping right moves the dot right
            self.raw = court
            if self.pos is None:
                self.pos = court
            else:
                # adaptive: heavy smoothing when you stand still, light when you move fast
                d = ((court[0] - self.pos[0]) ** 2 + (court[1] - self.pos[1]) ** 2) ** 0.5
                a = min(0.9, self.smooth * 0.6 + 1.5 * d) if self.adaptive_smooth else self.smooth
                if self.smooth_tau > 0 and self._last_update is not None:
                    a = 1.0 - math.exp(-max(now - self._last_update, 1e-3) / self.smooth_tau)
                self.pos = (self.pos[0] + a * (court[0] - self.pos[0]),
                            self.pos[1] + a * (court[1] - self.pos[1]))
            self.last_seen = now
            self._last_update = now
            self.last_known = self.pos
            return self.pos, status

        if self.pos is not None and now - self.last_seen <= self.hold_s:
            return self.pos, "held"
        self.pos = None
        return None, "lost"


BACKENDS = {"MSMF": cv2.CAP_MSMF, "DSHOW": cv2.CAP_DSHOW, "ANY": cv2.CAP_ANY}


def open_camera(index=None, width=1280, height=720, backend="MSMF", mjpg=True):
    """Open a camera (Windows). If index is None and several exist, ask which one."""
    if index is None:
        found = []
        for i in range(5):
            cap = cv2.VideoCapture(i, BACKENDS[backend])
            if cap is not None and cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    found.append(i)
                cap.release()
        if not found:
            raise RuntimeError("No cameras found.")
        if len(found) == 1:
            index = found[0]
            print(f"Using camera {index} (only one found).")
        else:
            while True:
                choice = input(f"Select camera index {found}: ").strip()
                if choice.isdigit() and int(choice) in found:
                    index = int(choice)
                    break
    cap = cv2.VideoCapture(index, BACKENDS[backend])
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {index}")
    # MJPG first: many webcams only reach 30 fps at 720p in MJPG
    if mjpg:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap