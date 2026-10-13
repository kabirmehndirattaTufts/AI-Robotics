"""
Runs the camera, the AprilTag tracker and the pose model in background threads,
so the game can draw and move the ball at full speed no matter how slow vision is.

    vt = VisionThread(cap, tracker, pose, pose_assist=True)
    vt.start()
    s = vt.snapshot()     # call every game frame; cheap
    s["position"]         (court_x, court_z) or None
    s["tag_status"]       'both' / 'left' / 'right' / 'pose' / 'held' / 'lost'
                          ('pose' = tags are blurred out, side-to-side comes from your body in the pose image)
    s["raw"]              latest unsmoothed tag position (debugging)
    s["lift"]             arm height number or None
    s["frame"]            latest camera frame (for the optional preview)
    ...
    vt.stop()

Thread 1 (camera):  read a frame, find the tags.
Thread 2 (pose):    take the newest frame, run the pose model on it.
They run at their own speeds; the pose thread simply skips frames it has no time for.

POSE ASSIST (side-to-side only)
  While the tags are visible we learn, continuously, how your body position in the pose image
  lines up with the tag position:
        x_from_pose = -(body_center_x - 0.5) * distance        (same camera model the tags use)
        bias        = tag_x - x_from_pose                      (averaged over time)
  When the tags are lost (blur), we use  x_from_pose + bias  and hold the distance from the last
  tag reading. When the tags come back, the difference fades out over HANDOFF_S seconds so the dot
  doesn't pop. Distance toward/away from the camera is NOT estimated from pose.
"""
import math
import threading
import time
from collections import deque

BODY_FRESH_S = 0.30     # pose reading must be newer than this to be trusted
HANDOFF_S = 0.25        # fade time when the tags return


class VisionThread:
    def __init__(self, cap, tracker, pose, pose_assist=False, assist_max_s=1.5, paddle=None):
        self.cap = cap
        self.tracker = tracker
        self.pose = pose
        self.pose_assist = pose_assist
        self.assist_max_s = assist_max_s
        self.paddle = paddle           # PaddleColor or None (color mode off)
        self._faces = deque(maxlen=120)  # (frame_time, 'orange'|'blue') for frames where the color was clear
        self._serve_open = False
        self._serve_lift = None
        self._serve_wrist = None
        self._last_face = None           # sticky: the last clear color, kept until a different one shows up
        self._frame_t = 0.0
        self.color_ms = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []

        self._frame = None
        self._frame_id = 0
        self._position = None
        self._raw = None
        self._tag_status = "lost"
        self._detections = []
        self._points = None
        self._body = None            # (center_x, width, time)
        self._fix_t = -1e9           # last time a tag actually measured something
        self._fix_pos = None
        self.error = None

        # pose-assist state (only touched from snapshot(), i.e. the game thread)
        self._bias = None
        self._last_out = None
        self._last_source = None
        self._handoff = None         # (dx, dz, start_time)

        self.cam_fps = 0.0
        self.tag_ms = 0.0
        self.pose_hz = 0.0
        self.pose_ms = 0.0

    def start(self):
        loops = [self._camera_loop, self._pose_loop] + ([self._color_loop] if self.paddle is not None else [])
        for target in loops:
            t = threading.Thread(target=target, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._stop.set()
        for t in self._threads:
            t.join(timeout=2.0)

    # --- what the game reads ---------------------------------------------------------
    def snapshot(self):
        now = time.time()
        gate = self.pose.gate
        with self._lock:
            pos, status = self._position, self._tag_status
            body, fix_t, fix_pos = self._body, self._fix_t, self._fix_pos
            out = {
                "raw": self._raw,
                "detections": list(self._detections),
                "points": self._points,
                "frame": self._frame,
                "frame_id": self._frame_id,
                "lift": gate.lift,
                "gate_open": gate.open,
                "armed": gate.armed(now),
                "reason": gate.reason,
                "error": self.error,
                "face": self._last_face,
                "serve_open": self._serve_open,
                "serve_lift": self._serve_lift,
                "serve_wrist": self._serve_wrist,
            }
        if self.pose_assist:
            pos, status = self._assist(now, pos, status, body, fix_t, fix_pos)
        out["position"] = pos
        out["tag_status"] = status
        return out

    def face_before(self, t, max_age=None):
        """The most recent clear paddle color seen at or before time t (and, if max_age is set, not older than that), else None."""
        with self._lock:
            for ft, face in reversed(self._faces):
                if ft <= t:
                    return face if (max_age is None or t - ft <= max_age) else None
        return None

    # --- pose assist -------------------------------------------------------------------
    def _assist(self, now, pos, status, body, fix_t, fix_pos):
        body_ok = body is not None and now - body[2] <= BODY_FRESH_S
        if status in ("both", "left", "right") and pos is not None:
            if body_ok:
                x_pose = -(body[0] - 0.5) * pos[1]
                b = pos[0] - x_pose
                self._bias = b if self._bias is None else self._bias + 0.1 * (b - self._bias)
            if self._last_source == "pose" and self._last_out is not None:
                self._handoff = (self._last_out[0] - pos[0], self._last_out[1] - pos[1], now)
            out = pos
            if self._handoff is not None:
                f = math.exp(-(now - self._handoff[2]) / HANDOFF_S)
                if f < 0.02:
                    self._handoff = None
                else:
                    out = (pos[0] + self._handoff[0] * f, pos[1] + self._handoff[1] * f)
            source = status
        elif (self._bias is not None and body_ok and fix_pos is not None
              and now - fix_t <= self.assist_max_s):
            z = fix_pos[1]
            out = (-(body[0] - 0.5) * z + self._bias, z)
            source = "pose"
            self._handoff = None
        else:
            out, source = pos, status
            self._handoff = None
        self._last_out = out
        self._last_source = "pose" if source == "pose" else "tag"
        return out, source

    # --- thread 1 -----------------------------------------------------------------
    def _camera_loop(self):
        last = time.time()
        while not self._stop.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.error = "Camera read failed."
                return
            now = time.time()
            t0 = time.perf_counter()
            position, status = self.tracker.update(frame, now)
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._frame = frame
                self._frame_id += 1
                self._frame_t = now
                self._position = position
                self._raw = self.tracker.raw
                self._tag_status = status
                self._detections = list(self.tracker.detections)
                if status in ("both", "left", "right") and position is not None:
                    self._fix_t = now
                    self._fix_pos = position
            self.tag_ms = 0.9 * self.tag_ms + 0.1 * ms
            self.cam_fps = 0.9 * self.cam_fps + 0.1 / max(now - last, 1e-3)
            last = now

    # --- thread 2 -----------------------------------------------------------------
    def _pose_loop(self):
        seen = -1
        last = time.time()
        while not self._stop.is_set():
            with self._lock:
                frame, fid = self._frame, self._frame_id
            if frame is None or fid == seen:
                time.sleep(0.003)
                continue
            seen = fid
            now = time.time()
            t0 = time.perf_counter()
            try:
                self.pose.update(frame, now)
            except Exception as e:
                self.error = f"pose error: {e}"
                return
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                self._points = self.pose.points
                sg = self.pose.serve_gate
                self._serve_open = bool(sg.open) if sg is not None else False
                self._serve_lift = sg.lift if sg is not None else None
                self._serve_wrist = self.pose.serve_wrist
                b = self.pose.body
                self._body = (b[0], b[1], now) if b is not None else None
            self.pose_ms = 0.9 * self.pose_ms + 0.1 * ms
            self.pose_hz = 0.9 * self.pose_hz + 0.1 / max(now - last, 1e-3)
            last = now

    # --- thread 3 (only with color mode) ------------------------------------------------
    def _color_loop(self):
        seen = -1
        while not self._stop.is_set():
            with self._lock:
                frame, fid, ft = self._frame, self._frame_id, self._frame_t
            if frame is None or fid == seen:
                time.sleep(0.003)
                continue
            seen = fid
            t0 = time.perf_counter()
            try:
                face, _ = self.paddle.classify_frame(frame)
            except Exception as e:
                self.error = f"color error: {e}"
                return
            ms = (time.perf_counter() - t0) * 1000
            with self._lock:
                if face is not None:
                    self._last_face = face
                    self._faces.append((ft, face))
            self.color_ms = 0.9 * self.color_ms + 0.1 * ms