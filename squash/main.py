"""
The squash game, with everything joined up (no MQTT / AprilTag levels yet).

    python main.py              full game: LEGO motor + camera
    python main.py --cam        also open the camera window (tags + wrist ring)
    python main.py --no-motor   no LEGO motor: k = soft swing, space = medium, l = hard

A swing counts as a HIT when all three agree:
    1. the IMU says you swung (peak over SWING_THRESHOLD)
    2. the pose gate says your swing arm is raised (armed)
    3. the ball was inside your reach circle (your two AprilTags set where you are)

Keys (click a window first):  n = serve   1/2/3 = level   q = quit
"""
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

import config
from game import SquashGame
from haptics import Haptics
from pose import ArmGate, PoseTracker
from swing import SwingDetector
from vision import TagTracker, open_camera
from coach import Coach, draw_coach
from scoreboard import ScorePublisher, draw_panel
from vision_thread import VisionThread

NO_MOTOR = "--no-motor" in sys.argv
SHOW_CAM = "--cam" in sys.argv
PROFILE = "--profile" in sys.argv

W_M = getattr(config, "COURT_WIDTH", 2.0)
D_M = getattr(config, "COURT_DEPTH", 3.0)
REACH = getattr(config, "REACH_RADIUS", 0.5)
PX_PER_M = 220
CAMERA_VIEW_SCALE = getattr(config, "CAMERA_VIEW_SCALE", 0.5)   # size of the --cam window (1.0 = full size)
COURT_VIEW_SCALE = getattr(config, "COURT_VIEW_SCALE", 1.0)
COACH_ENABLED = getattr(config, "COACH_ENABLED", True)
COACH_VIEW_SCALE = getattr(config, "COACH_VIEW_SCALE", 0.75)
MQTT_ENABLED = getattr(config, "MQTT_ENABLED", True)
MQTT_HEARTBEAT_S = getattr(config, "MQTT_HEARTBEAT_S", 2.0)
SERVE_GESTURE = getattr(config, "SERVE_GESTURE", True)
SERVE_HOLD_S = getattr(config, "SERVE_HOLD_S", 0.4)
SERVE_COOLDOWN_S = getattr(config, "SERVE_COOLDOWN_S", 1.0)
COLOR_MAX_AGE_S = getattr(config, "COLOR_MAX_AGE_S", None)
COLOR_MODE = getattr(config, "COLOR_MODE", False)
BALL_BGR = {"orange": (0, 140, 255), "blue": (255, 120, 0)}
MARGIN = 50
HUD_H = 185
FOLLOW_TAU_S = getattr(config, "FOLLOW_TAU_S", 0.0)   # 0 = off. Otherwise: smaller = follows faster, larger = smoother


def to_px(x, z):
    return int(MARGIN + (x + W_M / 2) * PX_PER_M), int(MARGIN + z * PX_PER_M)


def draw_court(game, tag_status, armed, lift, msg, msg_age, fps, cam_fps=0.0, raw=None, face=None, serve_hint=None, mqtt_on=False):
    W = int(W_M * PX_PER_M) + 2 * MARGIN
    H = int(D_M * PX_PER_M) + MARGIN + HUD_H
    img = np.full((H, W, 3), 35, np.uint8)
    x0, y0 = to_px(-W_M / 2, 0)
    x1, y1 = to_px(W_M / 2, D_M)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), 5)
    cv2.line(img, (x0, y1), (x1, y1), (60, 60, 200), 2)                  # back line = miss

    px, py = to_px(*game.player)
    # the ring is always the color of the paddle face seen last. Thick and bright = arm raised, thin and dim = arm down.
    if COLOR_MODE and face in BALL_BGR:
        base = BALL_BGR[face]
    else:
        base = (80, 220, 80)
    if armed:
        ring, thick = base, 4
    else:
        ring, thick = tuple(int(0.45 * c + 0.55 * 110) for c in base), 1
    cv2.circle(img, (px, py), int(REACH * PX_PER_M), ring, thick, cv2.LINE_AA)
    cv2.circle(img, (px, py), 9, (230, 230, 230), -1)
    if raw is not None:                                                   # unsmoothed tag position (--cam only)
        rx, ry = to_px(max(-W_M / 2, min(W_M / 2, raw[0])), max(0.0, min(D_M, raw[1])))
        cv2.circle(img, (rx, ry), 5, (60, 60, 255), -1)

    if game.state == "playing":
        bx, by = to_px(*game.ball)
        wait = game.ready_in(time.time())
        cv2.circle(img, (bx, by), 11, (120, 120, 120) if wait > 0 else (0, 200, 255) if not game.color_mode else BALL_BGR[game.ball_color], -1)
        if wait > 0:
            cv2.putText(img, f"get ready {wait:.1f}", (x0 + 10, y0 + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)

    # --- scoreboard under the court ---
    ty = y1 + 26
    for x, label, value, col in ((14, "POINTS", f"{game.score:.0f}", (255, 255, 255)),
                                 (200, "STREAK", f"{game.streak}", (255, 255, 255)),
                                 (350, "RECORD STREAK", f"{game.best}", (255, 220, 120))):
        cv2.putText(img, label, (x, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(img, value, (x, ty + 36), cv2.FONT_HERSHEY_SIMPLEX, 1.2, col, 2, cv2.LINE_AA)
    if mqtt_on:
        cv2.putText(img, "sent to MQTT", (350, ty + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 220, 120), 1, cv2.LINE_AA)
    if game.color_mode:
        cy = ty + 78
        cv2.putText(img, "HIT WITH", (14, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.circle(img, (100, cy), 11, BALL_BGR[game.ball_color], -1)
        cv2.putText(img, "PADDLE SEES", (200, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 150, 150), 1, cv2.LINE_AA)
        if face in BALL_BGR:
            cv2.circle(img, (312, cy), 11, BALL_BGR[face], -1)
            cv2.putText(img, "match" if face == game.ball_color else "WRONG", (332, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (90, 220, 90) if face == game.ball_color else (70, 70, 235), 1, cv2.LINE_AA)
        else:
            cv2.circle(img, (312, cy), 11, (110, 110, 110), 2)
    line = None
    if msg and msg_age < 1.5:
        line, col = msg, (90, 220, 90)
    elif game.state == "waiting":
        line, col = serve_hint or "press n to serve", (0, 220, 255)
    if line:
        words, rows, cur = line.split(), [], ""
        for wd in words:                                                # wrap to the court width, two lines at most
            if len(cur) + len(wd) + 1 > 60 and cur:
                rows.append(cur)
                cur = wd
            else:
                cur = (cur + " " + wd).strip()
        rows.append(cur)
        for i, r in enumerate(rows[:2]):
            cv2.putText(img, r, (14, ty + 108 + 19 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    lift_txt = "no pose" if lift is None else f"{lift:.2f}"
    cv2.putText(img, f"best {game.best_score:.0f} pts   misses {game.misses}   tags {tag_status}   arm {lift_txt} {'ARMED' if armed else 'down'}   "
                     f"{fps:.0f} fps (cam {cam_fps:.0f})",
                (14, ty + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 120, 120), 1, cv2.LINE_AA)
    return img


def draw_camera(frame, s, pose):
    for tag_id, pts in s["detections"]:
        p = pts.astype(int)
        cv2.polylines(frame, [p], True, (0, 255, 0), 2)
        cv2.putText(frame, f"ID {tag_id}", (p[0][0], p[0][1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if s["points"] is not None:
        h, w = frame.shape[:2]
        sh, hp, wr = (tuple(int(v * k) for v, k in zip(pt, (w, h))) for pt in s["points"])
        cv2.line(frame, sh, hp, (200, 200, 200), 2)
        wc = (80, 220, 80) if s["gate_open"] else (0, 0, 255)
        if COLOR_MODE and s["gate_open"] and s.get("face") in BALL_BGR:
            wc = BALL_BGR[s["face"]]
        cv2.circle(frame, wr, 22, wc, 3)
    sw = s.get("serve_wrist")
    if SERVE_GESTURE and sw is not None:
        h, w = frame.shape[:2]
        cv2.circle(frame, (int(sw[0] * w), int(sw[1] * h)), 22, (255, 255, 0) if s["serve_open"] else (120, 120, 120), 3)
        cv2.putText(frame, "serve hand", (int(sw[0] * w) - 40, int(sw[1] * h) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    g = pose.gate
    cv2.putText(frame, "pose: not found" if pose.points is None else
                (f"lift {'--' if g.lift is None else format(g.lift, '.2f')}  reach {'--' if g.reach is None else format(g.reach, '.2f')}"
                 f"  mode {g.mode}  {'OPEN' if g.open else 'closed'}"),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    return frame


def main():
    if config.SWING_THRESHOLD is None:
        print("Set SWING_THRESHOLD in config.py first.")
        return

    # --- game + haptics + IMU -------------------------------------------------
    motor = None
    if not NO_MOTOR:
        from lelib import doubleMotor
        motor = doubleMotor()
        print("Connecting to the motor...")
        motor.connect(card_serial=config.CARD_SERIAL, card_color=config.CARD_COLOR)
        time.sleep(1.0)
        print("Connected.")

    haptics = Haptics(
        motor,
        speed_min=getattr(config, "HAPTIC_SPEED_MIN", 85),
        speed_max=getattr(config, "HAPTIC_SPEED_MAX", 100),
        pulse_s=getattr(config, "HAPTIC_PULSE_S", 0.25),
        cycles=getattr(config, "HAPTIC_CYCLES", 1),
        dry_run=NO_MOTOR,
    )
    detector = SwingDetector(
        motor, config.SWING_THRESHOLD, getattr(config, "SWING_MAX", 12500),
        collect_s=0.15, cooldown_s=0.6,
        gyro_deadband=getattr(config, "STROKE_GYRO_DEADBAND", 200),
    )
    if motor is not None:
        detector.start()

    state = {"msg": "", "t": 0.0, "prev_best": 0}
    coach = None
    if COACH_ENABLED:
        coach = Coach(tag_map=getattr(config, "COACH_TAGS", {2: "nice", 3: "mean"}),
                      names=getattr(config, "COACH_NAMES", None), images=getattr(config, "COACH_IMAGES", None),
                      keys=getattr(config, "COACH_KEYS", {"t": "nice", "s": "mean"}),
                      base_dir=os.path.dirname(os.path.abspath(__file__)))

    def say(text):
        state["msg"], state["t"] = text, time.time()
        print(text)

    def on_hit(power, speed, angle, streak):
        say(f"HIT  streak {streak}  +{game.last_points:.0f} pts (speed x{game.last_speed_factor:.2f}, "
            f"reaction {game.last_reaction_s:.2f}s x{game.last_reaction_factor:.2f})  total {game.score:.0f}  next: {game.ball_color.upper()}")
        detector.mute(haptics.duration + 0.4)       # the wiggle shakes the IMU
        haptics.hit(power)
        if coach is not None:
            if streak > state["prev_best"] and state["prev_best"] >= 3:
                coach.say("record", time.time())
            elif streak % 3 == 0:
                coach.say("hit", time.time())
        state["prev_best"] = max(state["prev_best"], streak)

    def on_miss(streak_before):
        say(f"MISS  streak was {streak_before}, rally score {game.best_score:.0f} best. press n to serve")
        if coach is not None:
            coach.say_miss(game.last_miss, time.time())

    def on_wrong_face(needed, seen):
        say(f"WRONG FACE: ball is {needed.upper()}, paddle was {seen.upper() if seen else 'not seen'}")

    game = SquashGame(
        court_w=W_M, court_d=D_M, reach=REACH,
        hit_window_s=getattr(config, "HIT_WINDOW_S", 0.2),
        level_speeds=config.LEVEL_SPEEDS, level=0,
        speed_step=getattr(config, "STREAK_SPEED_STEP", 0.05),
        max_ramp=getattr(config, "MAX_SPEED_RAMP", 1.8),
        serve_delay_s=getattr(config, "SERVE_DELAY_S", 1.0),
        points_base=getattr(config, "POINTS_BASE", 10.0),
        reaction_ref_s=getattr(config, "REACTION_REF_S", 1.0),
        on_hit=on_hit, on_miss=on_miss,
        color_mode=COLOR_MODE, on_wrong_face=on_wrong_face,
        color_max_repeat=getattr(config, "COLOR_MAX_REPEAT", 3),
    )

    # --- vision -----------------------------------------------------------------
    tracker = TagTracker(
        tag_size_m=getattr(config, "TAG_SIZE_MM", 100) / 1000.0,
        left_id=getattr(config, "TAG_LEFT_ID", 0),
        right_id=getattr(config, "TAG_RIGHT_ID", 1),
        spacing_m=getattr(config, "TAG_SPACING_M", 0.30),
        hold_s=getattr(config, "TAG_HOLD_S", 0.3),
        distance_scale=getattr(config, "DISTANCE_SCALE", 1.0),
        detect_width=getattr(config, "TAG_DETECT_WIDTH", 960),
        adaptive_smooth=getattr(config, "TAG_ADAPTIVE_SMOOTH", False),
        remember_offset=getattr(config, "TAG_REMEMBER_OFFSET", False),
        smooth_tau=getattr(config, "TAG_SMOOTH_TAU_S", 0.0),
        blur_tolerant=getattr(config, "TAG_BLUR_TOLERANT", False),
        true_spacing_m=getattr(config, "TAG_TRUE_SPACING_M", 0.0),
    )
    gate = ArmGate(
        open_at=getattr(config, "GATE_OPEN_AT", 0.30),
        close_at=getattr(config, "GATE_CLOSE_AT", 0.20),
        remember_s=getattr(config, "GATE_REMEMBER_S", 0.6),
        mode=getattr(config, "GATE_MODE", "height"),
        side_open_at=getattr(config, "GATE_SIDE_OPEN_AT", 0.80),
        side_close_at=getattr(config, "GATE_SIDE_CLOSE_AT", 0.60),
    )
    serve_gate = ArmGate(
        open_at=getattr(config, "SERVE_OPEN_AT", 1.3), close_at=getattr(config, "SERVE_CLOSE_AT", 1.1),
        remember_s=0.0, mode="height",
    ) if SERVE_GESTURE else None
    pose = PoseTracker(serve_gate=serve_gate,
        hand=getattr(config, "SWING_HAND", "right"),
        swap_lr=getattr(config, "POSE_SWAP_LR", False),
        gate=gate,
    )
    cap = open_camera(getattr(config, "CAMERA_INDEX", None),
                      backend=getattr(config, "CAMERA_BACKEND", "MSMF"),
                      mjpg=getattr(config, "CAMERA_MJPG", True))
    print("Running. Click the Squash window, then: n = serve, 1/2/3 = level, q = quit.")
    if NO_MOTOR:
        print("No-motor mode: k = soft swing, space = medium, l = hard.")

    paddle = None
    if COLOR_MODE:
        from paddle import PaddleColor
        paddle = PaddleColor()
        paddle.min_area_frac = getattr(config, "COLOR_MIN_AREA_FRAC", 0.003)
        print(f"Color mode on. Ranges: {paddle.ranges}  skin: {paddle.skin}")
    vision = VisionThread(cap, tracker, pose, paddle=paddle,
                          pose_assist=getattr(config, "POSE_ASSIST", False),
                          assist_max_s=getattr(config, "POSE_ASSIST_MAX_S", 1.5))
    vision.start()

    def start_serve(t):
        game.serve(t)
        if coach is not None:
            coach.say("serve", t)

    pub = None
    if MQTT_ENABLED:
        pub = ScorePublisher(getattr(config, "MQTT_BROKER", "test.mosquitto.org"), getattr(config, "MQTT_PORT", 1883),
                             config.MQTT_TOPIC, qos=0, retain=False)
        pub.start()
    last_vals, last_pub_t = {}, 0.0
    shown, shown_t = None, time.time()
    serve_hold_t = None            # when the serve hand went up
    serve_rearmed = False          # the serve hand must come down once before it can serve again
    waiting_since = time.time()
    windows_placed = False
    panel_placed = False
    coach_placed = False
    prev_state = "waiting"
    fps, last = 0.0, time.time()
    prof_t = time.time()
    try:
        while True:
            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
            last = now

            s = vision.snapshot()
            if s["error"]:
                print(s["error"])
                break
            if s["position"] is not None:
                target = (max(-W_M / 2, min(W_M / 2, s["position"][0])),
                          max(0.0, min(D_M, s["position"][1])))
                if shown is None:
                    shown = list(target)
                # glide toward the newest tag position every game frame (the camera only updates ~20 times a second)
                k = 1.0 - math.exp(-max(now - shown_t, 0.0) / FOLLOW_TAU_S) if FOLLOW_TAU_S > 0 else 1.0
                shown[0] += (target[0] - shown[0]) * k
                shown[1] += (target[1] - shown[1]) * k
                game.set_player(shown[0], shown[1])
            shown_t = now
            lift, armed = s["lift"], s["armed"]

            game.update(now)
            if coach is not None:
                coach.see_tags({tid for tid, _ in s["detections"]}, now, allowed=game.state == "waiting")

            # --- serve gesture: raise the serve hand above your head and hold it ---
            if game.state != prev_state:
                prev_state = game.state
                if game.state == "waiting":
                    waiting_since = now
            serve_hint = None
            if SERVE_GESTURE:
                if not s["serve_open"]:
                    serve_hold_t, serve_rearmed = None, True
                if game.state == "waiting":
                    if s["serve_open"] and serve_rearmed and now - waiting_since >= SERVE_COOLDOWN_S:
                        serve_hold_t = serve_hold_t if serve_hold_t is not None else now
                        if now - serve_hold_t >= SERVE_HOLD_S:
                            start_serve(now)
                            serve_hold_t, serve_rearmed = None, False
                    held = 0.0 if serve_hold_t is None else now - serve_hold_t
                    sl = s["serve_lift"]
                    serve_hint = (f"raise your LEFT hand high to serve (or n)   hand {'--' if sl is None else format(sl, '.2f')}"
                                  + (f"   hold {held:.1f}/{SERVE_HOLD_S:.1f}" if held > 0 else ""))

            for ev in detector.pop_events():
                tag = f"{ev['stroke']}, power {ev['power']:.2f}"
                if game.state != "playing" or game.ready_in(now) > 0 or game.serving:
                    continue                                        # swinging between rallies / during the serve delay is free
                if not armed:
                    game.note_event("arm_down")
                    say(f"swing ignored: arm down, lift {'none' if lift is None else format(lift, '.2f')} ({tag})")
                elif not game.swing(ev["power"], ev["t_onset"], vision.face_before(ev["t_onset"], COLOR_MAX_AGE_S) if COLOR_MODE else None):
                    say(f"WHIFF: ball not in reach ({tag})")

            court_img = draw_court(game, s["tag_status"], armed, lift,
                                   state["msg"], time.time() - state["t"], fps, vision.cam_fps,
                                   s["raw"] if SHOW_CAM else None, s["face"], serve_hint, pub is not None)
            if COURT_VIEW_SCALE != 1.0:
                court_img = cv2.resize(court_img, None, fx=COURT_VIEW_SCALE, fy=COURT_VIEW_SCALE, interpolation=cv2.INTER_AREA)
            cv2.imshow("Squash", court_img)
            if SHOW_CAM and s["frame"] is not None:
                cam_img = draw_camera(s["frame"].copy(), s, pose)
                if CAMERA_VIEW_SCALE != 1.0:
                    cam_img = cv2.resize(cam_img, None, fx=CAMERA_VIEW_SCALE, fy=CAMERA_VIEW_SCALE, interpolation=cv2.INTER_AREA)
                cv2.imshow("Camera", cam_img)
            if not windows_placed and (not SHOW_CAM or s["frame"] is not None):
                cv2.moveWindow("Squash", 0, 0)                       # court on the left, camera right next to it
                if SHOW_CAM:
                    cv2.moveWindow("Camera", court_img.shape[1] + 12, 0)
                windows_placed = True

            if coach is not None:
                cimg = draw_coach(coach, time.time())
                if COACH_VIEW_SCALE != 1.0:
                    cimg = cv2.resize(cimg, None, fx=COACH_VIEW_SCALE, fy=COACH_VIEW_SCALE, interpolation=cv2.INTER_AREA)
                cv2.imshow("Coach", cimg)
                if not coach_placed:
                    pos = getattr(config, "COACH_WINDOW_POS", None)
                    if pos is None:
                        try:                                            # bottom right corner of the screen (Windows)
                            import ctypes
                            sw, sh = ctypes.windll.user32.GetSystemMetrics(0), ctypes.windll.user32.GetSystemMetrics(1)
                            pos = (sw - cimg.shape[1] - 8, sh - cimg.shape[0] - 90)
                        except Exception:
                            pos = (900, 500)
                    cv2.moveWindow("Coach", pos[0], pos[1])
                    coach_placed = True
            if pub is not None:
                # the main topic carries the record streak (the assignment's score); the others update on every hit / miss
                vals = {"record": float(game.best), "streak": float(game.streak),
                        "points": round(game.score, 1), "best_points": round(game.best_score, 1)}
                record = vals["record"]
                heartbeat = now - last_pub_t >= MQTT_HEARTBEAT_S
                for name, v in vals.items():
                    if heartbeat or last_vals.get(name) != v:
                        sent = pub.publish_score(v) if name == "record" else pub.publish(name, v)
                        if sent:
                            last_vals[name] = v
                if heartbeat:
                    last_pub_t = now if pub.connected else now - MQTT_HEARTBEAT_S + 0.5   # not connected: retry in 0.5 s
                panel = draw_panel(pub, record, time.time())
                cv2.imshow("MQTT", panel)
                if not panel_placed and windows_placed:
                    px0 = court_img.shape[1] + 12
                    py0 = (cam_img.shape[0] + 40) if (SHOW_CAM and s["frame"] is not None) else 0
                    cv2.moveWindow("MQTT", px0, py0)
                    panel_placed = True

            key = cv2.waitKey(15) & 0xFF                            # also caps the game at ~60 fps
            if PROFILE and time.time() - prof_t >= 1.0:
                prof_t = time.time()
                print(f"{game.state:8} game {fps:3.0f} fps   camera {vision.cam_fps:3.0f} fps (tags {vision.tag_ms:4.0f} ms)"
                      f"   pose {vision.pose_hz:3.0f} Hz ({vision.pose_ms:4.0f} ms)   color {vision.color_ms:4.0f} ms")
            if key == ord("q"):
                break
            elif key == ord("n"):
                start_serve(time.time())
            elif coach is not None and key != 255 and chr(key).lower() in coach.keys:
                coach.select(coach.keys[chr(key).lower()], time.time())
            elif key in (ord("1"), ord("2"), ord("3")):
                game.set_level(key - ord("1"))
                say(f"level {game.level}")
            elif NO_MOTOR and key in (ord("k"), ord(" "), ord("l")):
                detector.inject({ord("k"): 0.2, ord(" "): 0.5, ord("l"): 0.9}[key])
    finally:
        if pub is not None:
            pub.stop()
        detector.stop()
        vision.stop()
        cap.release()
        cv2.destroyAllWindows()
        if motor is not None:
            try:
                motor.disconnect()
            except Exception:
                pass
        print(f"\nDone. Best streak {game.best}, best rally score {game.best_score:.0f}, hits {game.hits}, misses {game.misses}.")


if __name__ == "__main__":
    main()