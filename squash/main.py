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
from vision_thread import VisionThread

NO_MOTOR = "--no-motor" in sys.argv
SHOW_CAM = "--cam" in sys.argv
PROFILE = "--profile" in sys.argv

W_M = getattr(config, "COURT_WIDTH", 2.0)
D_M = getattr(config, "COURT_DEPTH", 3.0)
REACH = getattr(config, "REACH_RADIUS", 0.5)
PX_PER_M = 220
MARGIN = 50
HUD_H = 150
FOLLOW_TAU_S = getattr(config, "FOLLOW_TAU_S", 0.0)   # 0 = off. Otherwise: smaller = follows faster, larger = smoother


def to_px(x, z):
    return int(MARGIN + (x + W_M / 2) * PX_PER_M), int(MARGIN + z * PX_PER_M)


def draw_court(game, tag_status, armed, lift, msg, msg_age, fps, cam_fps=0.0, raw=None):
    W = int(W_M * PX_PER_M) + 2 * MARGIN
    H = int(D_M * PX_PER_M) + MARGIN + HUD_H
    img = np.full((H, W, 3), 35, np.uint8)
    x0, y0 = to_px(-W_M / 2, 0)
    x1, y1 = to_px(W_M / 2, D_M)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), 5)
    cv2.line(img, (x0, y1), (x1, y1), (60, 60, 200), 2)                  # back line = miss

    px, py = to_px(*game.player)
    ring = (80, 220, 80) if armed else (110, 110, 110)                    # green ring = arm raised
    cv2.circle(img, (px, py), int(REACH * PX_PER_M), ring, 2)
    cv2.circle(img, (px, py), 9, (230, 230, 230), -1)
    if raw is not None:                                                   # unsmoothed tag position (--cam only)
        rx, ry = to_px(max(-W_M / 2, min(W_M / 2, raw[0])), max(0.0, min(D_M, raw[1])))
        cv2.circle(img, (rx, ry), 5, (60, 60, 255), -1)

    if game.state == "playing":
        bx, by = to_px(*game.ball)
        cv2.circle(img, (bx, by), 11, (0, 200, 255), -1)

    ty = y1 + 28
    cv2.putText(img, f"STREAK {game.streak}", (10, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, f"best {game.best}   level {game.level}   hits {game.hits}  misses {game.misses}",
                (10, ty + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    lift_txt = "no pose" if lift is None else f"{lift:.2f}"
    cv2.putText(img, f"tags {tag_status}   arm {lift_txt} {'ARMED' if armed else 'down'}   {fps:.0f} fps (cam {cam_fps:.0f})",
                (10, ty + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    if game.state == "waiting":
        cv2.putText(img, "press n to serve", (10, ty + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    if msg and msg_age < 1.5:
        cv2.putText(img, msg, (10, ty + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
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
        cv2.circle(frame, wr, 22, (80, 220, 80) if s["gate_open"] else (0, 0, 255), 3)
    lift = pose.gate.lift
    cv2.putText(frame, "pose: not found" if pose.points is None else
                (f"lift --  ({pose.gate.reason})" if lift is None else f"lift {lift:.2f}  (opens at {pose.gate.open_at})"),
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
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

    state = {"msg": "", "t": 0.0}

    def say(text):
        state["msg"], state["t"] = text, time.time()
        print(text)

    def on_hit(power, speed, angle, streak):
        say(f"HIT  streak {streak}   power {power:.2f}   angle {angle:+.0f}")
        detector.mute(haptics.duration + 0.4)       # the wiggle shakes the IMU
        haptics.hit(power)

    def on_miss(streak_before):
        say(f"MISS  (streak was {streak_before})  press n to serve")

    game = SquashGame(
        court_w=W_M, court_d=D_M, reach=REACH,
        hit_window_s=getattr(config, "HIT_WINDOW_S", 0.2),
        level_speeds=config.LEVEL_SPEEDS, level=0,
        on_hit=on_hit, on_miss=on_miss,
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
    )
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
    cap = open_camera(getattr(config, "CAMERA_INDEX", None),
                      backend=getattr(config, "CAMERA_BACKEND", "MSMF"),
                      mjpg=getattr(config, "CAMERA_MJPG", True))
    print("Running. Click the Squash window, then: n = serve, 1/2/3 = level, q = quit.")
    if NO_MOTOR:
        print("No-motor mode: k = soft swing, space = medium, l = hard.")

    vision = VisionThread(cap, tracker, pose,
                          pose_assist=getattr(config, "POSE_ASSIST", False),
                          assist_max_s=getattr(config, "POSE_ASSIST_MAX_S", 1.5))
    vision.start()

    shown, shown_t = None, time.time()
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

            for ev in detector.pop_events():
                tag = f"{ev['stroke']}, power {ev['power']:.2f}"
                if game.state != "playing":
                    continue                                        # swinging between rallies is free
                if not armed:
                    say(f"swing ignored: arm down, lift {'none' if lift is None else format(lift, '.2f')} ({tag})")
                elif not game.swing(ev["power"], ev["t_onset"]):
                    say(f"WHIFF: ball not in reach ({tag})")

            cv2.imshow("Squash", draw_court(game, s["tag_status"], armed, lift,
                                            state["msg"], time.time() - state["t"], fps, vision.cam_fps,
                                            s["raw"] if SHOW_CAM else None))
            if SHOW_CAM and s["frame"] is not None:
                cv2.imshow("Camera", draw_camera(s["frame"].copy(), s, pose))

            key = cv2.waitKey(15) & 0xFF                            # also caps the game at ~60 fps
            if PROFILE and time.time() - prof_t >= 1.0:
                prof_t = time.time()
                print(f"{game.state:8} game {fps:3.0f} fps   camera {vision.cam_fps:3.0f} fps (tags {vision.tag_ms:4.0f} ms)"
                      f"   pose {vision.pose_hz:3.0f} Hz ({vision.pose_ms:4.0f} ms)")
            if key == ord("q"):
                break
            elif key == ord("n"):
                game.serve(time.time())
            elif key in (ord("1"), ord("2"), ord("3")):
                game.set_level(key - ord("1"))
                say(f"level {game.level}")
            elif NO_MOTOR and key in (ord("k"), ord(" "), ord("l")):
                detector.inject({ord("k"): 0.2, ord(" "): 0.5, ord("l"): 0.9}[key])
    finally:
        detector.stop()
        vision.stop()
        cap.release()
        cv2.destroyAllWindows()
        if motor is not None:
            try:
                motor.disconnect()
            except Exception:
                pass
        print(f"\nDone. Best streak {game.best}, hits {game.hits}, misses {game.misses}.")


if __name__ == "__main__":
    main()