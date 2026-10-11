"""
M2: play the squash game with the mouse and keyboard (no camera, no motor).

    python tools/game_sim.py

Mouse   moves the player (this stands in for your AprilTag position)
Keys    n = serve the ball          space or k = swing, medium power
        j = soft swing              l = hard swing
        1 / 2 / 3 = level 0 / 1 / 2     q = quit

Move so the ball is inside your reach circle, then swing.
Ball to the right of your center goes right, to the left goes left.
"""
import os
import sys
import time
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import cv2
import numpy as np

import config
from game import SquashGame

PX_PER_M = 200
MARGIN = 50
W_M = getattr(config, "COURT_WIDTH", 2.0)
D_M = getattr(config, "COURT_DEPTH", 3.0)
HUD_H = 110


def to_px(x, z):
    return int(MARGIN + (x + W_M / 2.0) * PX_PER_M), int(MARGIN + z * PX_PER_M)


def from_px(px, py):
    return (px - MARGIN) / PX_PER_M - W_M / 2.0, (py - MARGIN) / PX_PER_M


def draw(game, trail, banner, now):
    W = int(W_M * PX_PER_M) + 2 * MARGIN
    H = int(D_M * PX_PER_M) + 2 * MARGIN + HUD_H
    img = np.full((H, W, 3), 35, np.uint8)
    x0, y0 = to_px(-W_M / 2, 0)
    x1, y1 = to_px(W_M / 2, D_M)
    cv2.rectangle(img, (x0, y0), (x1, y1), (90, 90, 90), 1)
    cv2.line(img, (x0, y0), (x1, y0), (255, 255, 255), 5)
    cv2.line(img, (x0, y1), (x1, y1), (60, 60, 200), 2)                      # back line = miss
    cv2.putText(img, "WALL / SCREEN", (x0, y0 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    px, py = to_px(*game.player)
    cv2.circle(img, (px, py), int(game.reach * PX_PER_M), (110, 110, 110), 1)
    cv2.circle(img, (px, py), 9, (80, 220, 80), -1)

    for i, (tx, tz) in enumerate(trail):
        c = 60 + int(150 * i / max(1, len(trail)))
        cv2.circle(img, to_px(tx, tz), 3, (c, c, c), -1)
    if game.state == "playing":
        cv2.circle(img, to_px(*game.ball), 8, (255, 255, 255), -1)

    y = int(D_M * PX_PER_M) + 2 * MARGIN - 10
    cv2.putText(img, f"streak {game.streak}    best {game.best}    level {game.level}",
                (10, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
    cv2.putText(img, "n serve  space swing  j soft  l hard  1/2/3 level  q quit",
                (10, y + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    if game.state == "waiting":
        cv2.putText(img, "press n to serve", (10, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 220, 255), 1)
    if banner and now < banner[2]:
        cv2.putText(img, banner[0], (W // 2 - 70, MARGIN + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, banner[1], 2)
    return img


def main():
    state = {"banner": None}

    def say(text, color, now=None):
        state["banner"] = (text, color, time.time() + 0.8)

    game = SquashGame(
        court_w=W_M, court_d=D_M,
        reach=getattr(config, "REACH_RADIUS", 0.5),
        hit_window_s=getattr(config, "HIT_WINDOW_S", 0.2),
        level_speeds=getattr(config, "LEVEL_SPEEDS", None),
        on_hit=lambda p, s, a, k: (print(f"HIT  streak {k}  power {p:.2f}  speed {s:.2f}  angle {a:+.0f} deg"),
                                   say("HIT", (80, 230, 80))),
        on_miss=lambda k: (print(f"MISS  streak was {k}"), say("MISS", (80, 80, 240))),
        on_whiff=lambda: (print("whiff"), say("whiff", (80, 200, 240))),
    )
    game.set_player(0.0, D_M * 0.6)

    def on_mouse(event, mx, my, flags, param):
        x, z = from_px(mx, my)
        game.set_player(max(-W_M / 2, min(W_M / 2, x)), max(0.2, min(D_M, z)))

    cv2.namedWindow("Squash sim")
    cv2.setMouseCallback("Squash sim", on_mouse)
    trail = deque(maxlen=12)

    while True:
        now = time.time()
        game.update(now)
        if game.state == "playing":
            trail.append(tuple(game.ball))
        else:
            trail.clear()
        cv2.imshow("Squash sim", draw(game, trail, state["banner"], now))

        key = cv2.waitKey(10) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("n"):
            game.serve(time.time())
        elif key in (ord(" "), ord("k")):
            game.swing(0.7, time.time())
        elif key == ord("j"):
            game.swing(0.3, time.time())
        elif key == ord("l"):
            game.swing(1.0, time.time())
        elif key in (ord("1"), ord("2"), ord("3")):
            game.set_level(key - ord("1"))
            print(f"level {game.level}  (base ball speed {game.base_speed})")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
