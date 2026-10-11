"""
Squash game logic. No camera, no Bluetooth, no drawing: just the rules.

Court (meters, top-down):
    court_x   -W/2 .. +W/2     left/right. Positive = the player's right.
    court_z    0 .. D          0 = the front wall (the screen), D = the back line.

Rules:
  * serve()  puts the ball at the front wall, coming toward the player.
  * swing(power, ...) is a hit if, within the last hit_window_s seconds, the
    ball was inside the player's reach circle. (The window forgives Bluetooth
    delay: a swing is detected a bit after it happens.)
  * After a hit the ball goes toward the front wall. Speed = level speed times
    a power multiplier. Angle depends on where in the reach circle the ball was:
    ball to the right of the player's center -> it goes right, like Pong.
  * Ball bounces off the side walls and front wall.
  * Ball passes the back line -> miss: streak resets to 0.

Everything takes `now` (seconds) so it can be tested without real time.
"""
import math
import random
from collections import deque


class SquashGame:
    def __init__(self, court_w=2.0, court_d=3.0, reach=0.5, hit_window_s=0.2,
                 level_speeds=None, level=0, max_angle_deg=35.0,
                 min_power_mult=0.6, max_power_mult=1.4,
                 on_hit=None, on_miss=None, on_whiff=None, rng=None):
        self.W = court_w
        self.D = court_d
        self.reach = reach
        self.hit_window_s = hit_window_s
        self.level_speeds = level_speeds or {0: 1.0, 1: 1.6, 2: 2.2}
        self.level = level
        self.max_angle = math.radians(max_angle_deg)
        self.min_mult = min_power_mult
        self.max_mult = max_power_mult
        self.on_hit = on_hit        # on_hit(power, speed, angle_deg, streak)
        self.on_miss = on_miss      # on_miss(streak_before_reset)
        self.on_whiff = on_whiff    # on_whiff() swing with no ball in reach
        self.rng = rng or random.Random()

        self.state = "waiting"      # 'waiting' or 'playing'
        self.ball = [0.0, 0.0]
        self.vel = [0.0, 0.0]
        self.player = [0.0, self.D * 0.6]
        self.streak = 0
        self.best = 0
        self.hits = 0
        self.misses = 0
        self._last_t = None
        self._history = deque()     # (t, ball_x, ball_z, player_x, player_z)

    # --- setup ---------------------------------------------------------------
    @property
    def base_speed(self):
        if self.level in self.level_speeds:
            return self.level_speeds[self.level]
        return self.level_speeds[min(self.level_speeds)]

    def set_level(self, level):
        self.level = level

    def set_player(self, x, z):
        self.player[0], self.player[1] = x, z

    def serve(self, now):
        """Start a rally: ball leaves the front wall toward the player."""
        self.state = "playing"
        self.streak = 0
        self.ball = [self.rng.uniform(-0.3, 0.3) * self.W, 0.25]
        angle = self.rng.uniform(-0.15, 0.15)
        s = self.base_speed
        self.vel = [s * math.sin(angle), s * math.cos(angle)]
        self._history.clear()
        self._last_t = now

    # --- per-frame update -----------------------------------------------------
    def update(self, now):
        if self._last_t is None:
            self._last_t = now
        dt = min(max(now - self._last_t, 0.0), 0.1)
        self._last_t = now
        if self.state != "playing":
            return

        self.ball[0] += self.vel[0] * dt
        self.ball[1] += self.vel[1] * dt

        half = self.W / 2.0
        if self.ball[0] > half:                       # side walls
            self.ball[0] = half
            self.vel[0] = -abs(self.vel[0])
        elif self.ball[0] < -half:
            self.ball[0] = -half
            self.vel[0] = abs(self.vel[0])
        if self.ball[1] < 0.0:                        # front wall
            self.ball[1] = 0.0
            self.vel[1] = abs(self.vel[1])

        self._history.append((now, self.ball[0], self.ball[1], self.player[0], self.player[1]))
        while self._history and now - self._history[0][0] > 1.0:
            self._history.popleft()

        if self.ball[1] > self.D:                     # past the back line
            before = self.streak
            self.streak = 0
            self.misses += 1
            self.state = "waiting"
            if self.on_miss:
                self.on_miss(before)

    # --- the hit rule -----------------------------------------------------------
    def swing(self, power, now):
        """
        A swing was detected (power 0..1). Returns True if it hit the ball.
        """
        if self.state != "playing":
            return False
        power = max(0.0, min(1.0, power))

        best = None
        slack = 0.04   # one frame: the ball may have left the circle just after its last recorded sample
        for t, bx, bz, px, pz in self._history:
            if now - t > self.hit_window_s + slack:
                continue
            d = math.hypot(bx - px, bz - pz)
            if d <= self.reach and (best is None or d < best[0]):
                best = (d, bx, px)

        # the ball must be coming toward the player, so it can't be hit twice in a row
        if best is None or self.vel[1] <= 0:
            if self.on_whiff:
                self.on_whiff()
            return False

        _, bx, px = best
        offset = max(-1.0, min(1.0, (bx - px) / self.reach))
        angle = offset * self.max_angle
        speed = self.base_speed * (self.min_mult + (self.max_mult - self.min_mult) * power)
        self.vel = [speed * math.sin(angle), -speed * math.cos(angle)]

        self.streak += 1
        self.hits += 1
        self.best = max(self.best, self.streak)
        if self.on_hit:
            self.on_hit(power, speed, math.degrees(angle), self.streak)
        return True
