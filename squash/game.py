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
  * Ball passes the back line -> miss: streak and score reset to 0.
  * Each hit makes the next ball faster (speed_step per hit, up to max_ramp).
  * Points per hit = points_base x (incoming speed / base speed) x reaction bonus, where the reaction
    bonus is reaction_ref_s / (seconds since the ball left the front wall), clamped. A fast ball hit
    soon after it leaves the wall (you stood close to the wall) is worth the most.

Everything takes `now` (seconds) so it can be tested without real time.
"""
import math
import random
from collections import deque


class SquashGame:
    def __init__(self, court_w=2.0, court_d=3.0, reach=0.5, hit_window_s=0.2,
                 level_speeds=None, level=0, max_angle_deg=35.0,
                 min_power_mult=0.6, max_power_mult=1.4,
                 speed_step=0.0, max_ramp=1.0,
                 serve_delay_s=0.0,
                 points_base=10.0, reaction_ref_s=1.0,
                 min_reaction_factor=0.5, max_reaction_factor=3.0,
                 on_hit=None, on_miss=None, on_whiff=None, color_mode=False, on_wrong_face=None, rng=None):
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
        self.speed_step = speed_step          # ball gets this much faster (fraction of base) per hit in the rally
        self.max_ramp = max_ramp              # ... up to this multiple of the base speed
        self.serve_delay_s = serve_delay_s    # the ball waits this long at the wall after a serve
        self._serve_at = 0.0
        self.points_base = points_base
        self.reaction_ref_s = reaction_ref_s  # a hit with this much reaction time earns x1 from the reaction bonus
        self.min_rf = min_reaction_factor
        self.max_rf = max_reaction_factor
        self.rng = rng or random.Random()
        self.color_mode = color_mode          # ball has a color; you must hit with that paddle face
        self.on_wrong_face = on_wrong_face    # on_wrong_face(needed, seen)  seen may be None
        self.ball_color = "orange"
        self.serving = False          # True while a served ball is still flying to the front wall (not hittable)

        self.state = "waiting"      # 'waiting' or 'playing'
        self.ball = [0.0, 0.0]
        self.vel = [0.0, 0.0]
        self.player = [0.0, self.D * 0.6]
        self.streak = 0
        self.best = 0
        self.score = 0.0            # points this rally (resets on a miss)
        self.best_score = 0.0
        self.last_points = 0.0      # points from the latest hit, and what produced them
        self.last_reaction_s = 0.0
        self.last_speed_factor = 1.0
        self.last_reaction_factor = 1.0
        self.hits = 0
        self.misses = 0
        self._front_t = None        # when the ball last left the front wall (reaction time starts here)
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
        self.score = 0.0
        self._serve_at = now + self.serve_delay_s     # ball sits still until then
        self._front_t = self._serve_at                # (reset again at the front wall bounce)
        self.ball_color = self.rng.choice(("orange", "blue"))
        # the serve: the ball starts in the back part of the court and flies to the FRONT wall first.
        # It can't be hit until it has bounced off the front wall, which gives you time before the first swing.
        self.ball = [self.rng.uniform(-0.35, 0.35) * self.W, self.rng.uniform(0.65, 0.9) * self.D]
        target_x = self.rng.uniform(-0.3, 0.3) * self.W
        dx, dz = target_x - self.ball[0], -self.ball[1]
        norm = math.hypot(dx, dz)
        s = self.base_speed
        self.vel = [s * dx / norm, s * dz / norm]
        self.serving = True
        self._history.clear()
        self._last_t = now

    def ready_in(self, now):
        """Seconds until a served ball starts moving (0 if it is already moving)."""
        return max(0.0, self._serve_at - now)

    # --- per-frame update -----------------------------------------------------
    def update(self, now):
        if self._last_t is None:
            self._last_t = now
        dt = min(max(now - self._last_t, 0.0), 0.1)
        self._last_t = now
        if self.state != "playing":
            return
        if now < self._serve_at:                      # serve delay: the ball waits at the wall
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
            self._front_t = now
            self.serving = False

        self._history.append((now, self.ball[0], self.ball[1], self.player[0], self.player[1]))
        while self._history and now - self._history[0][0] > 1.0:
            self._history.popleft()

        if self.ball[1] > self.D:                     # past the back line
            before = self.streak
            self.streak = 0
            self.score = 0.0
            self.misses += 1
            self.state = "waiting"
            if self.on_miss:
                self.on_miss(before)

    # --- the hit rule -----------------------------------------------------------
    def swing(self, power, now, face=None):
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
                best = (d, bx, px, t)

        # the ball must be coming toward the player, so it can't be hit twice in a row
        if best is None or self.vel[1] <= 0:
            if self.on_whiff:
                self.on_whiff()
            return False

        _, bx, px, t_hit = best

        # color rule: the paddle face seen just before the swing must match the ball. Wrong face = no hit,
        # the ball keeps coming (so you can still hit it with the right face if there is time).
        if self.color_mode and face != self.ball_color:
            if self.on_wrong_face:
                self.on_wrong_face(self.ball_color, face)
            return False

        # --- points: faster incoming ball and less reaction time earn more ---
        incoming = math.hypot(self.vel[0], self.vel[1])
        reaction = max(t_hit - (self._front_t if self._front_t is not None else t_hit), 0.05)
        speed_factor = incoming / self.base_speed
        reaction_factor = max(self.min_rf, min(self.max_rf, self.reaction_ref_s / reaction))
        points = self.points_base * speed_factor * reaction_factor
        self.last_points, self.last_reaction_s = points, reaction
        self.last_speed_factor, self.last_reaction_factor = speed_factor, reaction_factor
        self.score += points
        self.best_score = max(self.best_score, self.score)

        # --- the ball goes back, and gets faster the longer the rally lasts ---
        offset = max(-1.0, min(1.0, (bx - px) / self.reach))
        angle = offset * self.max_angle
        ramp = min(self.max_ramp, 1.0 + self.speed_step * (self.streak + 1))
        speed = self.base_speed * ramp * (self.min_mult + (self.max_mult - self.min_mult) * power)
        self.vel = [speed * math.sin(angle), -speed * math.cos(angle)]

        if self.color_mode:
            self.ball_color = self.rng.choice(("orange", "blue"))   # next ball, new color
        self.streak += 1
        self.hits += 1
        self.best = max(self.best, self.streak)
        if self.on_hit:
            self.on_hit(power, speed, math.degrees(angle), self.streak)
        return True