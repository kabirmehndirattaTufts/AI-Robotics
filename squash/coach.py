"""
The coach: a character with a speech bubble that explains why you missed.

Two coaches. Show AprilTag 2 to the camera for the nice one, tag 3 for the mean one (set in config.py).
Every line below is written ahead of time (nothing is generated while you play). Edit them freely.

Why you missed (from game.py's diagnose()):
    wrong_face   you swung, but the paddle face did not match the ball's color
    arm_down     you swung with your arm below the pose gate
    position     you swung, but the ball was too far to your left / right
    early        you swung before the ball got to you
    late         you swung after the ball went past
    no_swing     no swing at all

Line templates can use {needed} {seen} {dir} {dist}.
"""
import random
import time

import cv2
import numpy as np

LINES = {
    # Ted-style: warm, folksy, hopeful. Ideas: belief and hope, being curious not judgmental, shaking off a mistake,
    # being uncomfortable means you are growing, it is about playing well more than winning, "make today a masterpiece".
    "nice": {
        "greet": ["Hey there, friend! Let's go make today a masterpiece.",
                  "I'll be honest, I know about two things about squash. One of 'em is that you've got this."],
        "serve": ["Here we go. Nobody's comfortable doing something worth doing.",
                  "Be curious out there, not judgmental. Just see what happens.",
                  "We're gonna try this on and see if it fits. Might be real flattering."],
        "hit": ["That's belief in action, right there.", "Hope looks good on you, friend.",
                "Winning's nice, but I sure love how you're playing.", "Now that's what I call a masterpiece swing."],
        "record": ["New record! And I'd be proud of you even if it wasn't.",
                   "Best you've ever done. That's what happens when you believe."],
        "wrong_face": ["Ball's {needed}, paddle showed {seen}. Be curious about it, not hard on yourself.",
                       "Mixed up the sides. Happens to the best of us. Flip it and fly!",
                       "Wrong face, friend. Shake it off like it never happened."],
        "arm_down": ["Arm took a break. Tell it the game's on, then lift it up!",
                     "Let's get that arm up before the swing. You're gonna feel great."],
        "position": ["Ball was {dist} to your {dir}. Slide on over, friend, and try it on again.",
                     "A little more {dir}ward next time. Easy fix!",
                     "You were {dist} off to the {dir}, but I bet those feet can make up for it."],
        "early": ["Whoa, too early! That ball's still doing its own thing. Let it come to you.",
                  "That swing was eager, and I love eager. Just a tad later."],
        "late": ["A smidge late, but you showed up. Start sooner and watch what happens.",
                 "Just a bit behind it. Being uncomfortable means you're growing!"],
        "no_swing": ["No swing that time. I believe in hope, but hope's gotta swing, friend.",
                     "Zero swings? That's just a tie in disguise. Swing at the next one!"],
    },
    # Sue-style: dramatic, scathing, short-tempered, convinced she's the best. Always aimed at the play, never at you.
    "mean": {
        "greet": ["Ah, you chose me. Your courage is adorable.",
                  "I'm detecting a faint scent of mediocrity. Let's fix that."],
        "serve": ["Serve is up. Make me proud, or at least less disappointed.",
                  "Fear is a great motivator. You're welcome.", "Clock's ticking, champ. Impress me."],
        "hit": ["Hm. That was... acceptable. Do not let it go to your head.", "Finally. A swing worthy of my time.",
                "Competence! I almost didn't recognize it.", "A lucky one. Prove me wrong."],
        "record": ["A new record. I will allow myself to be briefly impressed.",
                   "Your best yet. The bar was low, but you cleared it."],
        "wrong_face": ["{needed}! The ball was {needed}! You showed {seen}. A total disaster.",
                       "Wrong color, champ. I've seen better decisions made by a vending machine.",
                       "Orange, blue. Two options. You found a way to pick wrong."],
        "arm_down": ["Your arm is down. Is it on a union break?",
                     "Raise that arm! It's attached to you for a reason."],
        "position": ["The ball was {dist} to your {dir}. You were standing in the wrong zip code.",
                     "{dir} side, champ. Your feet are not decorative. Move them.",
                     "You stood there like a statue. The ball was {dist} to your {dir}."],
        "early": ["You swung before the ball even showed up. Bold, and wrong.",
                  "Patience. Swing early again and I'll hide your paddle."],
        "late": ["Late. The ball left without you, and honestly, I don't blame it.",
                 "That swing arrived after the party ended. Be faster."],
        "no_swing": ["You did not swing. At all. I have seen lamps show more effort.",
                     "Zero swings. I'd call it a protest, but you lack the commitment."],
    },
}

# a plain placeholder when no picture is given (just a cartoon face, in the coach's color)
PLACEHOLDER_BGR = {"nice": (120, 200, 255), "mean": (110, 110, 230)}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def load_image(path):
    """Read a picture (png with transparency is fine). Returns a BGRA array or None."""
    if not path:
        return None
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    except Exception:
        return None
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


class Coach:
    def __init__(self, tag_map=None, names=None, images=None, rng=None, base_dir=".", keys=None):
        import os
        self.tag_map = tag_map or {2: "nice", 3: "mean"}
        self.names = names or {"nice": "Coach Sunny", "mean": "Coach Granite"}
        self.keys = keys or {}                # e.g. {'t': 'nice', 's': 'mean'}, shown as a hint
        self.rng = rng or random.Random()
        self.images = {}
        for who in ("nice", "mean"):
            p = (images or {}).get(who)
            if p and not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            self.images[who] = load_image(p)
        self.active = None            # "nice" / "mean" / None (not chosen yet)
        self.text = ""
        self.kind = None              # what the line is about ("wrong_face", "hit", ...)
        self.text_t = 0.0
        self._last = {}
        self._tag_since = {}          # tag id -> when it first showed up (it must be held briefly)

    # --- choosing the coach with a tag ----------------------------------------------------------
    def see_tags(self, ids, now, allowed=True, hold_s=0.3):
        """ids = AprilTag ids in the current camera frame. Switches coach if tag 2 / 3 is held up."""
        for tid in self.tag_map:
            if tid in ids:
                self._tag_since.setdefault(tid, now)
            else:
                self._tag_since.pop(tid, None)
        if not allowed:
            return
        for tid, who in self.tag_map.items():
            if tid in self._tag_since and now - self._tag_since[tid] >= hold_s and self.active != who:
                self.active = who
                self.say("greet", now)

    def select(self, who, now):
        """Pick a coach directly (a key press), same as showing the tag."""
        if who in LINES and who != self.active:
            self.active = who
            self.say("greet", now)

    # --- talking ---------------------------------------------------------------------------------
    def say(self, kind, now, **fmt):
        if self.active is None:
            return
        pool = LINES[self.active].get(kind) or []
        if not pool:
            return
        choices = [l for l in pool if l != self._last.get((self.active, kind))] or pool
        line = self.rng.choice(choices)
        self._last[(self.active, kind)] = line
        try:
            self.text = line.format(**fmt)
        except (KeyError, IndexError):
            self.text = line
        self.kind, self.text_t = kind, now

    def say_miss(self, miss, now):
        """miss = game.last_miss: {'kind': ..., 'needed':..., 'seen':..., 'dir':..., 'dist':...}"""
        if not miss:
            return
        kind = miss["kind"]
        fmt = {"needed": str(miss.get("needed", "")).upper(),
               "seen": str(miss.get("seen") or "nothing").upper(),
               "dir": miss.get("dir", ""), "dist": miss.get("dist", "a bit")}
        self.say(kind, now, **fmt)


# ---------------------------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------------------------
def _paste_fit(canvas, img, box):
    """
    Draw img (BGRA) inside box (x, y, w, h).
    A picture with a transparent background is shrunk to fit and stands on the bottom of the box.
    A normal photo (no transparency) fills the whole box, cropped from the top so the face stays in view.
    """
    x, y, w, h = box
    ih, iw = img.shape[:2]
    if img[:, :, 3].min() == 255:                                   # opaque photo: fill the box
        s = max(w / iw, h / ih)
        nw, nh = max(w, int(round(iw * s))), max(h, int(round(ih * s)))
        big = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        x0 = (nw - w) // 2
        canvas[y:y + h, x:x + w] = big[0:h, x0:x0 + w, :3]
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (90, 90, 90), 1)
        return
    s = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * s)), max(1, int(ih * s))
    small = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    ox, oy = x + (w - nw) // 2, y + (h - nh)
    roi = canvas[oy:oy + nh, ox:ox + nw]
    a = small[:, :, 3:4].astype(np.float32) / 255.0
    roi[:] = (small[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def _placeholder(canvas, who, box, label=""):
    x, y, w, h = box
    cx, cy, r = x + w // 2, y + h // 2, min(w, h) // 2 - 4
    cv2.circle(canvas, (cx, cy), r, PLACEHOLDER_BGR[who], -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx - r // 3, cy - r // 6), max(2, r // 9), (30, 30, 30), -1, cv2.LINE_AA)
    cv2.circle(canvas, (cx + r // 3, cy - r // 6), max(2, r // 9), (30, 30, 30), -1, cv2.LINE_AA)
    if who == "nice":
        cv2.ellipse(canvas, (cx, cy + r // 6), (r // 2, r // 3), 0, 10, 170, (30, 30, 30), 2, cv2.LINE_AA)
    else:
        cv2.ellipse(canvas, (cx, cy + r // 2), (r // 2, r // 3), 0, 190, 350, (30, 30, 30), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx - r // 2, cy - r // 2), (cx - r // 8, cy - r // 3), (30, 30, 30), 2, cv2.LINE_AA)
        cv2.line(canvas, (cx + r // 2, cy - r // 2), (cx + r // 8, cy - r // 3), (30, 30, 30), 2, cv2.LINE_AA)
    if label:
        cv2.putText(canvas, label, (x + 4, y + h - 2), FONT, 0.4, (200, 200, 200), 1, cv2.LINE_AA)


def _wrap(text, max_px, scale, thick):
    rows, cur = [], ""
    for wd in text.split():
        trial = (cur + " " + wd).strip()
        if cv2.getTextSize(trial, FONT, scale, thick)[0][0] > max_px and cur:
            rows.append(cur)
            cur = wd
        else:
            cur = trial
    rows.append(cur)
    return rows


def draw_coach(coach, now, w=520, h=340):
    img = np.full((h, w, 3), 30, np.uint8)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (60, 60, 60), 2)
    tag_of = {who: tid for tid, who in coach.tag_map.items()}

    if coach.active is None:
        cv2.putText(img, "PICK YOUR COACH", (14, 32), FONT, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
        cw = (w - 3 * 14) // 2
        for i, who in enumerate(("nice", "mean")):
            x = 14 + i * (cw + 14)
            box = (x, 52, cw, h - 52 - 70)
            if coach.images[who] is not None:
                _paste_fit(img, coach.images[who], box)
            else:
                _placeholder(img, who, box, "add an image in config.py")
            key_of = {v: k for k, v in coach.keys.items()}
            cv2.putText(img, f"show TAG {tag_of.get(who, '?')}" + (f"  or press {key_of[who].upper()}" if who in key_of else ""), (x, h - 40), FONT, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(img, coach.names[who], (x, h - 16), FONT, 0.5, (170, 170, 170), 1, cv2.LINE_AA)
        return img

    who = coach.active
    cv2.putText(img, coach.names[who].upper(), (14, 30), FONT, 0.7, (245, 245, 245), 2, cv2.LINE_AA)
    other = "mean" if who == "nice" else "nice"
    ko = {v: k for k, v in coach.keys.items()}.get(other)
    cv2.putText(img, f"tag {tag_of.get(other, '?')}" + (f" / key {ko.upper()}" if ko else "") + f" = {coach.names[other].replace('Coach ', '')}", (w - 250, 30), FONT, 0.45, (150, 150, 150), 1, cv2.LINE_AA)

    # portrait on the left
    pbox = (12, 46, 170, h - 46 - 12)
    if coach.images[who] is not None:
        _paste_fit(img, coach.images[who], pbox)
    else:
        _placeholder(img, who, pbox, "add an image in config.py")

    # speech bubble on the right
    bx0, by0, bx1, by1 = 196, 46, w - 12, h - 56
    cv2.rectangle(img, (bx0, by0), (bx1, by1), (245, 245, 245), -1)
    cv2.fillPoly(img, [np.array([(bx0, by0 + 60), (bx0 - 16, by0 + 78), (bx0, by0 + 88)], np.int32)], (245, 245, 245))
    shown = coach.text[: int(max(0.0, now - coach.text_t) * 70)]          # typewriter effect
    scale, thick = 0.62, 1
    y = by0 + 32
    for row in _wrap(shown, bx1 - bx0 - 24, scale, thick)[:7]:
        cv2.putText(img, row, (bx0 + 12, y), FONT, scale, (40, 40, 40), 1, cv2.LINE_AA)
        y += 28

    label = {"wrong_face": "wrong paddle face", "arm_down": "arm was down", "position": "out of position",
             "early": "swung too early", "late": "swung too late", "no_swing": "no swing",
             "hit": "nice hit", "record": "new record", "serve": "", "greet": ""}.get(coach.kind, "")
    if label:
        cv2.putText(img, f"why: {label}", (196, h - 28), FONT, 0.55, (0, 200, 255), 1, cv2.LINE_AA)
    return img