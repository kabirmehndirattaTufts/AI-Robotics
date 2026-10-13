"""All tunable numbers for the squash game live here."""

# --- Hardware ---------------------------------------------------------------
import legoeducation as le

CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0997"
CAMERA_INDEX = None          # None = ask at startup
CAMERA_BACKEND = "MSMF"      # "MSMF" gives 30 fps at 720p on this laptop; "DSHOW" is stuck at 10 fps
CAMERA_MJPG = True           # ask the camera for compressed MJPG frames

# --- MQTT -------------------------------------------------------------------
MQTT_ENABLED = True                 # post the score and show the MQTT window
MQTT_BROKER = "test.mosquitto.org"
MQTT_PORT = 1883
MQTT_TOPIC = "ME193/Rogers/Kabir"   # the score (record streak, as a float) is posted here
MQTT_HEARTBEAT_S = 2.0              # also re-send the score this often, even if it did not change

# --- Court (meters, top-down) ----------------------------------------------
COURT_WIDTH = 2.0
COURT_DEPTH = 3.0
REACH_RADIUS = 0.35          # ball must be this close to the player dot to hit

# --- Swing / hit timing -----------------------------------------------------
SWING_THRESHOLD = 2250       # acc_dev units (about milli-g above gravity); non-swing max was 604
SWING_MAX = 10000            # roughly a hard swing; maps to full ball speed
HIT_WINDOW_S = 0.2           # swing + position must line up within this

# --- Levels (ball base speed in m/s), chosen by AprilTag ID ------------------
LEVEL_SPEEDS = {0: 1.5, 1: 2.8, 2: 3.4}   # old level 2 (2.2) is now level 1

SERVE_DELAY_S = 1.0          # the ball waits this long at the wall after you press n

# --- Scoring and speed ramp --------------------------------------------------
STREAK_SPEED_STEP = 0.05     # each hit in a rally makes the ball 5% (of base speed) faster
MAX_SPEED_RAMP = 1.8         # ... up to 1.8x the level speed
POINTS_BASE = 10             # points per hit before the speed and reaction bonuses
REACTION_REF_S = 1.0         # reaction time (s since the ball left the wall) that earns a x1 bonus

# --- Haptics ----------------------------------------------------------------
HAPTIC_SPEED_MIN = 85        # soft hit
HAPTIC_SPEED_MAX = 100       # hard hit
HAPTIC_CYCLES = 1            # one forward + one back
HAPTIC_PULSE_S = 0.25        # seconds per direction, so the whole kick is 0.5 s

# --- AprilTags (worn on the chest) ---------------------------------------------
TAG_SIZE_MM = 100            # black square, measured
TAG_LEFT_ID = 0              # worn on YOUR left
TAG_RIGHT_ID = 1             # worn on YOUR right
TAG_SPACING_M = 0.30         # rough guess; it refines itself when both tags are seen
TAG_HOLD_S = 0.3
DISTANCE_SCALE = 1.0         # intentionally uncorrected: court meters are about 1.83x real meters
                             # (measured: 1.5 m real read 2.75, 1.0 m real read 1.83)

# --- Tracking switches (all off = the behavior that felt smooth) -------------
TAG_BLUR_TOLERANT = False    # looser tag detector; finds more tags while moving, costs about 1.4x time
POSE_ASSIST = False          # fill tag dropouts with side-to-side from your body in the pose image
# TAG_SMOOTH_TAU_S = 0.15    # smoothing by time instead of per frame (try if the dot is jittery)
# TAG_DETECT_WIDTH = 1280    # detect tags on the full-size image instead of 960 px wide
# TAG_ADAPTIVE_SMOOTH = True
# TAG_REMEMBER_OFFSET = True
# FOLLOW_TAU_S = 0.07        # glide the dot between camera frames

# --- Pose gate (swing arm) ----------------------------------------------------
SWING_HAND = "right"         # the hand holding the Double Motor
POSE_SWAP_LR = False         # flip to True if the ring lands on the wrong wrist
GATE_MODE = "height"         # "height" (arm raised), "side" (wrist beside torso), or "either"
GATE_OPEN_AT = 0.20          # wrist this far up from hip toward shoulder opens the gate
GATE_CLOSE_AT = 0.10         # must be lower than OPEN_AT, or the gate can flicker
GATE_SIDE_OPEN_AT = 0.80     # only used by modes "side" / "either" (see tools/pose_view.py)
GATE_SIDE_CLOSE_AT = 0.60
GATE_REMEMBER_S = 0.2        # swings are detected late, so the gate is "armed" this long after
                             # (if real swings get "arm down", try 0.4)

# --- Paddle color (orange / blue face) ----------------------------------------------
COLOR_MODE = True           # True = ball is orange or blue, hit it with the matching paddle face
                             # (first run python tools/color_view.py to teach it your colors)
COLOR_MAX_REPEAT = 3         # the ball is never the same color more than this many times in a row
COLOR_MAX_AGE_S = None       # None = always use the last color seen before the swing; or a number of seconds to expire it

# --- Serve gesture (the hand that is NOT swinging) ----------------------------------
SERVE_GESTURE = True         # raise the other hand above your head and hold it to serve (n still works)
SERVE_OPEN_AT = 1.3          # wrist height: 1.0 = shoulder level, about 1.2 = chin, 1.4 = top of head
SERVE_CLOSE_AT = 1.1
SERVE_HOLD_S = 0.4           # hold it up this long
SERVE_COOLDOWN_S = 1.0       # after a miss, wait this long before a new serve can start

# --- Window sizes (for screen recordings) ---------------------------------------------
CAMERA_VIEW_SCALE = 0.75      # the --cam window: 0.5 = half size (640 x 360). The camera itself is unchanged.
COURT_VIEW_SCALE = 1.0       # the court window: try 0.8 if it does not fit your screen height

# --- Coach (speech bubble that explains why you missed) ---------------------------------
COACH_ENABLED = True
COACH_TAGS = {2: "nice", 3: "mean"}          # show this AprilTag id to the camera to pick that coach
COACH_NAMES = {"nice": "Coach Ted Lasso", "mean": "Coach Sue Sylvester"}      # the name shown above the picture
COACH_IMAGES = {"nice": "coaches/ted.jpg", "mean": "coaches/sue.jpg"}   # put your pictures here (relative to squash\)
COACH_VIEW_SCALE = 0.8      # size of the coach card: 1.0 = full size (520 x 340), 0.6 = smaller
COACH_KEYS = {"t": "nice", "s": "mean"}       # keys that pick a coach (same as showing the tag): t = Ted, s = Sue
COACH_WINDOW_POS = None      # None = bottom right of the screen. Or (x, y) in pixels.

TAG_TRUE_SPACING_M = 0.2032   # real distance between the centers of your two tags (8 inches)