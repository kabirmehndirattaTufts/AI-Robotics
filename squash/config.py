"""All tunable numbers for the squash game live here."""

# --- Hardware ---------------------------------------------------------------
import legoeducation as le

CARD_COLOR = le.LEGO_COLOR_GREEN
CARD_SERIAL = "0997"
CAMERA_INDEX = None          # None = ask at startup

# --- MQTT -------------------------------------------------------------------
MQTT_TOPIC = "ME193/Rogers/Kabir"   # check the exact topic name the instructor wants

# --- Court (meters, top-down) ----------------------------------------------
COURT_WIDTH = 2.0
COURT_DEPTH = 3.0
REACH_RADIUS = 0.5           # ball must be this close to the player dot to hit

# --- Swing / hit timing -----------------------------------------------------
SWING_THRESHOLD = 2250       # acc_dev units (about milli-g above gravity); non-swing max was 604
SWING_MAX = 12500            # roughly a hard swing; maps to full ball speed      # fill in from M0 data
HIT_WINDOW_S = 0.2           # pose gate + swing + position must line up within this

# --- Levels (ball base speed in m/s), chosen by AprilTag ID ------------------
LEVEL_SPEEDS = {0: 1.0, 1: 1.6, 2: 2.2}

HAPTIC_SPEED_MIN = 85     # soft hit (was 60)
HAPTIC_SPEED_MAX = 100    # hard hit
HAPTIC_CYCLES = 1         # one forward + one back
HAPTIC_PULSE_S = 0.25     # seconds per direction, so the whole kick is 0.5 s

TAG_SIZE_MM = 100        # black square, measured
TAG_LEFT_ID = 0          # worn on YOUR left
TAG_RIGHT_ID = 1         # worn on YOUR right
TAG_SPACING_M = 0.30     # rough guess; it refines itself when both tags are seen
TAG_HOLD_S = 0.3
DISTANCE_SCALE = 1.0     # intentionally uncorrected: court meters are about 1.83x real meters
                         # (measured: 1.5 m real read 2.75, 1.0 m real read 1.83)
SWING_HAND = "right"       # the hand holding the Double Motor
POSE_SWAP_LR = False       # flip to True if the ring lands on the wrong wrist
GATE_OPEN_AT = 0.30        # wrist this far up from hip toward shoulder opens the gate
GATE_CLOSE_AT = 0.20
GATE_REMEMBER_S = 0.6      # swings are detected late, so the gate is "armed" this long after