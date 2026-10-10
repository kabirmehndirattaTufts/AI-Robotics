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
