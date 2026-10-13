# Virtual Squash

A single-player squash game you play with your body. A camera tracks where you stand, a pose model checks that your
swinging arm is up, a LEGO Education Double Motor in your hand senses the swing and buzzes when you hit, and your
record streak is published live over MQTT. Made for ME193.

```
camera ──► AprilTags ───────► where you stand ─────┐
       └─► pose (MediaPipe) ► arm raised? ─────────┤
       └─► color mask ──────► paddle face seen ────┼──► game ──► court window
Double Motor IMU ──────────► swing power / time ───┘      │
                                                           ├──► haptics (motor buzz)
                                                           ├──► MQTT score + scoreboard window
                                                           └──► coach window
```

## How a rally works

1. **Serve.** Raise your left hand above your head and hold it for 0.4 s (or press `n`). The ball starts in the back
   of the court, flies to the front wall, and bounces back toward you. It cannot be hit until it has bounced.
2. **Stand in the right place.** Two AprilTags on your chest give your position. The ball is hit only if it is
   inside the ring around your dot when you swing.
3. **Swing with your arm up.** The pose model must see your swing wrist raised (the ring around your dot is bright
   when it is).
4. **Match the color.** The ball is orange or blue. The paddle (the 3D printed case holding the Double Motor) has an
   orange face and a blue face. The face the camera saw just before the swing must match the ball.
5. **Swing in time.** The Double Motor's IMU reports the swing. The hit counts if the ball was in reach within
   0.2 s of the swing.
6. A hit sends the ball back to the front wall, a little faster each time, and buzzes the motor. Letting the ball
   past the back line ends the rally.

### Scoring

- **Streak:** hits in a row. The **record streak** is the best streak of the session and is the score posted to MQTT.
- **Points:** each hit earns `POINTS_BASE x (incoming ball speed / base speed) x reaction bonus`. The bonus is larger
  the sooner after the front-wall bounce you hit the ball (clamped to 0.5x to 3x).
- The ball speeds up by 5% of its base speed with every hit, up to 1.8x.

### The coach

After a miss, a coach (Ted Lasso or Sue Sylvester) says why in a speech bubble: wrong paddle face, arm down, out of
position (too far left or right), too early, too late, or no swing. Pick the coach by pressing `T` or `S`, or by
showing AprilTag 2 (Ted) or 3 (Sue) to the camera. All the lines live in `coach.py`.

## Hardware

| Item | Notes |
|---|---|
| LEGO Education Double Motor | Held in your swinging hand inside the case with an orange side and a blue side |
| LEGO connection card | Green card, serial `0997` (set in `config.py`) |
| Webcam | 720p. On Windows the `MSMF` backend gives about 30 fps (`DSHOW` is stuck at 10 fps on the dev laptop) |
| 2 AprilTags (family 36h11, ids 0 and 1) | Black square 100 mm. Worn on the chest, id 0 on your left, id 1 on your right, centers 8 inches apart |
| 2 more tags (optional, ids 2 and 3) | Pick the coach by showing the tag to the camera |

## Setup (Windows PowerShell)

```
cd squash
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Needs `lelib.py` (the course's LEGO Bluetooth library) in this folder. The pose model
(`pose_landmarker_lite.task`) downloads itself on first run.

Install only `opencv-contrib-python` for OpenCV. The AprilTag code needs the contrib build, and having plain
`opencv-python` installed as well breaks it.

Coach pictures: make a folder `coaches/` and put `ted.jpg` and `sue.jpg` in it (names are set by `COACH_IMAGES` in
`config.py`). Without them the coach window shows a plain placeholder face.

### Printing the tags

```
python tools/make_tags.py 0 1 2 3
```

Print the PDFs in `tags/` at 100% ("actual size", not "fit to page"). Measure the black square and set
`TAG_SIZE_MM` in `config.py` (default 100). Keep the white border around each tag.

### Teaching the game your paddle colors

```
python tools/color_view.py
```

1. Hold the orange face to the camera and click on the orange.
2. Hold the blue face up and click on the blue.
3. Check the mask window: the paddle face should fill in solid with nothing else lighting up. Press `s` and click on
   your skin if skin leaks into the orange mask. `=` and `-` widen or narrow the hue range.

The result is saved to `color_calib.json` and read by the game. Re-run this if the lighting changes.

## Run

```
python main.py --cam
```

| Flag | What it does |
|---|---|
| `--cam` | Also show the camera window (tags, wrist ring, serve hand) |
| `--no-motor` | Play without the Double Motor: `k` soft swing, `space` medium, `l` hard |
| `--profile` | Print the speed of each vision thread once a second |

Click a game window first so it has keyboard focus.

| Key | Action |
|---|---|
| `n` | Serve |
| `t` / `s` | Pick Coach Ted Lasso / Coach Sue Sylvester |
| `q` | Quit |

You get four windows: the **court** (top-down view, scoreboard, color indicators), the **camera**, the **MQTT**
status panel, and the **coach**.

## MQTT

Published with `paho-mqtt` to `test.mosquitto.org:1883` (QoS 0, not retained). Every value is a float sent as text,
sent whenever it changes and re-sent every 2 s.

| Topic | Value |
|---|---|
| `ME193/Rogers/Kabir` | Record streak (the score) |
| `ME193/Rogers/Kabir/streak` | Current streak |
| `ME193/Rogers/Kabir/points` | Current points |
| `ME193/Rogers/Kabir/best_points` | Highest point total |

The program also subscribes to its own topics. The MQTT window shows the connection state, each message as it is
sent, and the broker echoing it back with the round-trip time, which confirms the link works end to end. To watch
from another client, subscribe to `ME193/Rogers/Kabir/#`.

## How the code is organized

```
squash/
  main.py            game loop, windows, keys, wiring everything together
  config.py          every tunable number (hardware, thresholds, switches)
  game.py            ball physics, hit rule, scoring, color rule, miss diagnosis
  vision.py          AprilTag tracker (position) and camera opening
  vision_thread.py   runs the camera/tags, pose and color mask in background threads
  pose.py            MediaPipe pose landmarker + arm gates (swing arm and serve arm)
  paddle.py          HSV color mask that tells which paddle face the camera sees
  swing.py           reads the IMU in its own thread and reports swings
  haptics.py         motor buzz on a hit, in its own thread
  scoreboard.py      MQTT publisher and the MQTT status panel
  coach.py           coach lines, character window, miss reasons
  lelib.py           LEGO Bluetooth library (course provided)
  coaches/           coach pictures
  tags/              printable AprilTags
  tools/             test and calibration scripts (below)
```

### Vision

Three background threads read the newest camera frame, so a slow step never slows the game:

- **Tags** (`vision.py`): OpenCV's AprilTag detector (36h11) on a 960 px wide copy of the frame. Each tag's pose is
  solved with `solvePnP` (an uncalibrated pinhole model, focal length = frame width). Your position is the point
  halfway between the two tags, `court_x` left/right of the camera's center and `court_z` away from the camera. When only one tag
  is visible, the other tag's place is known from the real 8 inch spacing, so the halfway point is still found
  (`TAG_TRUE_SPACING_M`). A short hold keeps the last position through brief dropouts.
- **Pose** (`pose.py`): MediaPipe's pose landmarker. The swing arm's `lift` is
  `(hip_y - wrist_y) / (hip_y - shoulder_y)`: 0 is hip height and 1 is shoulder height. It is relative to
  your own body, so it works at any distance. When your hips are out of frame, the torso length is estimated from
  shoulder width. The gate opens at `GATE_OPEN_AT` and closes at a lower value (hysteresis), and it is
  "remembered" for `GATE_REMEMBER_S` because the IMU reports a swing slightly after it happens. The other arm has its
  own gate that triggers the serve.
- **Color** (`paddle.py`): the frame is shrunk to 320 px, converted to HSV, and masked for orange and blue using the
  ranges clicked in `color_view.py`. Only the biggest blob of each color counts, and the larger blob has to beat
  the other by 1.5x. The most recent clear face is remembered, and a swing is judged against the face seen at or just
  before the moment the swing started.

### Swing detection (`swing.py`)

The Double Motor's accelerometer streams over Bluetooth at about 10 Hz. The swing signal is
`| |acceleration| - 1 g |`. The first reading above `SWING_THRESHOLD` opens a 0.15 s window, and the biggest reading
in it is the swing's peak. `power` is the peak scaled to 0 to 1 by `SWING_MAX`. The sign of the gyro at the peak gives
forehand or backhand (printed in the log). Detection is muted during the haptic buzz, because the motor shaking
would be counted as a swing.

### Game rule (`game.py`)

A swing is a hit when all of these are true:

1. The IMU reported a swing.
2. The pose gate was open (arm raised) in the last `GATE_REMEMBER_S`.
3. The ball was inside the ring of radius `REACH_RADIUS` around your dot within `HIT_WINDOW_S` of the swing, and it
   was coming toward you.
4. The paddle face last seen matches the ball's color.

The hit angle depends on where in the ring the ball was (to your right sends it right), and the return speed
depends on swing power. A color is chosen at random for every serve and every hit, never more than
`COLOR_MAX_REPEAT` in a row.

When you miss, `game.py` looks at what happened during that approach and picks the most useful reason for the
coach: wrong face, then arm down, then position, then early or late, then no swing.

## Configuration

Everything is in `config.py`. The settings you are most likely to touch:

| Setting | Meaning |
|---|---|
| `CARD_SERIAL`, `CARD_COLOR` | Which connection card to pair with |
| `CAMERA_INDEX`, `CAMERA_BACKEND` | Camera selection (`None` asks at startup) |
| `MQTT_TOPIC`, `MQTT_BROKER`, `MQTT_ENABLED` | Where the score goes |
| `LEVEL_SPEEDS` | Ball base speed in m/s (level 0 is used) |
| `STREAK_SPEED_STEP`, `MAX_SPEED_RAMP` | How fast the ball speeds up |
| `REACH_RADIUS`, `HIT_WINDOW_S` | How forgiving a hit is |
| `SWING_THRESHOLD`, `SWING_MAX` | IMU sensitivity and the full power point |
| `GATE_OPEN_AT`, `GATE_CLOSE_AT`, `GATE_REMEMBER_S` | Arm-raised gate |
| `SERVE_GESTURE`, `SERVE_OPEN_AT`, `SERVE_HOLD_S` | Left-hand serve |
| `COLOR_MODE`, `COLOR_MIN_AREA_FRAC` | Turn the color rule on, and how big the paddle face must look |
| `TAG_TRUE_SPACING_M` | Real distance between your tag centers (0.2032 m = 8 in) |
| `COACH_*` | Coach tags, keys, names, pictures, window size and place |
| `CAMERA_VIEW_SCALE`, `COURT_VIEW_SCALE` | Window sizes for screen recording |

## Tools

Each script in `tools/` tests one part on its own.

| Script | Purpose |
|---|---|
| `imu_logger.py`, `analyze_imu.py` | Record labeled swings and suggest a swing threshold |
| `swing_test.py` | Live swing counter with power |
| `motor_check.py` | Check which motor commands work and how long each takes |
| `haptics_test.py` | Try the hit buzz (`--live` for the real motor) |
| `tag_view.py`, `track_view.py` | See AprilTag tracking and the dot position |
| `pose_view.py` | See the pose gate (lift, reach, open or closed) |
| `color_view.py` | Teach and check the paddle colors |
| `game_sim.py` | Play the game with the mouse and keyboard, no hardware |
| `make_tags.py` | Make printable AprilTags |
| `fps_check.py`, `cam_probe.py` | Find where time goes and which camera mode is fastest |

## Troubleshooting

- **Camera stuck near 10 fps:** use `CAMERA_BACKEND = "MSMF"`, plug in the charger, and close other apps using the
  camera. Run `python tools/cam_probe.py` to compare modes.
- **The ring sits on the wrong wrist:** set `POSE_SWAP_LR = True`.
- **Swings say "arm down":** lower `GATE_OPEN_AT` or raise `GATE_REMEMBER_S` (try 0.4).
- **The ring stays green and the paddle color is never seen:** make sure `COLOR_MODE = True`, that
  `color_calib.json` exists, and lower `COLOR_MIN_AREA_FRAC`. Re-run `tools/color_view.py` if the lighting changed.
- **MQTT window stays on CONNECTING:** check the internet connection. The game keeps running without it.
- **The dot jumps when you move:** AprilTags blur at speed. Keep the tags flat and in good light, and check that
  `TAG_TRUE_SPACING_M` matches the real spacing.