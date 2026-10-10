# Squash game (ME193)

Virtual squash: AprilTag = player position and level, pose = arm raised gate,
Double Motor IMU = swing strength, MQTT = live streak score, motor wiggle = haptics.

## Setup (Windows PowerShell)
```
cd "C:\Users\13013\Documents\AI Robotics\AI-robotics\squash"
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```
Also needs `lelib.py` and `mqttlib.py` in this folder (instructor libraries).

## Every new session
```
cd "C:\Users\13013\Documents\AI Robotics\AI-robotics\squash"
.\.venv\Scripts\Activate.ps1
git pull
```

## Hardware notes
- Double Motor: green connection card, serial 0997
- Camera index: set in config.py

## Run
- IMU discovery: `python tools/imu_logger.py discover`
- Record swings: `python tools/imu_logger.py record swing 10`

## Milestones
- [ ] M0 IMU logger: see fields, pick swing threshold, check update rate
- [ ] M1 Tag to top-down dot
- [ ] M2 Ball physics (keyboard stand-in)
- [ ] M3 Pose gate
- [ ] M4 Swing detection
- [ ] M5 Hit rule and miss logic
- [ ] M6 Levels and start via tag
- [ ] M7 MQTT score
- [ ] M8 Haptics, polish, YOLO paddle face
