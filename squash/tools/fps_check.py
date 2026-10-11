"""
Where is the time going? Measures frames per second for each part, 5 s each:

    python tools/fps_check.py

  1. camera only
  2. camera + AprilTags
  3. camera + pose
  4. camera + both  (what the game will do)

Stand where you will play, with your tags on, in the lighting you will use.
Camera only is the speed limit: if it is already about 10 fps, the camera (usually
dim light making it expose longer) is the problem, not the code.
"""
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import config

SECONDS = 5.0


def measure(label, cap, work, seconds=SECONDS):
    n = 0
    t0 = time.time()
    work_time = 0.0
    while time.time() - t0 < seconds:
        ok, frame = cap.read()
        if not ok:
            break
        w0 = time.time()
        work(frame)
        work_time += time.time() - w0
        n += 1
    elapsed = time.time() - t0
    fps = n / elapsed if elapsed > 0 else 0.0
    ms_work = 1000.0 * work_time / n if n else 0.0
    print(f"  {label:22} {fps:5.1f} fps    {ms_work:6.1f} ms of processing per frame")
    return fps


def main():
    from pose import PoseTracker
    from vision import TagTracker, open_camera

    cap = open_camera(getattr(config, "CAMERA_INDEX", None))
    tags = TagTracker(tag_size_m=getattr(config, "TAG_SIZE_MM", 100) / 1000.0)
    pose = PoseTracker(hand=getattr(config, "SWING_HAND", "right"),
                       swap_lr=getattr(config, "POSE_SWAP_LR", False))
    for _ in range(10):                      # let the camera settle
        cap.read()

    print(f"\nCamera reports {cap.get(5):.0f} fps setting, frame {cap.get(3):.0f} x {cap.get(4):.0f}")
    print(f"Measuring {SECONDS:.0f} s each...\n")
    try:
        measure("camera only", cap, lambda f: None)
        measure("camera + tags", cap, lambda f: tags.update(f))
        measure("camera + pose", cap, lambda f: pose.update(f))
        measure("camera + tags + pose", cap, lambda f: (tags.update(f), pose.update(f)))
    finally:
        cap.release()
    print("\nDone.")


if __name__ == "__main__":
    main()
