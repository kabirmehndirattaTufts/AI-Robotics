"""Try camera backend / resolution / format combos and print the real fps of each.

Run:  python tools/cam_probe.py
Takes about 1 minute. Paste the whole table back.
"""
import time
import cv2

BACKENDS = [("MSMF", cv2.CAP_MSMF), ("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY)]
SIZES = [(1280, 720), (640, 480)]
FORMATS = [None, "MJPG"]
SECONDS = 2.0


def fps_of(backend, w, h, fmt):
    cap = cv2.VideoCapture(0, backend)
    if not cap.isOpened():
        return None
    if fmt:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fmt))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, 30)
    for _ in range(8):  # warm up
        cap.read()
    n = 0
    got = None
    t0 = time.time()
    while time.time() - t0 < SECONDS:
        ok, frame = cap.read()
        if ok:
            n += 1
            got = frame.shape
    dt = time.time() - t0
    cap.release()
    return (n / dt, got)


print(f"{'backend':8} {'asked':10} {'format':8} {'fps':>6}  got")
for bname, b in BACKENDS:
    for (w, h) in SIZES:
        for fmt in FORMATS:
            try:
                r = fps_of(b, w, h, fmt)
            except Exception as e:
                r = None
            label = f"{w}x{h}"
            if r is None:
                print(f"{bname:8} {label:10} {str(fmt):8} {'fail':>6}")
            else:
                f, shp = r
                got = f"{shp[1]}x{shp[0]}" if shp else "no frames"
                print(f"{bname:8} {label:10} {str(fmt):8} {f:6.1f}  {got}")
