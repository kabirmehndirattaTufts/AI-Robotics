"""
Look at the recorded IMU data and suggest a swing threshold.

    python tools/analyze_imu.py

Reads every data/imu_<label>_*.csv, builds two simple strength measures,
prints a summary per label, and saves data/summary.png.

  acc_dev  = | length of acceleration vector - 1000 |   (1000 is gravity)
  gyro_mag = length of gyro vector
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
GRAVITY = 1000.0
MIN_PEAK_GAP_S = 0.7


def load(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    t = np.array([float(r["t"]) for r in rows])
    ax, ay, az = (np.array([float(r[k]) for r in rows]) for k in ("accelerometerX", "accelerometerY", "accelerometerZ"))
    gx, gy, gz = (np.array([float(r[k]) for r in rows]) for k in ("gyroscopeX", "gyroscopeY", "gyroscopeZ"))
    raw = np.stack([ax, ay, az, gx, gy, gz], axis=1)
    # keep only rows where a new reading actually arrived
    keep = np.ones(len(t), dtype=bool)
    keep[1:] = np.any(raw[1:] != raw[:-1], axis=1)
    t, ax, ay, az, gx, gy, gz = (v[keep] for v in (t, ax, ay, az, gx, gy, gz))
    acc_dev = np.abs(np.sqrt(ax**2 + ay**2 + az**2) - GRAVITY)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    signed = np.stack([ax, ay, az, gx, gy, gz], axis=1)
    return t, acc_dev, gyro_mag, signed


def peak_indices(t, x, floor):
    """Indices of local maxima above floor, at least MIN_PEAK_GAP_S apart (keep the larger)."""
    found = []
    for i in range(len(x)):
        lo, hi = max(0, i - 1), min(len(x), i + 2)
        if x[i] >= floor and x[i] == x[lo:hi].max():
            if found and t[i] - t[found[-1]] < MIN_PEAK_GAP_S:
                if x[i] > x[found[-1]]:
                    found[-1] = i
            else:
                found.append(i)
    return found


def peaks(t, x, floor):
    return [x[i] for i in peak_indices(t, x, floor)]


def compare_strokes(by_label):
    """Compare forehand vs backhand: strength and which signed axes differ."""
    names = ["accX", "accY", "accZ", "gyrX", "gyrY", "gyrZ"]
    stats = {}
    for label in ("forehand", "backhand"):
        pk_acc, pk_vec = [], []
        for _n, t, acc, _g, signed in by_label[label]:
            for i in peak_indices(t, acc, 1500):
                pk_acc.append(acc[i])
                pk_vec.append(signed[i])
        stats[label] = (np.array(pk_acc), np.array(pk_vec))

    print("\nForehand vs backhand (at each swing peak):")
    for label, (a, v) in stats.items():
        print(f"  {label:9} swings found: {len(a):2d}   peak acc_dev mean {a.mean():6.0f}  std {a.std():5.0f}  min {a.min():6.0f}  max {a.max():6.0f}")
    fa, ba = stats["forehand"][0], stats["backhand"][0]
    if len(fa) and len(ba):
        print(f"  backhand / forehand strength ratio: {ba.mean() / fa.mean():.2f}")

    print("\n  Signed values at the peak (mean +/- std). Look for axes whose sign flips:")
    print(f"  {'axis':6} {'forehand':>18} {'backhand':>18}   verdict")
    fv, bv = stats["forehand"][1], stats["backhand"][1]
    for k, name in enumerate(names):
        fm, fs = fv[:, k].mean(), fv[:, k].std()
        bm, bs = bv[:, k].mean(), bv[:, k].std()
        gap = abs(fm - bm)
        spread = max(fs, bs, 1.0)
        if fm * bm < 0 and gap > 1.5 * spread:
            verdict = "SIGN FLIPS (good separator)"
        elif gap > 1.5 * spread:
            verdict = "differs in size"
        else:
            verdict = "similar"
        print(f"  {name:6} {fm:8.0f} +/- {fs:6.0f} {bm:8.0f} +/- {bs:6.0f}   {verdict}")


def main():
    files = sorted(glob.glob(os.path.join(DATA_DIR, "imu_*_*.csv")))
    if not files:
        print("No files in data/. Run the recorder first.")
        sys.exit(1)

    by_label = {}
    for f in files:
        label = os.path.basename(f).split("_")[1]
        t, acc, gyr, signed = load(f)
        by_label.setdefault(label, []).append((os.path.basename(f), t, acc, gyr, signed))

    print("\nPer label (unique readings only):")
    print(f"{'label':8} {'n':>5}   {'acc_dev median/p90/p99/max':>30}   {'gyro_mag median/p90/p99/max':>30}")
    for label, items in by_label.items():
        acc = np.concatenate([i[2] for i in items])
        gyr = np.concatenate([i[3] for i in items])
        a = [np.median(acc), np.percentile(acc, 90), np.percentile(acc, 99), acc.max()]
        g = [np.median(gyr), np.percentile(gyr, 90), np.percentile(gyr, 99), gyr.max()]
        print(f"{label:8} {len(acc):5d}   {'/'.join(f'{v:.0f}' for v in a):>30}   {'/'.join(f'{v:.0f}' for v in g):>30}")

    if "swing" in by_label:
        swing_peaks = []
        for _, t, acc, _g, _s in by_label["swing"]:
            swing_peaks += peaks(t, acc, floor=1500)
        swing_peaks = np.array(sorted(swing_peaks))
        print(f"\nSwing peaks found (acc_dev above 1500): {len(swing_peaks)}")
        if len(swing_peaks):
            print("  values:", ", ".join(f"{v:.0f}" for v in swing_peaks))
            nonswing = [by_label[k] for k in ("still", "other") if k in by_label]
            ns = np.concatenate([i[2] for items in nonswing for i in items]) if nonswing else np.array([0.0])
            ns_top = np.percentile(ns, 99.5)
            soft = np.percentile(swing_peaks, 20)
            print(f"  non-swing 99.5th percentile: {ns_top:.0f}")
            print(f"  swing peaks 20th percentile: {soft:.0f}")
            if soft > ns_top:
                thr = (soft + ns_top) / 2
                print(f"  -> suggested SWING_THRESHOLD (acc_dev): about {thr:.0f}")
            else:
                print("  -> swings and non-swings overlap on acc_dev; need a better feature or a classifier")

    if "forehand" in by_label and "backhand" in by_label:
        compare_strokes(by_label)

    n = sum(len(v) for v in by_label.values())
    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharey=True, squeeze=False)
    k = 0
    for label, items in by_label.items():
        for name, t, acc, _g, _s in items:
            axes[k][0].plot(t, acc, ".-", lw=0.8, ms=3)
            axes[k][0].set_ylabel(label)
            axes[k][0].set_title(name, fontsize=8, loc="left")
            k += 1
    axes[-1][0].set_xlabel("seconds")
    fig.tight_layout()
    out = os.path.join(DATA_DIR, "summary.png")
    fig.savefig(out, dpi=110)
    print(f"\nSaved plot: {out}")


if __name__ == "__main__":
    main()