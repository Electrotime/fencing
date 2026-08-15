"""Calibrate an "is this live fencing?" gate. Measure the cues, don't guess them."""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from ultralytics import YOLO

import demo_video as D
from src.person_detector import MIN_CONFIDENCE, PERSON_CLASS_ID


def load_intervals(path):
    """Any labelled span counts as fencing, whatever the class."""
    spans = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(x for x in f if not x.startswith("#")):
            spans.append((float(r["start"]), float(r["end"])))
    return sorted(spans)


def geometry(xyxy, W, H):
    """(n_tall, h_ratio, sep, foot_dy) for the two most confident tall boxes."""
    n = len(xyxy)
    if n < 2:
        h = float(np.median(xyxy[:, 3] - xyxy[:, 1]) / H) if n else 0.0
        return n, 0.0, 0.0, 1.0, h
    hs = xyxy[:, 3] - xyxy[:, 1]
    order = np.argsort(hs)[::-1][:2]          # two biggest: closest to camera
    a, b = xyxy[order[0]], xyxy[order[1]]
    ha, hb = a[3] - a[1], b[3] - b[1]
    h_ratio = float(min(ha, hb) / max(ha, hb))
    ca, cb = (a[0] + a[2]) / 2, (b[0] + b[2]) / 2
    sep = float(abs(ca - cb) / W)
    foot_dy = float(abs(a[3] - b[3]) / H)
    return n, h_ratio, sep, foot_dy, float(np.median(hs[order]) / H)


def main() -> int:
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    video = Path(args[0]) if args else PROJECT / "data" / "raw_video" / "4.mp4"
    labels = (Path(args[1]) if len(args) > 1 else
              PROJECT / "data" / "labels" / "bout4_intervals_2track.csv")
    step = 5
    for x in sys.argv[1:]:
        if x.startswith("--step"):
            step = int(x.split("=")[1]) if "=" in x else step

    spans = load_intervals(labels)
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows, idx, prev = [], 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step:
            idx += 1
            continue
        t = idx / fps
        idx += 1
        r = model(frame, classes=[PERSON_CLASS_ID], conf=MIN_CONFIDENCE, verbose=False)[0]
        xyxy = (r.boxes.xyxy.cpu().numpy() if r.boxes is not None
                else np.empty((0, 4), np.float32))
        if len(xyxy):
            xyxy = xyxy[(xyxy[:, 3] - xyxy[:, 1]) >= D.MIN_BOX_H_FRAC * H]
        n, hr, sep, dy, bh = geometry(xyxy, W, H)
        small = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        mot = 0.0 if prev is None else float(np.abs(small - prev).mean())
        prev = small
        fencing = any(s <= t < e for s, e in spans)
        rows.append((fencing, n, hr, sep, dy, bh, mot))
    cap.release()

    a = np.array([[r[1], r[2], r[3], r[4], r[5], r[6]] for r in rows], dtype=np.float32)
    y = np.array([r[0] for r in rows], dtype=bool)
    print(f"{len(y)} sampled frames: {y.sum()} fencing, {(~y).sum()} filler "
          f"({y.mean():.0%} fencing)\n")

    names = ["n_tall", "h_ratio", "sep", "foot_dy", "box_h", "motion"]
    print(f"{'cue':<10}{'fencing med':>13}{'filler med':>12}{'AUC':>7}")

    def auc(pos, neg):
        v = np.concatenate([pos, neg]); o = v.argsort()
        rk = np.empty(len(v)); rk[o] = np.arange(1, len(v) + 1)
        return float((rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                     / (len(pos) * len(neg)))

    for i, nm in enumerate(names):
        print(f"{nm:<10}{np.median(a[y, i]):>13.2f}{np.median(a[~y, i]):>12.2f}"
              f"{auc(a[y, i], a[~y, i]):>7.2f}")

    print("\n=== candidate rule: exactly 2 tall people, similar size, separated ===")
    print(f"{'h_ratio>=':>10}{'sep>=':>8}{'dy<=':>7}{'precision':>11}{'recall':>9}{'on':>7}")
    best = None
    for hr in (0.0, 0.5, 0.6, 0.7):
        for sp in (0.0, 0.10, 0.15, 0.20):
            for dy in (1.0, 0.15, 0.10):
                keep = (a[:, 0] == 2) & (a[:, 1] >= hr) & (a[:, 2] >= sp) & (a[:, 3] <= dy)
                if keep.sum() == 0:
                    continue
                prec = float(y[keep].mean())
                rec = float(keep[y].mean())
                print(f"{hr:>10.2f}{sp:>8.2f}{dy:>7.2f}{prec:>11.0%}{rec:>9.0%}"
                      f"{keep.mean():>7.0%}")
                score = prec + 0.3 * rec
                if best is None or score > best[0]:
                    best = (score, hr, sp, dy, prec, rec)
    if best:
        print(f"\nbest by precision+0.3*recall: h_ratio>={best[1]}, sep>={best[2]}, "
              f"foot_dy<={best[3]}  ->  {best[4]:.0%} precision, {best[5]:.0%} recall")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
