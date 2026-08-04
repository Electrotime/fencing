"""FIRST REAL EVALUATION — model predictions against Aaron's interval labels.

Everything before this was measured either on hand-trimmed clips (which carry
artifacts of how they were cut) or on the bout LABEL MIX (a distribution, which
twice looked fine while individual labels were wrong). With ground truth we can
finally compute per-class precision and recall on continuous footage.

Reuses demo_video's own functions rather than reimplementing the loop -- a
previous diagnostic reimplemented it, stubbed camera pan to ZERO, and produced a
conclusion that was off by 3x. Pan feeds net-forward and travel, so it is not
optional.

Scoring rules:
  - a window is scored at the frame the call is made on (its newest frame)
  - unlabelled time is EXCLUDED, not treated as neutral
  - both the RAW model call and what the viewer actually SEES ("ready" for
    quiet classes / low confidence) are reported, because they answer different
    questions
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(r"c:\Users\aaron\OneDrive\Documents\GitHub\fencing\fencing")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import mediapipe as mp

import demo_video as D
from src.action_model import CLASS_NAMES, load_action_model
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import (N_LANDMARKS, VISIBILITY_THRESHOLD, _landmarks_to_array,
                               _make_landmarker)

VIDEO = PROJECT / "data" / "raw_video" / "1.mp4"   # verified aligned to the labels
LABELS = PROJECT / "data" / "labels" / "bout1_intervals.csv"
SLOT_OF = {"left": "A", "right": "B"}

# ---- ground truth ----------------------------------------------------------
truth = defaultdict(list)          # slot -> [(start, end, label)]
with open(LABELS) as f:
    for row in csv.DictReader(r for r in f if not r.startswith("#")):
        truth[SLOT_OF[row["fencer"]]].append(
            (float(row["start"]), float(row["end"]), row["label"].strip()))


def truth_at(slot, t):
    for s, e, lab in truth[slot]:
        if s <= t < e:
            return lab
    return None


# ---- run the demo pipeline -------------------------------------------------
person_model = load_person_model()
action_model = load_action_model(PROJECT / "models" / "action_lstm.pth",
                                 device=torch.device("cpu"))
cap = cv2.VideoCapture(str(VIDEO))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracks = {s: D.FencerTrack() for s in ("A", "B")}
landmarkers = {s: _make_landmarker(mp.tasks.vision.RunningMode.VIDEO).__enter__()
               for s in ("A", "B")}
prev_gray, pan_windows = None, {}
preds = []          # (slot, time, raw_label, shown_label)
frame_idx = 0

while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    pan = D._frame_pan(prev_gray, gray, pan_windows)
    prev_gray = gray

    box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=D.MIN_BOX_H_FRAC)
    boxes = D._assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)

    for slot, box in (("A", boxes["A"]), ("B", boxes["B"])):
        track = tracks[slot]
        kp = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
        if box is not None:
            crop = crop_box(frame, box)
            if crop is not None:
                res = landmarkers[slot].detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                    int(frame_idx * 1000 / fps))
                kp = _landmarks_to_array(res)
                x1, y1, x2, y2 = box
                kp[:, 0] = (x1 + kp[:, 0] * (x2 - x1)) / W
                kp[:, 1] = (y1 + kp[:, 1] * (y2 - y1)) / H
                low = kp[:, 3] < VISIBILITY_THRESHOLD
                kp[low, :3] = track.prev[low, :3]
                track.prev = kp.copy()
                track.last_hip_x = float((kp[23, 0] + kp[24, 0]) / 2)
        track.kp.append(kp)
        track.motion.append((pan, track.last_hip_x))

    if frame_idx % D.PREDICT_EVERY == 0 and frame_idx >= D.WINDOW_LONG:
        for slot in ("A", "B"):
            t = tracks[slot]
            t.label = None
            D._predict(action_model, t)
            if t.label is not None:
                shown = ("ready" if (t.label in D.QUIET_CLASSES
                                     or t.conf < D.ACTION_CONF_FLOOR) else t.label)
                preds.append((slot, frame_idx / fps, t.label, shown))
    frame_idx += 1

cap.release()
for s in landmarkers.values():
    s.__exit__(None, None, None)

# ---- score -----------------------------------------------------------------
pairs = [(s, truth_at(s, t), raw, shown) for s, t, raw, shown in preds
         if truth_at(s, t) is not None]
print(f"{len(preds)} predictions, {len(pairs)} inside labelled time\n")

for scope in ("RAW model call", "WHAT THE VIEWER SEES"):
    idx = 2 if scope == "RAW model call" else 3
    print(f"=== {scope} ===")
    labels = sorted({p[1] for p in pairs} | {p[idx] for p in pairs})
    tp = Counter(); fp = Counter(); fn = Counter()
    for _, gt, raw, shown in pairs:
        pr = raw if idx == 2 else shown
        if pr == gt:
            tp[gt] += 1
        else:
            fp[pr] += 1
            fn[gt] += 1
    acc = sum(tp.values()) / len(pairs)
    print(f"  overall accuracy {acc:.1%}")
    print(f"  {'class':<10}{'n_true':>8}{'precision':>11}{'recall':>9}")
    for c in labels:
        n_true = sum(1 for _, gt, _, _ in pairs if gt == c)
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else float("nan")
        r = tp[c] / n_true if n_true else float("nan")
        if n_true or (tp[c] + fp[c]):
            print(f"  {c:<10}{n_true:>8}{p:>11.0%}{r:>9.0%}")
    print()

print("=== CONFUSION (raw call), rows = truth ===")
cls = sorted({p[1] for p in pairs})
cols = sorted({p[2] for p in pairs})
print(f"{'truth\\pred':<12}" + "".join(f"{c[:7]:>9}" for c in cols))
for g in cls:
    row = Counter(raw for _, gt, raw, _ in pairs if gt == g)
    print(f"{g:<12}" + "".join(f"{row[c]:>9}" for c in cols))

print("\n=== per fencer (raw) ===")
for s in ("A", "B"):
    sp = [p for p in pairs if p[0] == s]
    if sp:
        a = sum(1 for _, gt, raw, _ in sp if gt == raw) / len(sp)
        print(f"  fencer {s} ({'left' if s=='A' else 'right'}): {a:.1%} on {len(sp)} windows")
