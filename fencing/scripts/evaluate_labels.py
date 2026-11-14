"""Score the demo pipeline against interval labels and cache its window probabilities."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import mediapipe as mp

import demo_video as D
from src.action_model import (ActionFrameLSTM, ActionLSTM, N_AGG_FEATURES,
                              N_AGG_WIDE, load_action_model)
from src.labels import load_intervals, UNSCORABLE
from src.person_detector import get_fencer_boxes, load_person_model
from src.pose_pipeline import _make_landmarker

LAB = PROJECT / "data" / "labels"
DEFAULT_VIDEO = PROJECT / "data" / "raw_video" / "1.mp4"

_ap = argparse.ArgumentParser(description=__doc__)
_ap.add_argument("video", type=Path, nargs="?", default=DEFAULT_VIDEO)
_ap.add_argument("labels", type=Path, nargs="?",
                 help="interval CSV; default data/labels/bout<stem>_intervals[_2track].csv")
_ap.add_argument("--model", help="checkpoint path, or a file name under models/")
_ap.add_argument("--tag", default="", help="suffix for the probability cache, e.g. _held")
_ap.add_argument("--pool", default=D.POOL_MODE, choices=ActionLSTM.POOL_MODES)
_ap.add_argument("--window", type=int, default=0, help="long-window length in frames")
_ap.add_argument("--frame-model", action="store_true")
_ap.add_argument("--no-prior", action="store_true")
_ap.add_argument("--no-fps-normalise", action="store_true",
                 help="reproduce pre-2026-08-22 runs on 60 fps input")
_ap.add_argument("--no-labels", action="store_true",
                 help="cache probabilities only, score nothing")
_ap.add_argument("--dump-tracks", action="store_true")
_a = _ap.parse_args()

# On by default: no correct workflow wants a 60-frame window to mean 1 s on
# 60 fps input.
FPS_NORM = not _a.no_fps_normalise
USE_FRAME = _a.frame_model
NO_PRIOR = _a.no_prior
MODEL_OVERRIDE = _a.model
TAG = _a.tag
WINDOW = _a.window
DUMP_TRACKS = _a.dump_tracks
VIDEO = _a.video


def _find_labels(video):
    """The interval file for this video, if one sits in data/labels under the usual name."""
    for name in (f"bout{video.stem}_intervals.csv", f"bout{video.stem}_intervals_2track.csv"):
        if (LAB / name).exists():
            return LAB / name
    return None


LABELS = None if _a.no_labels else (_a.labels or _find_labels(VIDEO))
if not VIDEO.exists():
    sys.exit(f"video not found: {VIDEO}")
if LABELS is not None and not LABELS.exists():
    sys.exit(f"labels not found: {LABELS}")
CACHE_OUT = LAB / f"{VIDEO.stem}_probs{'_frame' if USE_FRAME else ''}{TAG}.npz"

if WINDOW:
    D.WINDOW_LONG = WINDOW
    D.MIN_REAL_FRAMES = min(D.MIN_REAL_FRAMES, max(D.MIN_REAL_SHORT, WINDOW // 2))
    print(f"window override: {WINDOW} frames, min real {D.MIN_REAL_FRAMES}")

if LABELS is None:
    truth, two_track = {}, False
    print(f"no labels for {VIDEO.name}: caching probabilities only")
else:
    truth, two_track = load_intervals(LABELS)   # slot -> [(start, end, label)]
    print(f"labels: {LABELS.name} "
          f"({'two-track, blade-priority collapse' if two_track else 'single-track'})")


def long_window_probs(model, track, opp_track=None):
    """Full class-probability vector for the track's LONG window, or None."""
    kp_seq = np.stack(track.kp)[-D.WINDOW_LONG:]
    mot = np.array(track.motion, dtype=np.float32)[-D.WINDOW_LONG:]
    got = D._window_inputs(kp_seq, mot, D.MIN_REAL_FRAMES)
    if got is None:
        return None
    flat, agg, n_real = got
    if D.USE_OPPONENT:
        opp_got = None
        if opp_track is not None and len(opp_track.kp) == len(track.kp):
            opp_got = D._window_inputs(
                np.stack(opp_track.kp)[-D.WINDOW_LONG:],
                np.array(opp_track.motion, dtype=np.float32)[-D.WINDOW_LONG:],
                D.MIN_REAL_FRAMES)
        agg = D.wide_agg(agg, None if opp_got is None else opp_got[1])
    lengths = torch.tensor([n_real])
    with torch.no_grad():
        logits = model(torch.from_numpy(flat[:D.SEQ_LEN])[None],
                       torch.from_numpy(agg)[None], lengths)
        if logits.ndim == 3:
            logits = D.frame_logits_to_window(logits, lengths, mode="last")
        return torch.softmax(logits, dim=1)[0].numpy()


def truth_at(slot, t):
    for s, e, lab in truth.get(slot, ()):
        if s <= t < e:
            return lab
    return None


# ---- run the demo pipeline -------------------------------------------------
person_model = load_person_model()
_default = PROJECT / "models" / ("action_frame.pth" if USE_FRAME else D.MODEL_PATH.name)
_mpath = Path(MODEL_OVERRIDE) if MODEL_OVERRIDE else _default
if not _mpath.is_absolute() and not _mpath.exists():
    _mpath = PROJECT / "models" / _mpath.name
_pool = _a.pool
action_model = load_action_model(
    _mpath, device=torch.device("cpu"),
    cls=ActionFrameLSTM if USE_FRAME
    else (lambda: ActionLSTM(pool=_pool,
                             n_agg=N_AGG_WIDE if D.USE_OPPONENT else N_AGG_FEATURES)))
print(f"model: {_mpath.name}" + ("" if USE_FRAME else f"  pool={_pool}")
      + ("  [prior DISABLED for this run]" if NO_PRIOR else ""))
if NO_PRIOR:
    # A model trained on continuous windows at their natural frequencies already
    # has the right prior; multiplying CLASS_PRIOR in again would correct twice.
    D.APPLY_CLASS_PRIOR = False
cap = cv2.VideoCapture(str(VIDEO))
if not cap.isOpened():
    sys.exit(f"couldn't open {VIDEO}")
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

tracks = {s: D.FencerTrack() for s in ("A", "B")}
landmarkers = {s: _make_landmarker(mp.tasks.vision.RunningMode.VIDEO).__enter__()
               for s in ("A", "B")}
prev_gray, pan_windows = None, {}
preds = []          # (slot, time, raw_label, shown_label)
prob_rows = []      # (slot, time, prob_vector) -- cached for offline experiments
# The window is 60 FRAMES, so decimating to a common rate keeps its DURATION equal.
STRIDE = D.fps_stride(fps) if FPS_NORM else 1
if STRIDE > 1:
    print(f"fps-normalise: {fps:.1f} fps, keeping every {STRIDE} frames "
          f"-> window spans {D.WINDOW_LONG * STRIDE / fps:.2f}s")
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
kept_times = []
# FencerTrack.kp is a deque(maxlen=SEQ_LEN), so it only ever holds the last 60
# frames -- accumulate the full history separately or the dump is 60 frames long.
full_kp = {"A": [], "B": []}
full_mot = {"A": [], "B": []}
frame_idx = 0
raw_idx = -1

while True:
    if raw_idx > 0 and raw_idx % 3000 == 0:
        print(f"  ...{raw_idx}/{n_frames} frames ({raw_idx/n_frames:.0%})", flush=True)
    ok, frame = cap.read()
    if not ok:
        break
    raw_idx += 1
    if raw_idx % STRIDE:
        continue
    now_s = raw_idx / fps
    kept_times.append(now_s)
    gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
    pan = D._frame_pan(prev_gray, gray, pan_windows)
    prev_gray = gray

    box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=D.MIN_BOX_H_FRAC)
    boxes = D._assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)

    kps = D.track_frame(frame, boxes, tracks, landmarkers, pan, int(now_s * 1000))
    if DUMP_TRACKS:
        for slot in ("A", "B"):
            full_kp[slot].append(kps[slot].copy())
            full_mot[slot].append((pan, tracks[slot].last_hip_x))

    if frame_idx % D.PREDICT_EVERY == 0 and frame_idx >= D.WINDOW_LONG:
        for slot in ("A", "B"):
            tracks[slot].label = None
            # opponent track: both slots already have this frame appended
            D._predict(action_model, tracks[slot],
                       tracks["B" if slot == "A" else "A"])
        # scored AFTER the gate, so this measures what the demo actually shows
        D._apply_parry_gate(tracks)
        for slot in ("A", "B"):
            t = tracks[slot]
            if t.label is not None:
                shown = ("ready" if (t.label in D.QUIET_CLASSES
                                     or t.conf < D.ACTION_CONF_FLOOR) else t.label)
                preds.append((slot, now_s, t.label, shown))
                p = long_window_probs(action_model, t,
                                      tracks["B" if slot == "A" else "A"])
                if p is not None:
                    prob_rows.append((slot, now_s, p))
    frame_idx += 1

cap.release()
for s_ in landmarkers.values():
    s_.__exit__(None, None, None)

if not prob_rows:
    sys.exit("no fencer was tracked long enough to predict anything -- nothing cached")
np.savez(CACHE_OUT,
         slot=np.array([r[0] for r in prob_rows]),
         time=np.array([r[1] for r in prob_rows], dtype=np.float32),
         probs=np.stack([r[2] for r in prob_rows]))
print(f"cached {len(prob_rows)} probability vectors -> {CACHE_OUT.name}\n")

if DUMP_TRACKS:
    tp = CACHE_OUT.with_name(f"{VIDEO.stem}_tracks{TAG}.npz")
    np.savez_compressed(
        tp, times=np.array(kept_times, dtype=np.float32),
        **{f"kp_{s_}": np.stack(full_kp[s_]).astype(np.float32) for s_ in ("A", "B")},
        **{f"motion_{s_}": np.array(full_mot[s_], dtype=np.float32) for s_ in ("A", "B")})
    print(f"dumped tracks to {tp.name} ({tp.stat().st_size / 1e6:.0f} MB, "
          f"{len(kept_times)} frames)", flush=True)

# ---- score -----------------------------------------------------------------
if LABELS is None:
    sys.exit(0)
_scored = [(s, truth_at(s, t), raw, shown) for s, t, raw, shown in preds
           if truth_at(s, t) is not None]
n_unscorable = sum(1 for p in _scored if p[1] in UNSCORABLE)
pairs = [p for p in _scored if p[1] not in UNSCORABLE]
if n_unscorable:
    print(f"excluded {n_unscorable} windows labelled {sorted(UNSCORABLE)} "
          f"(not predictable classes)")
print(f"{len(preds)} predictions, {len(pairs)} inside labelled time\n")
if not pairs:
    sys.exit("nothing to score: no prediction falls inside a labelled interval")

for scope in ("RAW model call", "WHAT THE VIEWER SEES"):
    idx = 2 if scope == "RAW model call" else 3
    print(f"=== {scope} ===")
    def as_shown(c):
        return "ready" if (idx == 3 and c in D.QUIET_CLASSES) else c

    scored = [(as_shown(gt), raw if idx == 2 else shown)
              for _, gt, raw, shown in pairs]
    labels = sorted({g for g, _ in scored} | {p for _, p in scored})
    tp = Counter(); fp = Counter(); fn = Counter()
    for gt, pr in scored:
        if pr == gt:
            tp[gt] += 1
        else:
            fp[pr] += 1
            fn[gt] += 1
    acc = sum(tp.values()) / len(pairs)
    print(f"  overall accuracy {acc:.1%}")
    print(f"  {'class':<10}{'n_true':>8}{'precision':>11}{'recall':>9}")
    for c in labels:
        n_true = sum(1 for gt, _ in scored if gt == c)
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else float("nan")
        r = tp[c] / n_true if n_true else float("nan")
        if n_true or (tp[c] + fp[c]):
            print(f"  {c:<10}{n_true:>8}{p:>11.0%}{r:>9.0%}")
    print()
print("=== CONFUSION (raw call), rows = truth ===")
cls = sorted({p[1] for p in pairs})
cols = sorted({p[2] for p in pairs})
_corner = "truth\\pred"     # a backslash inside an f-string needs Python 3.12
print(f"{_corner:<12}" + "".join(f"{c[:7]:>9}" for c in cols))
for g in cls:
    row = Counter(raw for _, gt, raw, _ in pairs if gt == g)
    print(f"{g:<12}" + "".join(f"{row[c]:>9}" for c in cols))

print("\n=== per fencer (raw) ===")
for s in ("A", "B"):
    sp = [p for p in pairs if p[0] == s]
    if sp:
        a = sum(1 for _, gt, raw, _ in sp if gt == raw) / len(sp)
        print(f"  fencer {s} ({'left' if s=='A' else 'right'}): {a:.1%} on {len(sp)} windows")
