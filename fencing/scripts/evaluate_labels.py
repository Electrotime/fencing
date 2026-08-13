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
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(r"c:\Users\aaron\OneDrive\Documents\GitHub\fencing\fencing")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import mediapipe as mp

import demo_video as D
from src.action_model import (ActionFrameLSTM, ActionLSTM, N_AGG_FEATURES,
                              N_AGG_WIDE, load_action_model)
from src.labels import load_intervals, UNSCORABLE
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import (N_LANDMARKS, VISIBILITY_THRESHOLD, _landmarks_to_array,
                               _make_landmarker)

# usage: py -3 scripts/evaluate_labels.py [video.mp4] [labels.csv] [--frame-model]
#
# --frame-model scores models/action_frame.pth instead of the window model. Worth
# a run specifically for the transient classes: the window model reduces its LSTM
# output with `out.mean(dim=1)` over all 60 frames, while a window is SCORED at its
# newest frame -- so a 0.92 s parry sits only at the recent end and gets averaged
# against up to 2 s of whatever preceded it. The per-frame head is reduced with
# frame_logits_to_window(mode="last"), which does not dilute. It is on record as
# hurting inside the ENSEMBLE; it has never been scored standalone against interval
# labels, which is a different question.
# --model PATH   score a different checkpoint (ensemble members `<stem>.m*.pth`
#                beside it are picked up automatically)
# --no-prior     skip the CLASS_PRIOR correction. Required for checkpoints trained
#                on continuous windows at natural frequencies: their prior is
#                already right, so applying it again corrects twice.
# --tag NAME     suffix for the probability cache, so an experimental run does not
#                overwrite the cache belonging to the shipped configuration
_flags = {a.split("=")[0] for a in sys.argv[1:] if a.startswith("--")}


def _flag_value(name, default=None):
    for i, a in enumerate(sys.argv[1:]):
        if a == name and i + 2 <= len(sys.argv[1:]):
            return sys.argv[1:][i + 1]
        if a.startswith(f"{name}="):
            return a.split("=", 1)[1]
    return default


_skip = set()
for _n in ("--model", "--tag"):
    _v = _flag_value(_n)
    if _v is not None:
        _skip.add(_v)
_args = [a for a in sys.argv[1:] if not a.startswith("--") and a not in _skip]
USE_FRAME = "--frame-model" in _flags
NO_PRIOR = "--no-prior" in _flags
MODEL_OVERRIDE = _flag_value("--model")
TAG = _flag_value("--tag", "")
VIDEO = Path(_args[0]) if _args else PROJECT / "data" / "raw_video" / "1.mp4"
LABELS = (Path(_args[1]) if len(_args) > 1
          else PROJECT / "data" / "labels" / "bout1_intervals.csv")
CACHE_OUT = (PROJECT / "data" / "labels" /
             f"{VIDEO.stem}_probs{'_frame' if USE_FRAME else ''}{TAG}.npz")
# UNSCORABLE (`extension` -- a real blade action, but not one of the six classes, so
# the model can never emit it) and the left/right -> A/B slot map both come from
# src.labels now. Windows labelled with it are counted and excluded rather than
# scored as misses.

# ---- ground truth ----------------------------------------------------------
# Accepts both schemas. A TWO-TRACK file (fencer,start,end,footwork,blade) is
# collapsed to one label with BLADE TAKING PRIORITY:
#
#     blade != none  ->  the blade label    (parry)
#     otherwise      ->  the footwork label (retreat, advance, ...)
#
# The model has six mutually-exclusive classes and must pick one, so a collapse is
# unavoidable until there is a second head. Blade-priority is the right collapse
# for two reasons. It makes the convention CONSISTENT -- the same physical event
# (parry while retreating) was previously written `parry` sometimes and `retreat`
# other times, which is unlearnable noise rather than merely coarse labelling. And
# it matches what the model already does: 22 retreat windows were called `parry`,
# and they cluster on real parries (median 2.35 s away, 68% within 3 s) while
# correctly-called retreats do not (median 999 s, 16%).
#
# Collapsing here rather than in the label file keeps the footwork column intact
# on disk for the two-track model, so nothing has to be re-watched later.
# Blade only wins if it is a label the MODEL CAN EMIT. `extension` is not one of
# the six classes, so deferring to it would mark the window UNSCORABLE -- and in
# bout 3 ten of fourteen lunges are written `lunge` + `arm ext`, so a naive
# blade-priority collapse silently deleted almost every lunge from a bout labelled
# specifically for its lunges. Fall through to footwork for anything unemittable.
# The parser and the collapse rule now live in src/labels.py, because this file,
# calibrate_gate.py and bout_timeline.py all need them and a hand-copied third copy
# is how train/serve drift starts. Verified identical on all seven label files
# before the move.
truth, two_track = load_intervals(LABELS)   # slot -> [(start, end, label)]
print(f"labels: {LABELS.name} "
      f"({'two-track, blade-priority collapse' if two_track else 'single-track'})")


def long_window_probs(model, track, opp_track=None):
    """Full class-probability vector for the track's LONG window, or None.

    Uses demo_video._window_inputs rather than reimplementing the preprocessing --
    this function used to be a hand-copy of it, which is exactly how train/serve
    drift starts. Returns the whole distribution rather than the argmax, which is
    what the prior and threshold experiments read from the cache.
    """
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
    for s, e, lab in truth[slot]:
        if s <= t < e:
            return lab
    return None


# ---- run the demo pipeline -------------------------------------------------
person_model = load_person_model()
_default = PROJECT / "models" / ("action_frame.pth" if USE_FRAME else D.MODEL_PATH.name)
_mpath = Path(MODEL_OVERRIDE) if MODEL_OVERRIDE else _default
if not _mpath.is_absolute() and not _mpath.exists():
    _mpath = PROJECT / "models" / _mpath.name
# --pool defaults to demo_video's POOL_MODE so this scores what actually ships.
# Override when evaluating an older checkpoint: modes share parameter shapes, so a
# mismatch loads silently and reports numbers for a model nobody trained.
_pool = _flag_value("--pool", D.POOL_MODE)
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
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

tracks = {s: D.FencerTrack() for s in ("A", "B")}
landmarkers = {s: _make_landmarker(mp.tasks.vision.RunningMode.VIDEO).__enter__()
               for s in ("A", "B")}
prev_gray, pan_windows = None, {}
preds = []          # (slot, time, raw_label, shown_label)
prob_rows = []      # (slot, time, prob_vector) -- cached for offline experiments
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
                preds.append((slot, frame_idx / fps, t.label, shown))
                # cache the full LONG-window probability vector so prior-correction
                # and threshold experiments run offline instead of re-running this
                # 5-minute loop. Safe to compute here: the track already carries the
                # properly-estimated pan, so nothing is being recomputed.
                p = long_window_probs(action_model, t,
                                      tracks["B" if slot == "A" else "A"])
                if p is not None:
                    prob_rows.append((slot, frame_idx / fps, p))
    frame_idx += 1

cap.release()
for s_ in landmarkers.values():
    s_.__exit__(None, None, None)

np.savez(CACHE_OUT,
         slot=np.array([r[0] for r in prob_rows]),
         time=np.array([r[1] for r in prob_rows], dtype=np.float32),
         probs=np.stack([r[2] for r in prob_rows]))
print(f"cached {len(prob_rows)} probability vectors for offline analysis\n")

# ---- score -----------------------------------------------------------------
_scored = [(s, truth_at(s, t), raw, shown) for s, t, raw, shown in preds
           if truth_at(s, t) is not None]
n_unscorable = sum(1 for p in _scored if p[1] in UNSCORABLE)
pairs = [p for p in _scored if p[1] not in UNSCORABLE]
if n_unscorable:
    print(f"excluded {n_unscorable} windows labelled {sorted(UNSCORABLE)} "
          f"(not predictable classes)")
print(f"{len(preds)} predictions, {len(pairs)} inside labelled time\n")

for scope in ("RAW model call", "WHAT THE VIEWER SEES"):
    idx = 2 if scope == "RAW model call" else 3
    print(f"=== {scope} ===")
    # The overlay deliberately collapses neutral/walking to "ready" -- that is the
    # display saying "no action", which is CORRECT when the truth is neutral or
    # walking. Comparing that against the raw truth vocabulary scored every such
    # window as a miss, measuring the overlay's WORDING rather than the model. So
    # for this view the truth is mapped into the same vocabulary the display uses.
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
    print("=== Confusion matrix")
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
