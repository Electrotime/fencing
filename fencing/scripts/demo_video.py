"""Minimal end-to-end demo: run the whole stack on a video and save an annotated copy.

Per frame: YOLO finds the fencer(s) -> each one gets cropped and run through
MediaPipe pose -> sliding windows of keypoints feed the action LSTM at two
scales (a short one so brief actions like parries aren't buried, a long one
for sustained footwork) -> skeleton, boxes, labels and blade tip get drawn.
Works with one or two fencers. Offline, writes <video>_demo.mp4 next to the input.

Run from project root:  python scripts/demo_video.py path/to/video.mp4 [out.mp4]
"""

from __future__ import annotations

import sys
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.action_model import (ActionFrameLSTM, ActionLSTM, CLASS_NAMES, INPUT_SIZE,
                              SEQ_LEN, _engineered_features, frame_logits_to_window,
                              load_action_model)
from src.blade_detector import get_blade_tip, load_blade_model
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import (N_LANDMARKS, VISIBILITY_THRESHOLD,
                               _landmarks_to_array, _make_landmarker,
                               _normalize_sequence)
from src.utils import draw_action_label, draw_blade_tip, draw_skeleton

MODEL_PATH = PROJECT_ROOT / "models" / "action_lstm.pth"
FRAME_MODEL_PATH = PROJECT_ROOT / "models" / "action_frame.pth"  # --frame-model
BLADE_WEIGHTS = (PROJECT_ROOT / "models" / "blade_yolo" / "fencing_blade_v2"
                 / "weights" / "best.pt")

PREDICT_EVERY = 5     # frames between action predictions (labels hold in between)
MIN_REAL_FRAMES = 15  # don't guess an action until we've tracked this many frames

# multi-scale windows: a parry lasts ~12 frames, so 2s of surrounding footwork
# buries it in a single 60-frame window (measured: parry recall 3x better at
# short windows). Fast actions get judged on the short window, sustained ones
# (advance/retreat rhythm) on the long one.
WINDOW_LONG = SEQ_LEN   # 60 frames / 2 s
WINDOW_SHORT = 25       # ~0.8 s
MIN_REAL_SHORT = 12
FAST_CLASSES = {"parry"}   # brief actions the short window is allowed to override with
FAST_CONF = 0.65           # a short-window override has to be at least this sure
# quiet states: not fencing actions, so the overlay stays blank on them (this is why
# neutral/walking exist -- labels only light up when something real happens)
QUIET_CLASSES = {"neutral", "walking"}
ACTION_CONF_FLOOR = 0.50   # below this the call is too unsure to show as an action
# How often each class actually occurs in continuous footage, measured from
# data/labels/bout1_intervals.csv (787 labelled windows). The model is trained with
# inverse-frequency class weighting, which drives its effective prior to UNIFORM
# (16.7% each) -- but real fencing is nothing like uniform, and the clip corpus
# over-represents the exciting classes because clips were cut OF actions
# (lunge 10% of training windows vs 2.6% of reality, parry 10% vs 1.0%).
# Untreated, `lunge` stops behaving like a class and becomes a default: 21 true
# windows, predicted 378 times, swallowing 122 of 151 real advances.
# Correcting for it is standard label-shift: p(c|x) * target_prior / train_prior.
# Validated by fitting the prior on one half of the labelled footage and scoring
# the other (both directions): 15.7%->42.6% and 22.6%->41.2%. Per class, advance
# goes 5%->67% recall. NOTE this is one 104 s bout; re-estimate as more footage is
# labelled, and set APPLY_CLASS_PRIOR=False to measure without it.
APPLY_CLASS_PRIOR = True
# Pooled duration shares from BOTH labelled bouts (218 s). Re-derive with
# scripts/estimate_class_prior.py whenever more footage is labelled.
CLASS_PRIOR = {"advance": 0.184, "lunge": 0.045, "parry": 0.017,
               "retreat": 0.122, "neutral": 0.230, "walking": 0.401}
# Transfer is ASYMMETRIC and the prior is NOT universal. Measured:
#            bout1   bout2
#   uniform  19.2%   46.1%     bout2 needs little correction -- it is action-dense
#   bout1     50.2%*  43.3%    (*circular) transfers forward fine
#   bout2     24.4%   47.2%*   transfers BADLY backward: under-weights walking
#   pooled    45.9%   44.1%    most balanced, hence shipped
# Bout 1 is idle-heavy (walking 0.457) and bout 2 action-dense (walking 0.232), so
# a prior lifted from busy footage wrecks quiet footage. If you analyse material
# with very different action density, re-estimate. Unsupervised EM estimation does
# NOT work here (it put lunge at 0.635 vs a true 0.027) -- the prior has to come
# from labels.

MAX_FROZEN_FRAC = 0.25  # skip the window if more than this share of joint steps are
                        # exactly zero, i.e. held over from the previous frame by the
                        # visibility fallback. Partial/occluded bodies were producing
                        # confident `advance` calls off frozen skeletons.
MIN_BOX_H_FRAC = 0.25 # ignore "people" shorter than this fraction of frame height.
                      # Was 0.35, which was tuned on bout 2's tight framing and silently
                      # amputated wider shots: on bout 2 the fencers run 0.30-0.60 so 0.35
                      # was harmless, but bout 1 frames them at 0.20-0.50 (median box 0.360)
                      # and 0.35 discarded 44% of REAL detections -- a slot dropped to 27%
                      # coverage and fencers went unlabelled for stretches. Calibrated, not
                      # guessed: bout 2's banner-graphic detections top out at 0.20 and its
                      # real fencers start at 0.30, so 0.25 sits in that empty gap -- every
                      # banner still rejected, 96% of bout 1's real boxes kept.
                      # (banner graphics of fencers trigger real YOLO detections --
                      # measured one at 0.30 of frame height, real fencers 0.4+)
PAN_STRIP_FRAC = 0.22
PAN_MIN_RESPONSE = 0.08
SLOT_COLORS = {"A": (0, 200, 255), "B": (255, 200, 0)}  # amber / cyan-blue (BGR)


class FencerTrack:
    """Sliding history for one fencer slot (A = left, B = right)."""

    def __init__(self) -> None:
        self.kp: deque[np.ndarray] = deque(maxlen=SEQ_LEN)      # (33, 4) frame fractions
        self.motion: deque[tuple[float, float]] = deque(maxlen=SEQ_LEN)  # (pan, hip_x)
        self.prev = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
        self.last_hip_x = 0.5
        self.label: str | None = None
        self.conf = 0.0
        self.counts: dict[str, int] = {}  # how often each label got emitted


def _frame_pan(prev_gray: np.ndarray | None, gray: np.ndarray,
               windows: dict) -> float:
    """Horizontal background shift between two frames, from L/R border strips."""
    if prev_gray is None:
        return 0.0
    h, w = gray.shape
    strip_w = max(10, int(PAN_STRIP_FRAC * w))
    rows = slice(int(0.10 * h), int(0.75 * h))  # skip broadcast graphics / scoreboard
    if "win" not in windows:
        windows["win"] = cv2.createHanningWindow((strip_w, rows.stop - rows.start), cv2.CV_32F)
    shifts = []
    for a, b in [(0, strip_w), (w - strip_w, w)]:
        (dx, _), response = cv2.phaseCorrelate(prev_gray[rows, a:b], gray[rows, a:b],
                                               windows["win"])
        if response > PAN_MIN_RESPONSE:
            shifts.append(dx)
    return float(np.median(shifts)) if shifts else 0.0


def _assign_boxes(dets: list[np.ndarray], tracks: dict[str, FencerTrack],
                  W: int) -> dict[str, np.ndarray | None]:
    """Match this frame's detections to the A/B tracks by continuity.

    Pure left/right slotting splits a fencer's history across two tracks the
    moment a second person enters or she drifts over the frame midline -- the
    action window resets and labels vanish. So: whoever has history keeps the
    box nearest their last hip position; left/right only breaks fresh starts."""
    slots: dict[str, np.ndarray | None] = {"A": None, "B": None}
    if not dets:
        return slots
    history = {s: tracks[s].last_hip_x
               for s, t in tracks.items() if any(np.any(k) for k in t.kp)}
    cxs = [float((b[0] + b[2]) / 2 / W) for b in dets]

    if len(dets) == 1:
        if history:
            slot = min(history, key=lambda s: abs(cxs[0] - history[s]))
        else:
            slot = "A" if cxs[0] < 0.5 else "B"
        slots[slot] = dets[0]
        return slots

    # two detections: left/right by default, but swap if the remembered
    # positions say the fencers are the other way around
    order = np.argsort(cxs)
    left, right = dets[order[0]], dets[order[1]]
    lcx, rcx = cxs[order[0]], cxs[order[1]]
    straight = sum(abs(c - history[s]) for s, c in (("A", lcx), ("B", rcx)) if s in history)
    swapped = sum(abs(c - history[s]) for s, c in (("A", rcx), ("B", lcx)) if s in history)
    if history and swapped < straight:
        slots["A"], slots["B"] = right, left
    else:
        slots["A"], slots["B"] = left, right
    return slots


def _classify_window(model: ActionLSTM, kp_seq: np.ndarray, motion_seq: np.ndarray,
                     min_real: int) -> tuple[str | None, float]:
    """Classify one window of keypoints. Returns (label, confidence) or (None, 0)."""
    real = np.any(kp_seq.reshape(len(kp_seq), -1) != 0, axis=1)
    if real.sum() < min_real:
        return None, 0.0
    kp = kp_seq[np.argmax(real):]

    # Refuse to classify a skeleton that is mostly carried forward. When a fencer
    # walks out of frame or gets occluded, pose_pipeline holds low-visibility
    # joints at their previous position, so the window keeps arriving here as a
    # plausible-looking but frozen body. Measured on the bout: 40% of `advance`
    # calls came from windows at a frame edge or >25% frozen, and 14 of them were
    # classifying a skeleton more than HALF carried forward. Those are not
    # advances, they are missing data being labelled.
    if len(kp) >= 2:
        step = np.linalg.norm(np.diff(kp[:, :, :2], axis=0), axis=2)
        if float(np.mean(step < 1e-9)) > MAX_FROZEN_FRAC:
            return None, 0.0

    norm = _normalize_sequence(kp)
    agg = _engineered_features(norm, motion_seq)

    flat = norm.reshape(len(norm), -1).astype(np.float32)
    n_real = min(len(flat), SEQ_LEN)      # tell the model where the padding starts
    if len(flat) >= SEQ_LEN:
        flat = flat[:SEQ_LEN]
    else:
        flat = np.concatenate([flat, np.zeros((SEQ_LEN - len(flat), INPUT_SIZE), np.float32)])

    # the short window is 25 real frames padded to 60, so 58% of it is zeros --
    # without the length the model pools across that padding and reads it as a cue 
    lengths = torch.tensor([n_real])
    with torch.no_grad():
        logits = model(torch.from_numpy(flat)[None], torch.from_numpy(agg)[None], lengths)
        if logits.ndim == 3:
            # per-frame model: take the newest REAL frame, i.e. what is happening
            # now. Voting over the window would re-impose the single-label
            # assumption this model exists to avoid.
            logits = frame_logits_to_window(logits, lengths, mode="last")
        probs = torch.softmax(logits, dim=1)[0]

    if APPLY_CLASS_PRIOR:
        # train prior is uniform (inverse-frequency weighting), so dividing by it is
        # a constant and drops out of the renormalisation -- only the target matters
        w = torch.tensor([CLASS_PRIOR[c] for c in CLASS_NAMES], dtype=probs.dtype)
        probs = probs * w
        probs = probs / probs.sum().clamp(min=1e-12)

    idx = int(probs.argmax())
    return CLASS_NAMES[idx], float(probs[idx])


def _predict(model: ActionLSTM, track: FencerTrack) -> None:
    """Multi-scale prediction: short window catches fast actions (parry), long
    window reads sustained ones. A confident fast-action hit on the short window
    overrides the long-window call; otherwise the long window decides."""
    kp_full = np.stack(track.kp)
    mot_full = np.array(track.motion, dtype=np.float32)

    long_label, long_conf = _classify_window(
        model, kp_full[-WINDOW_LONG:], mot_full[-WINDOW_LONG:], MIN_REAL_FRAMES)
    short_label, short_conf = _classify_window(
        model, kp_full[-WINDOW_SHORT:], mot_full[-WINDOW_SHORT:], MIN_REAL_SHORT)

    # The override may only ADD fast-class calls, never remove them, so it is the one
    # path that can manufacture parries. Guard: a short window may not overrule a
    # MORE confident long window (a 0.65 parry should not beat a 0.95 lunge). A real
    # parry still wins easily, since a 2 s window mostly filled by the parry is
    # exactly where the long call is weak.
    # Honest sizing (bout 1, measured by actually disabling the path): the override
    # contributes 16 of fencer A's 68 parry calls and 5 of B's 22 — about a quarter.
    # This guard removes only ~4. So parry over-prediction is NOT mainly this logic;
    # it comes from the LONG window, i.e. the model. Fixing it means fixing parry
    # (one feature, wrist-speed p90 — see the parry notes in CLAUDE.md), not tuning
    # here. Kept because the pathology it blocks is real, not because it moved much.
    if (short_label in FAST_CLASSES and short_conf >= FAST_CONF
            and (long_label is None or short_conf > long_conf)):
        track.label, track.conf = short_label, short_conf
    elif long_label is not None:
        track.label, track.conf = long_label, long_conf
    if track.label is not None:
        track.counts[track.label] = track.counts.get(track.label, 0) + 1


def main() -> None:
    argv = [a for a in sys.argv[1:] if a != "--frame-model"]
    use_frame = "--frame-model" in sys.argv
    if not argv:
        sys.exit("usage: python scripts/demo_video.py path/to/video.mp4 [out.mp4] "
                 "[--frame-model]")
    video = Path(argv[0])
    if not video.exists():
        sys.exit(f"video not found: {video}")
    out_path = Path(argv[1]) if len(argv) > 1 else video.with_name(f"{video.stem}_demo.mp4")

    print("loading models...")
    person_model = load_person_model()
    if use_frame:
        # per-frame head: one call per frame, so a window can hold an advance AND
        # a lunge instead of being forced to pick one. Measured over 12 seeds:
        # bout advance 9.7% -> 14.3%, lunge 42.3% -> 31.0%, for ~3 pts of accuracy.
        if not FRAME_MODEL_PATH.exists():
            sys.exit(f"no per-frame checkpoint at {FRAME_MODEL_PATH} - train one first")
        action_model = load_action_model(FRAME_MODEL_PATH, device=torch.device("cpu"),
                                         cls=ActionFrameLSTM)
        print(f"using the per-frame model ({FRAME_MODEL_PATH.name})")
    else:
        # picks up the ensemble members if they are there, else the single checkpoint
        action_model = load_action_model(MODEL_PATH, device=torch.device("cpu"))
    blade_model = load_blade_model(BLADE_WEIGHTS) if BLADE_WEIGHTS.exists() else None
    if blade_model is None:
        print("(no blade weights found, skipping the blade tip overlay)")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"couldn't open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracks = {"A": FencerTrack(), "B": FencerTrack()}
    prev_gray = None
    pan_windows: dict = {}

    with ExitStack() as stack:
        landmarkers = {
            slot: stack.enter_context(_make_landmarker(mp.tasks.vision.RunningMode.VIDEO))
            for slot in tracks
        }
        for idx in tqdm(range(total), desc=video.stem[:40], unit="frame"):
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            pan = _frame_pan(prev_gray, gray, pan_windows)
            prev_gray = gray

            box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=MIN_BOX_H_FRAC)
            boxes = _assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)
            timestamp = int(idx * 1000 / fps)

            for slot, box in (("A", boxes["A"]), ("B", boxes["B"])):
                track = tracks[slot]
                kp = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
                crop = crop_box(frame, box)
                if crop is not None:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = landmarkers[slot].detect_for_video(
                        
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                    kp = _landmarks_to_array(result)
                    if np.any(kp):
                        # map crop coordinates back into full-frame fractions
                        x1, y1, x2, y2 = box
                        kp[:, 0] = (x1 + kp[:, 0] * (x2 - x1)) / W
                        kp[:, 1] = (y1 + kp[:, 1] * (y2 - y1)) / H
                        # carry occluded joints forward, same as training
                        low = kp[:, 3] < VISIBILITY_THRESHOLD

                        kp[low, :3] = track.prev[low, :3]
                        track.prev = kp.copy()
                        track.last_hip_x = float((kp[23, 0] + kp[24, 0]) / 2)
                
                track.kp.append(kp)
                track.motion.append((pan, track.last_hip_x))

                if idx % PREDICT_EVERY == 0 and len(track.kp) >= MIN_REAL_FRAMES:
                    _predict(action_model, track)
                # draw this fencer
                color = SLOT_COLORS[slot]
                if box is not None:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                if np.any(kp):
                    pts = np.stack([kp[:, 0] * W, kp[:, 1] * H], axis=1)
                    draw_skeleton(frame, pts)
                org = (10, 40) if slot == "A" else (W - 360, 40)
                is_action = (track.label is not None
                             and track.label not in QUIET_CLASSES
                             and track.conf >= ACTION_CONF_FLOOR)
                if is_action:
                    draw_action_label(frame, f"{slot}: {track.label}", track.conf,
                                      org=org, color=color)
                elif box is not None:
                    # tracked but not doing a scoring action -> a quiet "ready" tag
                    draw_action_label(frame, f"{slot}: ready", None, org=org, color=(150, 150, 150))

            if blade_model is not None:
                draw_blade_tip(frame, get_blade_tip(frame, blade_model))

            writer.write(frame)

    cap.release()
    writer.release()
    for slot, track in tracks.items():
        mix = ", ".join(f"{k} {v}" for k, v in sorted(track.counts.items(), key=lambda kv: -kv[1]))
        print(f"fencer {slot} label mix: {mix if mix else '(none)'}")
    print(f"\ndone -> {out_path}")


if __name__ == "__main__":
    main()
