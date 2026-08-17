"""Minimal end-to-end demo: run the whole stack on a video and save an annotated copy."""

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
                              N_AGG_WIDE, SEQ_LEN, _engineered_features,
                              frame_logits_to_window, load_action_model, wide_agg)
from src.blade_detector import get_blade_tip, load_blade_model
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import (N_LANDMARKS, VISIBILITY_THRESHOLD,
                               _landmarks_to_array, _make_landmarker,
                               _normalize_sequence)
from src.utils import draw_action_label, draw_blade_tip, draw_skeleton

# seven bouts, four venues, mirror-augmented; see CLAUDE.md.
MODEL_PATH = PROJECT_ROOT / "models" / "action_mirror7.pth"
POOL_MODE = "last"
USE_OPPONENT = True
FRAME_MODEL_PATH = PROJECT_ROOT / "models" / "action_frame.pth"  # --frame-model
BLADE_WEIGHTS = (PROJECT_ROOT / "models" / "blade_yolo" / "fencing_blade_v2"
                 / "weights" / "best.pt")

PREDICT_EVERY = 5     # frames between action predictions (labels hold in between)
MIN_REAL_FRAMES = 15  # don't guess an action until we've tracked this many frames

WINDOW_LONG = SEQ_LEN   # 60 frames / 2 s
WINDOW_SHORT = 25       # ~0.8 s
MIN_REAL_SHORT = 12
FAST_CLASSES = {"parry"}   # brief actions the short window is allowed to override with
FAST_CONF = 0.65           # a short-window override has to be at least this sure
# quiet states: not fencing actions, so the overlay stays blank on them (this is why
# neutral/walking exist -- labels only light up when something real happens)
QUIET_CLASSES = {"neutral", "walking"}
ACTION_CONF_FLOOR = 0.50   # below this the call is too unsure to show as an action
APPLY_CLASS_PRIOR = False
# Pooled duration shares from BOTH labelled bouts (218 s). Re-derive with
# scripts/estimate_class_prior.py whenever more footage is labelled.
CLASS_PRIOR = {"advance": 0.184, "lunge": 0.045, "parry": 0.017,
               "retreat": 0.122, "neutral": 0.230, "walking": 0.401}

PARRY_NEEDS_ATTACKER = True
PARRY_OPP_LUNGE_MIN = 0.20  # best-or-tied on both bouts; precision is flat 0.2-0.5 on
                            # bout 4 and still rising on bout 5, so this is the safe end

PARRY_PROMOTE = True
PARRY_PROMOTE_MIN = 0.15      # own parry probability, chosen on bout 5, confirmed on 4
PARRY_PROMOTE_OPP_MIN = 0.60  # opponent lunge -- far above the veto's 0.20, because
PARRY_LAMP_COLOR = (0, 165, 255)   # amber (BGR)
PARRY_LAMP_DY = 44                 # pixels below the footwork line

MAX_FROZEN_FRAC = 0.25  # skip the window if more than this share of joint steps are
MIN_BOX_H_FRAC = 0.25 # ignore "people" shorter than this fraction of frame height.
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
        self.probs: np.ndarray | None = None  # long-window distribution, for the parry gate
        self.footwork: str | None = None
        self.footwork_conf = 0.0
        self.counts: dict[str, int] = {}  # how often each label got emitted


def _frame_pan(prev_gray: np.ndarray | None, gray: np.ndarray,
               windows: dict, strip_frac: float | None = None) -> float:
    """Horizontal background shift between two frames, from L/R border strips."""
    if prev_gray is None:
        return 0.0
    h, w = gray.shape
    strip_w = max(10, int((PAN_STRIP_FRAC if strip_frac is None else strip_frac) * w))
    rows = slice(int(0.10 * h), int(0.75 * h))  # skip broadcast graphics / scoreboard
    key = f"win{strip_w}"
    if key not in windows:
        windows[key] = cv2.createHanningWindow((strip_w, rows.stop - rows.start), cv2.CV_32F)
    shifts = []
    for a, b in [(0, strip_w), (w - strip_w, w)]:
        (dx, _), response = cv2.phaseCorrelate(prev_gray[rows, a:b], gray[rows, a:b],
                                               windows[key])
        if response > PAN_MIN_RESPONSE:
            shifts.append(dx)
    return float(np.median(shifts)) if shifts else 0.0


def _assign_boxes(dets: list[np.ndarray], tracks: dict[str, FencerTrack],
                  W: int) -> dict[str, np.ndarray | None]:
    """Match this frame's detections to the A/B tracks by continuity."""
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

    order = np.argsort(cxs)
    slots["A"], slots["B"] = dets[order[0]], dets[order[1]]
    return slots


def _window_inputs(kp_seq: np.ndarray, motion_seq: np.ndarray, min_real: int):
    """(flat, agg, n_real) for one fencer's window, or None if it fails the gates."""
    real = np.any(kp_seq.reshape(len(kp_seq), -1) != 0, axis=1)
    if real.sum() < min_real:
        return None
    kp = kp_seq[np.argmax(real):]
    if len(kp) >= 2:
        step = np.linalg.norm(np.diff(kp[:, :, :2], axis=0), axis=2)
        if float(np.mean(step < 1e-9)) > MAX_FROZEN_FRAC:
            return None
    norm = _normalize_sequence(kp)
    agg = _engineered_features(norm, motion_seq)
    flat = norm.reshape(len(norm), -1).astype(np.float32)
    n_real = min(len(flat), SEQ_LEN)
    if len(flat) >= SEQ_LEN:
        flat = flat[:SEQ_LEN]
    else:
        flat = np.concatenate([flat, np.zeros((SEQ_LEN - len(flat), INPUT_SIZE), np.float32)])
    return flat, agg, n_real


def _classify_window(model: ActionLSTM, kp_seq: np.ndarray, motion_seq: np.ndarray,
                     min_real: int,
                     opp: tuple | None = None) -> tuple[str | None, float, np.ndarray | None]:
    """Classify one window of keypoints. Returns (label, confidence) or (None, 0)."""
    got = _window_inputs(kp_seq, motion_seq, min_real)
    if got is None:
        return None, 0.0, None
    flat, agg, n_real = got

    if USE_OPPONENT:
        opp_got = _window_inputs(*opp, min_real) if opp is not None else None
        agg = wide_agg(agg, None if opp_got is None else opp_got[1])

    # the short window is 25 real frames padded to 60, so 58% of it is zeros --
    # without the length the model pools across that padding and reads it as a cue
    lengths = torch.tensor([n_real])
    with torch.no_grad():
        logits = model(torch.from_numpy(flat)[None], torch.from_numpy(agg)[None], lengths)
        if logits.ndim == 3:
            logits = frame_logits_to_window(logits, lengths, mode="last")
        probs = torch.softmax(logits, dim=1)[0]

    if APPLY_CLASS_PRIOR:
        # train prior is uniform (inverse-frequency weighting), so dividing by it is
        # a constant and drops out of the renormalisation -- only the target matters
        w = torch.tensor([CLASS_PRIOR[c] for c in CLASS_NAMES], dtype=probs.dtype)
        probs = probs * w
        probs = probs / probs.sum().clamp(min=1e-12)

    idx = int(probs.argmax())
    # the full vector goes back too: the parry gate needs the OPPONENT's lunge
    # probability, not just their argmax label
    return CLASS_NAMES[idx], float(probs[idx]), probs.numpy()


def _predict(model: ActionLSTM, track: FencerTrack,
             opp_track: FencerTrack | None = None) -> None:
    """Multi-scale prediction: short window catches fast actions (parry), long"""
    kp_full = np.stack(track.kp)
    mot_full = np.array(track.motion, dtype=np.float32)
    if opp_track is not None and len(opp_track.kp) == len(track.kp):
        okp = np.stack(opp_track.kp)
        omot = np.array(opp_track.motion, dtype=np.float32)
    else:
        # length mismatch means the two tracks are not frame-aligned, and a
        # misaligned opponent is worse than none at all
        okp = omot = None

    def opp_slice(n):
        return None if okp is None else (okp[-n:], omot[-n:])

    long_label, long_conf, long_probs = _classify_window(
        model, kp_full[-WINDOW_LONG:], mot_full[-WINDOW_LONG:], MIN_REAL_FRAMES,
        opp_slice(WINDOW_LONG))
    short_label, short_conf, _ = _classify_window(
        model, kp_full[-WINDOW_SHORT:], mot_full[-WINDOW_SHORT:], MIN_REAL_SHORT,
        opp_slice(WINDOW_SHORT))
    track.probs = long_probs

    if (short_label in FAST_CLASSES and short_conf >= FAST_CONF
            and (long_label is None or short_conf > long_conf)):
        track.label, track.conf = short_label, short_conf
    elif long_label is not None:
        track.label, track.conf = long_label, long_conf
    if track.label is not None:
        track.counts[track.label] = track.counts.get(track.label, 0) + 1


def _apply_parry_gate(tracks: dict[str, FencerTrack]) -> None:
    """Couple the two fencers' parry decisions to the OPPONENT's attack, both ways."""
    lunge_i, parry_i = CLASS_NAMES.index("lunge"), CLASS_NAMES.index("parry")
    for slot, track in tracks.items():
        track.footwork, track.footwork_conf = None, 0.0
        if track.probs is None:
            continue
        opp = tracks.get("B" if slot == "A" else "A")
        opp_attack = 0.0 if opp is None or opp.probs is None else float(opp.probs[lunge_i])

        alt = track.probs.copy()
        alt[parry_i] = -1.0
        fw_i = int(alt.argmax())
        rest = 1.0 - float(track.probs[parry_i])
        fw_name = CLASS_NAMES[fw_i]
        fw_conf = float(track.probs[fw_i] / rest) if rest > 1e-6 else 0.0

        if track.label == "parry":
            track.footwork, track.footwork_conf = fw_name, fw_conf
            if not PARRY_NEEDS_ATTACKER:
                continue
            if opp_attack >= PARRY_OPP_LUNGE_MIN:
                continue
            # gate rejected it: demote to the runner-up, and no lamp to light
            track.label, track.conf = fw_name, float(alt[fw_i])
            track.footwork, track.footwork_conf = None, 0.0
            continue

        # ---- the other direction: PROMOTE a parry that lost the argmax ----------
        if not PARRY_PROMOTE or int(track.probs.argmax()) == parry_i:
            continue
        if (float(track.probs[parry_i]) >= PARRY_PROMOTE_MIN
                and opp_attack >= PARRY_PROMOTE_OPP_MIN):
            track.footwork, track.footwork_conf = fw_name, fw_conf
            track.label, track.conf = "parry", float(track.probs[parry_i])


def _self_test_parry_gate() -> None:
    """The gate must fire only on parry, and only when the opponent is not attacking."""
    def mk(label, probs):
        t = FencerTrack()
        t.label, t.probs = label, np.array(probs, dtype=np.float32)
        t.conf = float(t.probs.max())
        return t

    def probs(**kw):
        v = np.full(len(CLASS_NAMES), 0.01, dtype=np.float32)
        for k, x in kw.items():
            v[CLASS_NAMES.index(k)] = x
        return v

    # opponent clearly lunging -> parry SURVIVES
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry", tr["A"].label

    # opponent NOT attacking -> parry demoted to its own runner-up
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.02))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label
    assert abs(tr["A"].conf - 0.3) < 1e-5

    # a NON-parry label is never touched, however quiet the opponent
    tr = {"A": mk("lunge", probs(lunge=0.7)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.0))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "lunge"

    # no opponent distribution at all -> treated as "nobody attacking", parry dropped
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)), "B": FencerTrack()}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    # exactly at threshold survives (>=, not >)
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.5, lunge=PARRY_OPP_LUNGE_MIN))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry"
    tr = {"A": mk("parry", probs(parry=0.5, retreat=0.3, walking=0.1)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry", "scoring label must not change"
    assert tr["A"].footwork == "retreat", tr["A"].footwork
    assert abs(tr["A"].footwork_conf - 0.6) < 1e-4, tr["A"].footwork_conf

    # a REJECTED parry lights no lamp and leaves no footwork field behind
    tr = {"A": mk("parry", probs(parry=0.5, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.0))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat" and tr["A"].footwork is None, vars(tr["A"])

    tr = {"A": mk("retreat", probs(retreat=0.5, parry=0.25)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry", tr["A"].label
    assert tr["A"].footwork == "retreat", tr["A"].footwork
    assert abs(tr["A"].conf - 0.25) < 1e-5, tr["A"].conf

    tr = {"A": mk("retreat", probs(retreat=0.5, parry=0.25)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.02))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat" and tr["A"].footwork is None, vars(tr["A"])

    # own parry probability below PARRY_PROMOTE_MIN -> not promoted however hard the
    # opponent attacks. Context alone must never manufacture a parry.
    tr = {"A": mk("retreat", probs(retreat=0.7, parry=0.10)),
          "B": mk("lunge", probs(lunge=0.95))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.0))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    # no opponent distribution -> conservative in BOTH directions: never promotes
    tr = {"A": mk("retreat", probs(retreat=0.5, parry=0.25)), "B": FencerTrack()}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    # a non-parry call never gets a lamp, and stale fields are cleared between frames
    tr = {"A": mk("parry", probs(parry=0.5, retreat=0.3)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].footwork == "retreat"
    tr["A"].label, tr["A"].probs = "advance", probs(advance=0.9)
    _apply_parry_gate(tr)
    assert tr["A"].footwork is None, "stale footwork survived into a non-parry frame"

    print("self-test ok: parry gate fires only on parry, only without an attacker; "
          "two-indicator fields set without touching the scoring label")


def _self_test_assign() -> None:
    """Slot assignment must be MEMORYLESS for two detections. Run with --self-test."""
    def box(cx, w=100):
        return np.array([cx - w / 2, 0, cx + w / 2, 300], dtype=np.float32)

    def cx_of(b, W=1000):
        return None if b is None else round(float((b[0] + b[2]) / 2 / W), 3)

    def tracks_at(pos):
        tr = {s: FencerTrack() for s in ("A", "B")}
        for s, hx in pos.items():
            tr[s].kp.append(np.ones((N_LANDMARKS, 4), dtype=np.float32))
            tr[s].last_hip_x = hx
        return tr

    dets = [box(250), box(750)]
    # the critical case: history says the fencers are the OTHER way round, which is
    # physically impossible on a piste and used to trigger a swap. x-order wins.
    out = _assign_boxes(dets, tracks_at({"A": 0.75, "B": 0.25}), 1000)
    assert (cx_of(out["A"]), cx_of(out["B"])) == (0.25, 0.75), "history overrode x-order"

    out = _assign_boxes(dets, tracks_at({"A": 0.25, "B": 0.75}), 1000)
    assert (cx_of(out["A"]), cx_of(out["B"])) == (0.25, 0.75), "agreeing history broke it"

    out = _assign_boxes(dets, {s: FencerTrack() for s in ("A", "B")}, 1000)
    assert (cx_of(out["A"]), cx_of(out["B"])) == (0.25, 0.75), "no history broke it"

    # a single detection still needs history -- x-order says nothing about WHICH
    # fencer is visible, so that path deliberately stays history-dependent
    out = _assign_boxes([box(700)], tracks_at({"A": 0.20, "B": 0.75}), 1000)
    assert out["B"] is not None and out["A"] is None, "lone box went to the wrong slot"

    print("self-test ok: two-box assignment is memoryless, lone box uses history")


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test_assign()
        _self_test_parry_gate()
        return
    def _opt(name, default=None):
        for i, a in enumerate(sys.argv):
            if a == name and i + 1 < len(sys.argv):
                return float(sys.argv[i + 1])
            if a.startswith(f"{name}="):
                return float(a.split("=", 1)[1])
        return default

    def _opt_str(name):
        for i, a in enumerate(sys.argv):
            if a == name and i + 1 < len(sys.argv):
                return sys.argv[i + 1]
            if a.startswith(f"{name}="):
                return a.split("=", 1)[1]
        return None

    start_s, end_s = _opt("--start"), _opt("--end")
    model_arg = _opt_str("--model")
    _skip = {str(v) for v in (start_s, end_s) if v is not None}
    if model_arg:
        _skip.add(model_arg)
    argv = [a for a in sys.argv[1:]
            if a != "--frame-model" and not a.startswith("--") and a not in _skip]
    use_frame = "--frame-model" in sys.argv
    if not argv:
        sys.exit("usage: python scripts/demo_video.py path/to/video.mp4 [out.mp4] "
                 "[--start S] [--end S] [--frame-model | --self-test]")
    video = Path(argv[0])
    if not video.exists():
        sys.exit(f"video not found: {video}")
    out_path = Path(argv[1]) if len(argv) > 1 else video.with_name(f"{video.stem}_demo.mp4")

    print("loading models...")
    person_model = load_person_model()
    if use_frame:
        if not FRAME_MODEL_PATH.exists():
            sys.exit(f"no per-frame checkpoint at {FRAME_MODEL_PATH} - train one first")
        action_model = load_action_model(FRAME_MODEL_PATH, device=torch.device("cpu"),
                                         cls=ActionFrameLSTM)
        print(f"using the per-frame model ({FRAME_MODEL_PATH.name})")
    else:
        mpath = MODEL_PATH
        if model_arg:
            mpath = Path(model_arg)
            if not mpath.exists():
                mpath = PROJECT_ROOT / "models" / mpath.name
            if not mpath.exists():
                sys.exit(f"no checkpoint at {model_arg}")
            print(f"model override: {mpath.name}")
        # picks up the ensemble members if they are there, else the single checkpoint
        action_model = load_action_model(
            mpath, device=torch.device("cpu"),
            cls=lambda: ActionLSTM(pool=POOL_MODE,
                                   n_agg=N_AGG_WIDE if USE_OPPONENT else 6))
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
    first = int(round((start_s or 0.0) * fps))
    last = int(round(end_s * fps)) if end_s is not None else total
    last = min(last, total)
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    n_frames = max(0, last - first)
    if start_s is not None or end_s is not None:
        print(f"segment {first / fps:.1f}s -> {last / fps:.1f}s "
              f"({n_frames} frames, {n_frames / fps:.1f}s)")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracks = {"A": FencerTrack(), "B": FencerTrack()}
    prev_gray = None
    pan_windows: dict = {}

    with ExitStack() as stack:
        landmarkers = {
            slot: stack.enter_context(_make_landmarker(mp.tasks.vision.RunningMode.VIDEO))
            for slot in tracks
        }
        for idx in tqdm(range(n_frames), desc=video.stem[:40], unit="frame"):
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            pan = _frame_pan(prev_gray, gray, pan_windows)
            prev_gray = gray

            box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=MIN_BOX_H_FRAC)
            boxes = _assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)
            timestamp = int(idx * 1000 / fps)

            drawn: dict[str, np.ndarray] = {}
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
                drawn[slot] = kp

            if idx % PREDICT_EVERY == 0:
                for slot in ("A", "B"):
                    track = tracks[slot]
                    if len(track.kp) >= MIN_REAL_FRAMES:
                        _predict(action_model, track, tracks["B" if slot == "A" else "A"])
                # both fencers now hold a distribution for THIS frame, which is what
                # the parry gate needs -- it asks whether the other one is attacking
                _apply_parry_gate(tracks)

            for slot, box in (("A", boxes["A"]), ("B", boxes["B"])):
                track = tracks[slot]
                kp = drawn[slot]
                # draw this fencer
                color = SLOT_COLORS[slot]
                if box is not None:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                if np.any(kp):
                    pts = np.stack([kp[:, 0] * W, kp[:, 1] * H], axis=1)
                    draw_skeleton(frame, pts)
                org = (10, 40) if slot == "A" else (W - 360, 40)
                if track.label == "parry" and track.footwork is not None:
                    quiet = (track.footwork in QUIET_CLASSES
                             or track.footwork_conf < ACTION_CONF_FLOOR)
                    if quiet:
                        draw_action_label(frame, f"{slot}: ready", None,
                                          org=org, color=(150, 150, 150))
                    else:
                        draw_action_label(frame, f"{slot}: {track.footwork}",
                                          track.footwork_conf, org=org, color=color)
                    draw_action_label(frame, "parry", track.conf,
                                      org=(org[0], org[1] + PARRY_LAMP_DY),
                                      color=PARRY_LAMP_COLOR)
                    continue
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
