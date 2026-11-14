"""Extract TRAINING windows from continuous footage + interval labels."""
import argparse
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import mediapipe as mp

import demo_video as D
from src.action_model import CLASS_NAMES
from src.labels import load_intervals
from src.person_detector import get_fencer_boxes, load_person_model
from src.pose_pipeline import _make_landmarker

OUTDIR = PROJECT / "data" / "train_continuous"
PAN_WIDE_FRAC = 0.44      # the wider aperture benchmarked in pan_benchmark.py


def window_tensors(track, mot_alt=None):
    """(flat, agg, n_real, agg_alt) as _classify_window builds them, or None."""
    kp_seq = np.stack(track.kp)[-D.WINDOW_LONG:]
    mot = np.array(track.motion, dtype=np.float32)[-D.WINDOW_LONG:]
    real = np.any(kp_seq.reshape(len(kp_seq), -1) != 0, axis=1)
    if real.sum() < D.MIN_REAL_FRAMES:
        return None
    kp = kp_seq[np.argmax(real):]
    if len(kp) >= 2:
        step = np.linalg.norm(np.diff(kp[:, :, :2], axis=0), axis=2)
        if float(np.mean(step < 1e-9)) > D.MAX_FROZEN_FRAC:
            return None
    norm = D._normalize_sequence(kp)
    agg = D._engineered_features(norm, mot)
    agg_alt = None
    if mot_alt is not None:
        agg_alt = D._engineered_features(
            norm, np.array(mot_alt, dtype=np.float32)[-D.WINDOW_LONG:])
    flat = norm.reshape(len(norm), -1).astype(np.float32)
    n_real = min(len(flat), D.SEQ_LEN)
    if len(flat) < D.SEQ_LEN:
        flat = np.concatenate(
            [flat, np.zeros((D.SEQ_LEN - len(flat), D.INPUT_SIZE), np.float32)])
    return flat[:D.SEQ_LEN], agg, n_real, agg_alt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("labels", type=Path)
    ap.add_argument("--out", type=Path,
                    help="default data/train_continuous/<video stem>.npz")
    ap.add_argument("--no-fps-normalise", action="store_true",
                    help="keep every frame, even above 30 fps")
    a = ap.parse_args()
    video = a.video
    out = a.out or OUTDIR / f"{video.stem}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)

    truth, _ = load_intervals(a.labels)

    def truth_at(slot, t):
        for s, e, lab in truth.get(slot, ()):
            if s <= t < e:
                return lab
        return None

    person_model = load_person_model()
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        print(f"couldn't open {video}")
        return 1
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # the window is 60 FRAMES; above 30 fps keep every Nth so it still spans 2 s
    stride = 1 if a.no_fps_normalise else D.fps_stride(fps)
    if stride > 1:
        print(f"fps-normalise: {fps:.1f} fps, keeping every {stride} frames")

    tracks = {s: D.FencerTrack() for s in ("A", "B")}
    lms = {s: _make_landmarker(mp.tasks.vision.RunningMode.VIDEO).__enter__()
           for s in ("A", "B")}
    prev_gray, pan_windows = None, {}
    X, AG, LN, Y, TM, SL, AGW = [], [], [], [], [], [], []
    wide_motion = {s: deque(maxlen=D.SEQ_LEN) for s in ("A", "B")}
    PAN_TRACK = []
    idx, raw_idx, skipped = 0, -1, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        raw_idx += 1
        if raw_idx % stride:
            continue
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        pan = D._frame_pan(prev_gray, gray, pan_windows)
        pan_wide = D._frame_pan(prev_gray, gray, pan_windows, PAN_WIDE_FRAC)
        prev_gray = gray
        PAN_TRACK.append((pan, pan_wide))

        # the demo's own detection and tracking step, so training sees what inference does
        box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=D.MIN_BOX_H_FRAC)
        boxes = D._assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)

        D.track_frame(frame, boxes, tracks, lms, pan, int(raw_idx * 1000 / fps))
        for slot in ("A", "B"):
            wide_motion[slot].append((pan_wide, tracks[slot].last_hip_x))

        if idx % D.PREDICT_EVERY == 0 and idx >= D.WINDOW_LONG:
            now = raw_idx / fps
            for slot in ("A", "B"):
                lab = truth_at(slot, now)
                if lab is None or lab not in CLASS_NAMES:
                    continue
                tens = window_tensors(tracks[slot], wide_motion[slot])
                if tens is None:
                    skipped += 1
                    continue
                flat, agg, n_real, agg_wide = tens
                X.append(flat); AG.append(agg); LN.append(n_real); AGW.append(agg_wide)
                Y.append(CLASS_NAMES.index(lab)); TM.append(now); SL.append(slot)
        idx += 1

    cap.release()
    for s in lms.values():
        s.__exit__(None, None, None)

    if not X:
        print("no labelled windows extracted -- check the label file lines up "
              "with this video")
        return 1

    np.savez_compressed(
        out,
        X=np.stack(X).astype(np.float32), agg=np.stack(AG).astype(np.float32),
        agg_wide=np.stack(AGW).astype(np.float32),
        lengths=np.array(LN, dtype=np.int64), y=np.array(Y, dtype=np.int64),
        time=np.array(TM, dtype=np.float32), slot=np.array(SL),
        pan_track=np.array(PAN_TRACK, dtype=np.float32))
    c = Counter(CLASS_NAMES[i] for i in Y)
    print(f"wrote {out.name}: {len(X)} labelled windows "
          f"({skipped} skipped by the frozen/too-short gates)")
    for k in CLASS_NAMES:
        print(f"    {k:<9}{c.get(k, 0):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
