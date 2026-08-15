"""Extract TRAINING windows from continuous footage + interval labels."""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import mediapipe as mp

import demo_video as D
from src.action_model import CLASS_NAMES
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import (N_LANDMARKS, VISIBILITY_THRESHOLD,
                               _landmarks_to_array, _make_landmarker)

OUTDIR = PROJECT / "data" / "train_continuous"
SLOT_OF = {"left": "A", "right": "B"}


def load_truth(path):
    """Interval labels -> slot -> [(start, end, label)], two-track collapsed."""
    truth = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(r for r in f if not r.startswith("#"))
        two = "footwork" in (rdr.fieldnames or [])
        for r in rdr:
            if two:
                b = r["blade"].strip()
                lab = b if b in CLASS_NAMES else r["footwork"].strip()
            else:
                lab = r["label"].strip()
            truth[SLOT_OF[r["fencer"]]].append(
                (float(r["start"]), float(r["end"]), lab))
    return truth


def window_tensors(track):
    """(flat, agg, n_real) exactly as _classify_window builds them, or None."""
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
    flat = norm.reshape(len(norm), -1).astype(np.float32)
    n_real = min(len(flat), D.SEQ_LEN)
    if len(flat) < D.SEQ_LEN:
        flat = np.concatenate(
            [flat, np.zeros((D.SEQ_LEN - len(flat), D.INPUT_SIZE), np.float32)])
    return flat[:D.SEQ_LEN], agg, n_real


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    video, labels = Path(sys.argv[1]), Path(sys.argv[2])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out = OUTDIR / f"{video.stem}.npz"

    truth = load_truth(labels)

    def truth_at(slot, t):
        for s, e, lab in truth[slot]:
            if s <= t < e:
                return lab
        return None

    person_model = load_person_model()
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tracks = {s: D.FencerTrack() for s in ("A", "B")}
    lms = {s: _make_landmarker(mp.tasks.vision.RunningMode.VIDEO).__enter__()
           for s in ("A", "B")}
    prev_gray, pan_windows = None, {}
    X, AG, LN, Y, TM, SL = [], [], [], [], [], []
    idx, skipped = 0, 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)),
                            cv2.COLOR_BGR2GRAY).astype(np.float32)
        pan = D._frame_pan(prev_gray, gray, pan_windows)
        prev_gray = gray

        # identical to demo_video's loop -- if this drifts, training optimises for
        # inputs that never occur at inference
        box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=D.MIN_BOX_H_FRAC)
        boxes = D._assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)

        for slot, box in (("A", boxes["A"]), ("B", boxes["B"])):
            t = tracks[slot]
            kp = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
            if box is not None:
                crop = crop_box(frame, box)
                if crop is not None:
                    res = lms[slot].detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                        int(idx * 1000 / fps))
                    kp = _landmarks_to_array(res)
                    x1, y1, x2, y2 = box
                    kp[:, 0] = (x1 + kp[:, 0] * (x2 - x1)) / W
                    kp[:, 1] = (y1 + kp[:, 1] * (y2 - y1)) / H
                    low = kp[:, 3] < VISIBILITY_THRESHOLD
                    kp[low, :3] = t.prev[low, :3]
                    t.prev = kp.copy()
                    t.last_hip_x = float((kp[23, 0] + kp[24, 0]) / 2)
            t.kp.append(kp)
            t.motion.append((pan, t.last_hip_x))

        if idx % D.PREDICT_EVERY == 0 and idx >= D.WINDOW_LONG:
            now = idx / fps
            for slot in ("A", "B"):
                lab = truth_at(slot, now)
                if lab is None or lab not in CLASS_NAMES:
                    continue
                tens = window_tensors(tracks[slot])
                if tens is None:
                    skipped += 1
                    continue
                flat, agg, n_real = tens
                X.append(flat); AG.append(agg); LN.append(n_real)
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
        lengths=np.array(LN, dtype=np.int64), y=np.array(Y, dtype=np.int64),
        time=np.array(TM, dtype=np.float32), slot=np.array(SL))
    c = Counter(CLASS_NAMES[i] for i in Y)
    print(f"wrote {out.name}: {len(X)} labelled windows "
          f"({skipped} skipped by the frozen/too-short gates)")
    for k in CLASS_NAMES:
        print(f"    {k:<9}{c.get(k, 0):>6}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
