"""Draft a two-track label CSV from model predictions, for a human to CORRECT."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import mediapipe as mp

import demo_video as D
from src.action_model import ActionLSTM, load_action_model
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import (N_LANDMARKS, VISIBILITY_THRESHOLD,
                               _landmarks_to_array, _make_landmarker)

SLOT_NAME = {"A": "left", "B": "right"}
MIN_CONF = 0.45      # below this the run is written TODO for a human to resolve
MIN_RUN = 0.40       # seconds; shorter runs are prediction flicker, not actions


def merge_runs(preds, min_run):
    """[(t, label, conf)] -> [(start, end, label, mean_conf)], consecutive equal"""
    if not preds:
        return []
    runs, cur = [], [preds[0][0], preds[0][0], preds[0][1], [preds[0][2]]]
    for t, lab, conf in preds[1:]:
        if lab == cur[2]:
            cur[1], _ = t, cur[3].append(conf)
        else:
            runs.append(cur)
            cur = [t, t, lab, [conf]]
    runs.append(cur)
    return [(s, e, lab, float(np.mean(cs))) for s, e, lab, cs in runs
            if e - s >= min_run]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=30.0)
    ap.add_argument("--end", type=float, default=330.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    video = Path(a.video)
    out = Path(a.out) if a.out else PROJECT / "data" / "labels" / f"{video.stem}_draft_2track.csv"

    person_model = load_person_model()
    action_model = load_action_model(PROJECT / "models" / "action_lstm.pth",
                                     device=torch.device("cpu"), cls=ActionLSTM)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # rewind far enough that the sliding window is already full at --start,
    # otherwise the first ~2 s of output is predicted from a partial history
    warmup = D.WINDOW_LONG / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int((a.start - warmup) * fps)))

    tracks = {s: D.FencerTrack() for s in ("A", "B")}
    lms = {s: _make_landmarker(mp.tasks.vision.RunningMode.VIDEO).__enter__()
           for s in ("A", "B")}
    prev_gray, pan_windows = None, {}
    preds = {"A": [], "B": []}
    idx = int(max(0, (a.start - warmup) * fps))

    while idx < int(a.end * fps):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        pan = D._frame_pan(prev_gray, gray, pan_windows)
        prev_gray = gray
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

        now = idx / fps
        if idx % D.PREDICT_EVERY == 0 and now >= a.start:
            for slot in ("A", "B"):
                tracks[slot].label = None
                # opponent track: both slots already have this frame appended
                D._predict(action_model, tracks[slot],
                           tracks["B" if slot == "A" else "A"])
            D._apply_parry_gate(tracks)
            for slot in ("A", "B"):
                t = tracks[slot]
                if t.label is not None:
                    preds[slot].append((now, t.label, t.conf))
        idx += 1

    cap.release()
    for s in lms.values():
        s.__exit__(None, None, None)

    rows = []
    for slot in ("A", "B"):
        for s, e, lab, conf in merge_runs(preds[slot], MIN_RUN):
            if conf < MIN_CONF:
                fw, bl = "TODO", "none"
            elif lab == "parry":
                fw, bl = "TODO", "parry"
            else:
                fw, bl = lab, "none"
            rows.append((s, e, SLOT_NAME[slot], fw, bl))
    rows.sort()

    n_todo = sum(1 for r in rows if r[3] == "TODO")
    with open(out, "w", encoding="utf-8", newline="") as f:
        f.write(f"# DRAFT for {video.name}, {a.start:.0f}-{a.end:.0f}s, generated by\n"
                f"# scripts/draft_labels.py. NOT ground truth -- the model runs at ~42%.\n"
                f"#\n"
                f"# CORRECT THIS FILE, do not trust it:\n"
                f"#  - every boundary is a guess; the model is weakest exactly where it\n"
                f"#    matters (lunge recall 9-19%, parry 0%)\n"
                f"#  - PARRIES ARE ABSENT. The blade column is `none` throughout because\n"
                f"#    the model scores 0% on parry across three bouts and two\n"
                f"#    architectures. Every parry must be added by hand.\n"
                f"#  - rows marked TODO had mean confidence below {MIN_CONF}; check_labels.py\n"
                f"#    will refuse the file until they are resolved\n"
                f"#  - anchoring is the real risk. Re-watch the action moments; do not\n"
                f"#    skim and accept.\n"
                f"#\n"
                f"# Runs shorter than {MIN_RUN}s were dropped as prediction flicker, so gaps\n"
                f"# are expected. Unlabelled time is excluded from scoring, not treated as\n"
                f"# neutral, so a gap is safe -- a WRONG label is not.\n"
                f"fencer,start,end,footwork,blade\n")
        for s, e, fencer, lab in rows:
            f.write(f"{fencer},{s:.3f},{e:.3f},{lab},none\n")

    print(f"wrote {out}")
    print(f"  {len(rows)} intervals over {a.end - a.start:.0f}s "
          f"({n_todo} marked TODO, {n_todo / max(len(rows),1):.0%})")
    print(f"  every parry still needs adding by hand -- the model contributes none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
