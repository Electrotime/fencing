"""Draft a two-track label CSV from model predictions, for a human to CORRECT.

Labelling 17 minutes of continuous footage from scratch is hours of work, and
most of it is boring: walking back to en garde, stoppages, resets. The model is
adequate at exactly that part (walking precision 65-67% across three bouts) and
poor at the interesting part. So it drafts the boring majority and the human
fixes the actions -- much cheaper than starting from an empty file.

WHAT THIS IS NOT: ground truth. The model runs at ~42% overall. Every action
boundary in here is a guess, and anything the model cannot do at all (parry: 0%
across three bouts and two architectures) will simply be ABSENT. Two safeguards:

  - runs whose mean confidence is below MIN_CONF are written as `TODO`, which
    check_labels.py already refuses to pass, so they cannot be forgotten
  - the blade column is left at `none` throughout. Parries MUST be added by hand;
    the model contributes nothing there and pretending otherwise would bake a
    known-blind spot into the labels

ANCHORING IS THE REAL RISK. Correcting a draft biases you toward accepting what
is written. Treat every non-quiet row as unverified, and re-watch the action
moments rather than skimming them.

usage:
  py -3 scripts/draft_labels.py "data/raw_video/Bout #1 without break (1).mp4" \
      --start 30 --end 330 --out data/labels/bout4_draft_2track.csv
"""
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
    """[(t, label, conf)] -> [(start, end, label, mean_conf)], consecutive equal
    labels merged. Runs shorter than min_run are dropped rather than emitted --
    they are almost always single-window flicker, and a label file full of 0.15 s
    intervals is harder to correct than one with gaps."""
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
                # `parry` is a BLADE value in this schema, never footwork -- that
                # separation is the entire point of two tracks. The model emits a
                # single label, so a parry call says nothing about the footwork
                # underneath, which has to be filled in by hand. (In all three
                # labelled bouts that footwork was `retreat`, 11 times out of 11 in
                # bout 3 -- but do not let the draft assume it.)
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
