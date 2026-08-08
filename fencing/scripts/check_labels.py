"""Validate an interval-label CSV before it gets used for training or scoring.

Written so Aaron can check a label file WHILE writing it, rather than finding a
typo after the whole bout is done. Accepts both schemas:

  single-track (bout1/bout2):  fencer,start,end,label
  two-track    (new):          fencer,start,end,footwork,blade

The two-track schema exists because the classes were never mutually exclusive --
a fencer parries WHILE retreating, which is why `parry` competed with `retreat`
for one slot and lost every time. Footwork and blade are separate questions and
get separate columns.

Checks, in rough order of how much damage each mistake causes:
  1. unknown class names (a typo silently becomes a class nobody trains on)
  2. start >= end, or negative times
  3. OVERLAPPING intervals for the same fencer -- truth_at() takes the first
     match, so an overlap means some windows are scored against a label you
     didn't intend
  4. intervals past the end of the video (only if the video is present)
  5. left/right coverage imbalance, reported not errored -- deliberate gaps are
     normal and excluded from scoring

usage: py -3 scripts/check_labels.py data/labels/bout3_intervals.csv [video.mp4]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
from src.action_model import CLASS_NAMES

# `extension` is not a predictable class (it was dropped and lives on as the
# arm-reach FEATURE) but it is a legal thing to WRITE DOWN -- the scorer counts
# and excludes it. Same idea for `none` in the blade column.
#
# In the TWO-TRACK schema `parry` is a blade value only -- allowing it as footwork
# too would rebuild the exact collision the split exists to remove: a fencer
# parrying while retreating needs both cells filled, not a choice between them.
# In the SINGLE-track schema `parry` is of course a legal label; that is the whole
# problem being migrated away from, not a typo to reject.
SINGLE_OK = set(CLASS_NAMES) | {"extension"}
FOOTWORK_OK = set(CLASS_NAMES) - {"parry"}
BLADE_OK = {"parry", "extension", "none", "attack", "beat"}
FENCER_OK = {"left", "right"}
TODO = "TODO"   # written by upgrade_labels.py where it refused to guess


def load(path):
    """Rows plus which schema the header declares."""
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(r for r in f if not r.startswith("#"))
        rows = [dict(r) for r in rdr]
        cols = set(rdr.fieldnames or [])
    two_track = "footwork" in cols
    return rows, cols, two_track


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    rows, cols, two_track = load(path)

    errors, warnings, n_todo = [], [], 0
    required = ({"fencer", "start", "end", "footwork", "blade"} if two_track
                else {"fencer", "start", "end", "label"})
    missing = required - cols
    if missing:
        print(f"FATAL: header is missing {sorted(missing)}")
        print(f"       got {sorted(cols)}")
        return 1

    print(f"{path.name}: {len(rows)} rows, "
          f"{'TWO-TRACK (footwork+blade)' if two_track else 'single-track'} schema")

    # duration is only knowable if the video is here; skip the check otherwise
    # rather than guessing, since a wrong duration would flag valid rows.
    duration = None
    if len(sys.argv) > 2:
        import cv2
        cap = cv2.VideoCapture(sys.argv[2])
        n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if fps:
            duration = n / fps
            print(f"video {Path(sys.argv[2]).name}: {duration:.1f}s")

    by_fencer = defaultdict(list)
    for i, r in enumerate(rows, start=2):   # +2: header is line 1, 1-indexed
        where = f"line {i}"
        fencer = r["fencer"].strip()
        if fencer not in FENCER_OK:
            errors.append(f"{where}: fencer {fencer!r} not in {sorted(FENCER_OK)}")
            continue
        try:
            s, e = float(r["start"]), float(r["end"])
        except ValueError:
            errors.append(f"{where}: start/end not numeric ({r['start']!r}, {r['end']!r})")
            continue
        if s < 0:
            errors.append(f"{where}: negative start {s}")
        if e <= s:
            errors.append(f"{where}: end {e} is not after start {s}")
        if duration and e > duration + 0.5:
            errors.append(f"{where}: end {e:.2f}s is past the video ({duration:.1f}s)")

        if two_track:
            fw, bl = r["footwork"].strip(), r["blade"].strip()
            # TODO is unfinished, not wrong -- report it as a count so a
            # half-labelled file gives one honest number instead of N errors.
            if fw == TODO or bl == TODO:
                n_todo += 1
            if fw not in FOOTWORK_OK and fw != TODO:
                errors.append(f"{where}: footwork {fw!r} not in {sorted(FOOTWORK_OK)}"
                              + (" (parry is a BLADE value in this schema)"
                                 if fw == "parry" else ""))
            if bl not in BLADE_OK and bl != TODO:
                errors.append(f"{where}: blade {bl!r} not in {sorted(BLADE_OK)}")
            tag = f"{fw}/{bl}"
        else:
            lab = r["label"].strip()
            if lab not in SINGLE_OK:
                errors.append(f"{where}: label {lab!r} not in {sorted(SINGLE_OK)}")
            tag = lab
        by_fencer[fencer].append((s, e, tag, i))

    # overlaps: the scorer's truth_at() returns the FIRST interval containing t,
    # so an accidental overlap silently scores windows against the wrong label.
    for fencer, iv in by_fencer.items():
        iv.sort()
        for (s1, e1, t1, l1), (s2, e2, t2, l2) in zip(iv, iv[1:]):
            if s2 < e1:
                errors.append(f"{fencer}: lines {l1} and {l2} OVERLAP "
                              f"({s1:.2f}-{e1:.2f} {t1}) vs ({s2:.2f}-{e2:.2f} {t2})")

    print()
    for f in sorted(by_fencer):
        iv = by_fencer[f]
        covered = sum(e - s for s, e, _, _ in iv)
        print(f"  {f:<6} {len(iv):>3} intervals, {covered:>7.1f}s covered"
              + (f" ({covered / duration:.0%} of video)" if duration else ""))
    if len(by_fencer) == 2:
        a, b = (sum(e - s for s, e, _, _ in by_fencer[f]) for f in ("left", "right"))
        if a and b and max(a, b) / min(a, b) > 1.5:
            warnings.append(f"left covers {a:.0f}s but right covers {b:.0f}s -- "
                            f"intended? per-fencer accuracy is compared directly")

    counts = defaultdict(int)
    for iv in by_fencer.values():
        for s, e, tag, _ in iv:
            counts[tag] += 1
    print(f"\n  {'class':<20}{'n':>5}{'median s':>11}")
    durs = defaultdict(list)
    for iv in by_fencer.values():
        for s, e, tag, _ in iv:
            durs[tag].append(e - s)
    for tag in sorted(counts):
        d = sorted(durs[tag])
        print(f"  {tag:<20}{counts[tag]:>5}{d[len(d) // 2]:>11.2f}")

    if warnings:
        print()
        for w in warnings:
            print(f"  WARN  {w}")
    print()
    if errors:
        for e_ in errors:
            print(f"  ERROR {e_}")
        print(f"\n{len(errors)} error(s) -- fix before using this file")
        return 1
    if n_todo:
        # exit 1: the file is valid but not yet usable, and a green check here
        # would be a lie the training script would then act on.
        print(f"no errors, but {n_todo} row(s) still contain {TODO} -- not ready to use")
        return 1
    print("no errors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
