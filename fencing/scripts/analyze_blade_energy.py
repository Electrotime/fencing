"""Does blade motion energy actually separate blade actions? Answer before building.

Reads the cache from blade_energy.py and scores it against the interval labels.
The point is to kill this idea cheaply if it does not work, rather than wire a
feature into the model and discover it later -- `stance_ratio` had the best AUC
of any feature tried (0.91) and still made the model WORSE, so the bar here is
"is there signal at all", not "is this good enough to ship".

Three comparisons, hardest last:

  blade-action vs QUIET (neutral/walking)   -- should be easy; if this fails the
                                               measurement itself is broken
  blade-action vs FOOTWORK (advance/retreat) -- the one that matters. During an
                                               advance the whole fencer is moving,
                                               so this is where a naive "motion"
                                               feature gets exposed as measuring
                                               nothing but overall activity
  TORSO control, same split                  -- an equal-area box on the hips. If
                                               the torso separates the classes as
                                               well as the blade box does, then the
                                               blade box is just reading body
                                               motion and the feature is worthless

AUC is reported as separability of a SINGLE number, which is not the same as what
the model would gain -- see CLAUDE.md on that distinction.

usage: py -3 scripts/analyze_blade_energy.py [labels.csv] [blade.npz]
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent

LABELS = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "data" / "labels" / "bout2_intervals.csv"
CACHE = Path(sys.argv[2]) if len(sys.argv) > 2 else PROJECT / "data" / "labels" / "2_blade.npz"
SLOT_OF = {"left": "A", "right": "B"}

BLADE_ACTION = {"parry", "lunge", "extension"}
FOOTWORK = {"advance", "retreat"}
QUIET = {"neutral", "walking"}


def auc(pos, neg):
    """Rank-based AUC, ties counted as half. NaNs dropped by the caller."""
    pos, neg = np.asarray(pos), np.asarray(neg)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), dtype=float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks within ties so a constant feature scores 0.5, not 1.0
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main() -> int:
    truth = defaultdict(list)
    with open(LABELS, encoding="utf-8") as f:
        rdr = csv.DictReader(r for r in f if not r.startswith("#"))
        two_track = "footwork" in (rdr.fieldnames or [])
        for row in rdr:
            lab = (row["blade"].strip() if two_track else row["label"].strip())
            if two_track and lab == "none":
                lab = row["footwork"].strip()
            truth[SLOT_OF[row["fencer"]]].append(
                (float(row["start"]), float(row["end"]), lab))

    d = np.load(CACHE)
    time, slot = d["time"], d["slot"]
    blade, torso, glob = d["blade"], d["torso"], d["global_e"]

    # normalise: raw energy scales with exposure, encoder noise and crowd motion.
    # ratio-to-torso is the quantity that answers "is the BLADE moving, beyond the
    # fencer moving", which is the actual question.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = blade / np.maximum(torso, 1e-6)
        blade_n = blade / np.maximum(glob, 1e-6)
        torso_n = torso / np.maximum(glob, 1e-6)
        # p99 of the PIXELS in the box, not the box mean. A thin smeared blade
        # changes a few percent of pixels hard and barely moves the mean -- the
        # mean-based first run scored AUC 0.53 (chance) against footwork, which is
        # what averaging over a mostly-background box predicts.
        has_p99 = "blade_p99" in d.files
        if has_p99:
            ratio99 = d["blade_p99"] / np.maximum(d["torso_p99"], 1e-6)
            blade99_n = d["blade_p99"] / np.maximum(glob, 1e-6)
        else:
            ratio99 = blade99_n = np.full_like(blade, np.nan)
        # v2: oriented strip differenced against the fencer's OWN torso alignment.
        # `ctrl` is the same-size strip laid across the torso, so strip/ctrl asks
        # "is the blade moving beyond whatever body motion survived the alignment" --
        # the same question ratio asked in v1, but of a measurement that can see it.
        if "strip" in d.files:
            strip, ctrl = d["strip"], d["ctrl"]
            sratio = strip / np.maximum(ctrl, 1e-6)
            sratio99 = d["strip_p99"] / np.maximum(d["ctrl_p99"], 1e-6)
            strip_n = strip / np.maximum(glob, 1e-6)
        else:
            sratio = sratio99 = strip_n = np.full_like(blade, np.nan)

    # collect per INTERVAL, not per frame: frames inside one interval are highly
    # correlated, so per-frame AUC would look far more confident than it is.
    # column order must match COLS below; index 0 is the label
    COLS = ["blade/torso", "blade/glob", "torso/glob", "p99 bl/to", "p99 bl/gl",
            "V2 str/ctrl", "V2 p99 s/c", "V2 str/glob"]
    per_interval = []
    for s in ("A", "B"):
        m_slot = slot == s
        for st, en, lab in truth[s]:
            m = m_slot & (time >= st) & (time < en)
            if m.sum() < 3:
                continue
            def hi(v):
                x = v[m]
                x = x[np.isfinite(x)]
                # a parry is a SPIKE inside the interval, so the peak is the
                # statistic of interest; p90 rather than max to shrug off one
                # bad frame from a pose glitch
                return float(np.percentile(x, 90)) if len(x) else np.nan
            per_interval.append((lab, hi(ratio), hi(blade_n), hi(torso_n),
                                 hi(ratio99), hi(blade99_n),
                                 hi(sratio), hi(sratio99), hi(strip_n)))

    print(f"{len(per_interval)} intervals with >=3 measured frames\n")
    print(f"{'class':<11}{'n':>4}" + "".join(f"{c:>13}" for c in COLS))
    by = defaultdict(list)
    for row in per_interval:
        by[row[0]].append(row[1:])
    for lab in sorted(by):
        arr = np.array(by[lab], dtype=float)
        med = np.nanmedian(arr, axis=0)
        print(f"{lab:<11}{len(arr):>4}" + "".join(f"{v:>13.2f}" for v in med))

    def pool(names, col):
        return [row[col] for row in per_interval
                if row[0] in names and np.isfinite(row[col])]

    print("\n=== separability (AUC), by interval ===")
    print(f"{'comparison':<34}" + "".join(f"{c:>13}" for c in COLS))
    for title, neg in (("blade-action vs quiet", QUIET),
                       ("blade-action vs footwork", FOOTWORK),
                       ("blade-action vs everything else", QUIET | FOOTWORK)):
        vals = [auc(pool(BLADE_ACTION, c), pool(neg, c)) for c in range(1, len(COLS) + 1)]
        print(f"{title:<34}" + "".join(f"{v:>13.2f}" for v in vals))
    print("  (col 3 is the TORSO control: if it matches the blade columns, the "
          "blade box\n   is reading body motion and the feature is worthless)")

    print("\n=== parry alone (the class this exists for) ===")
    for i, name in enumerate(COLS, start=1):
        a = auc(pool({"parry"}, i), pool(QUIET | FOOTWORK, i))
        print(f"  parry vs non-blade, {name:<12}: AUC {a:.2f}")

    n_meas = np.isfinite(blade).sum()
    print(f"\ncoverage: blade box measurable on {n_meas}/{len(blade)} tracked "
          f"fencer-frames ({n_meas / max(len(blade), 1):.0%})")
    print("compare: the blade DETECTOR fired on 1-2% of frames during parry/lunge.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
