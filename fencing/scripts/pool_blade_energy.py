"""Pool blade-energy intervals across bouts and score v1 against v2 ONCE."""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.labels import SLOT_OF  # noqa: F401  (kept so the slot convention is shared)

BOUTS = {"3": "bout3_intervals_2track.csv",
         "4": "bout4_intervals_2track.csv",
         "5": "bout5_intervals_2track.csv"}
LAB = PROJECT / "data" / "labels"
QUIET = {"neutral", "walking"}
FOOTWORK = {"advance", "retreat"}
BLADE_ACTION = {"parry", "lunge", "extension"}


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg])
    order = v.argsort()
    ranks = np.empty(len(v), float)
    ranks[order] = np.arange(1, len(v) + 1)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def load(stem, csv_name):
    """Per-interval feature rows for one bout: (bout, label, v1, v1_99, v2, v2_99)."""
    import csv as _csv
    cache = LAB / f"{stem}_blade.npz"
    if not cache.exists():
        return []
    d = np.load(cache)
    if "strip" not in d.files:
        print(f"  bout {stem}: cache predates v2, skipping")
        return []
    time, slot = d["time"], d["slot"]
    with np.errstate(invalid="ignore", divide="ignore"):
        v1 = d["blade"] / np.maximum(d["torso"], 1e-6)
        v1_99 = d["blade_p99"] / np.maximum(d["torso_p99"], 1e-6)
        v2 = d["strip"] / np.maximum(d["ctrl"], 1e-6)
        v2_99 = d["strip_p99"] / np.maximum(d["ctrl_p99"], 1e-6)

    truth = defaultdict(list)
    with open(LAB / csv_name, encoding="utf-8") as f:
        for r in _csv.DictReader(x for x in f if not x.startswith("#")):
            lab = r["blade"].strip()
            if lab == "none":
                lab = r["footwork"].strip()
            truth[{"left": "A", "right": "B"}[r["fencer"]]].append(
                (float(r["start"]), float(r["end"]), lab))

    rows = []
    for s in ("A", "B"):
        ms = slot == s
        for st, en, lab in truth[s]:
            m = ms & (time >= st) & (time < en)
            if m.sum() < 3:
                continue

            def hi(v):
                x = v[m][np.isfinite(v[m])]
                # a blade action is a SPIKE inside the interval; p90 rather than max
                # so one pose glitch cannot carry it
                return float(np.percentile(x, 90)) if len(x) else np.nan
            rows.append((stem, lab, hi(v1), hi(v1_99), hi(v2), hi(v2_99)))
    return rows


def main() -> int:
    allrows = []
    for stem, csv_name in BOUTS.items():
        r = load(stem, csv_name)
        print(f"bout {stem}: {len(r)} intervals with >=3 measured frames")
        allrows += r
    if not allrows:
        print("\nno v2 caches yet -- run scripts/blade_energy.py first")
        return 1

    COLS = [("v1 blade/torso", 2), ("v1 p99", 3), ("v2 strip/ctrl", 4), ("v2 p99", 5)]

    def pick(labels, col, bout=None):
        return [r[col] for r in allrows
                if r[1] in labels and np.isfinite(r[col])
                and (bout is None or r[0] == bout)]

    n_par = len(pick({"parry"}, 2))
    print(f"\n{len(allrows)} intervals pooled, {n_par} of them parries\n")

    print("=== POOLED -- the verdict ===")
    print(f"{'comparison':<34}" + "".join(f"{c:>16}" for c, _ in COLS))
    for title, pos, neg in (
            ("parry vs non-blade", {"parry"}, QUIET | FOOTWORK),
            ("blade-action vs footwork", BLADE_ACTION, FOOTWORK),
            ("blade-action vs quiet", BLADE_ACTION, QUIET)):
        vals = [auc(pick(pos, c), pick(neg, c)) for _, c in COLS]
        print(f"{title:<34}" + "".join(f"{v:>16.2f}" for v in vals))

    print("\n=== per bout, for CONSISTENCY only -- do not read a winner off one row ===")
    print(f"{'parry vs non-blade':<34}" + "".join(f"{c:>16}" for c, _ in COLS))
    for stem in BOUTS:
        if not any(r[0] == stem for r in allrows):
            continue
        n = len(pick({"parry"}, 2, stem))
        vals = [auc(pick({"parry"}, c, stem), pick(QUIET | FOOTWORK, c, stem))
                for _, c in COLS]
        print(f"  bout {stem} (n={n:<3})".ljust(34)
              + "".join(f"{v:>16.2f}" for v in vals))

    print("\n0.50 is chance. v1 pooled over SIX parries was 0.55-0.59, which is why "
          "this was closed;\nthe question now is whether v2 clears that with "
          f"{n_par} parries, and whether it does so on\nevery bout or only one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
