"""Pre-registered test on the priority target: who had right of way, from lamps plus award."""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import check_touches as CT
import exp_touch_probe as P
import read_scoreboard as RS

LAB = PROJECT / "data" / "labels"


def priority_labels(stem):
    """(time, priority side) for every halt where priority applies and is inferable."""
    _, rows = CT.check(LAB / f"bout{stem}_touches.tsv")
    t, ser = RS.lamp_series("", RS.LAYOUT[stem]["lamp"], 0.1, LAB / f"{stem}_lamp.npz")
    thr = RS.lamp_all_thresholds(ser)
    out = []
    for r in rows:
        st = RS.lamp_states(t, ser, r["t"], thr)
        out.append((r["t"], RS.priority_from(st, r["scorer"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default="5,6")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--lead", type=float, default=0.3)
    a = ap.parse_args()

    T, D, pri = [], [], []
    for stem in a.bouts.split(","):
        probs = np.load(LAB / P.CACHE[stem])
        rows = priority_labels(stem)
        n = sum(1 for _, p in rows if p)
        print(f"bout {stem}: {len(rows)} halts, {n} with a priority label")
        for u, p in rows:
            T.append((stem, u, p or "none"))
            D.append(probs)
            pri.append(p)

    pri = np.array([p if p else "" for p in pri])
    X, names, ok = P.build(T, D, a.lead)
    have = ok & (pri != "")
    y = pri[have] == "left"
    x = X[names.index(P.PREREG)][(pri[ok] != "")]

    print(f"\n=== PRE-REGISTERED: {P.PREREG} on the PRIORITY target ===")
    print(f"  {have.sum()} labelled halts ({y.sum()} left / {(~y).sum()} right)")
    v = P.auc(x, y)
    _, pf, _ = P.maxstat_p(x[None, :], y, a.perm)
    one = pf[0] / 2 if v > 0.5 else 1.0 - pf[0] / 2
    print(f"  AUC {v:.2f}   one-sided p {one:.4f}")
    overlap = sorted(set(P.SELECTED_ON) & set(a.bouts.split(",")))
    if overlap:
        print(f"  CIRCULAR -- bout(s) {','.join(overlap)} are where the feature was chosen.")
        return 1
    print("  VERDICT: " + ("CONFIRMED" if one < 0.05 else "not confirmed"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
