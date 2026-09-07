"""Pooled confirmation across every non-discovery bout, lamp-derived and hand-labelled alike."""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import exp_contested as C
import exp_priority as PR
import exp_touch_probe as P

LAB = PROJECT / "data" / "labels"

# Lamp-derived bouts read priority off the scoreboard; contested bouts have it
# by hand. Bouts 4 and 7 are excluded: the feature was chosen on them.
LAMP = ("5", "6")
HAND = ("8", "9", "10", "11", "12", "13", "14")


def rows_for(stem):
    """(time, priority, kind) for one bout, from whichever source it has."""
    if stem in HAND:
        return C.read_contested(stem)
    import check_touches as CT
    import read_scoreboard as RS
    _, rows = CT.check(LAB / f"bout{stem}_touches.tsv")
    t, ser = RS.lamp_series("", RS.LAYOUT[stem]["lamp"], 0.1, LAB / f"{stem}_lamp.npz")
    thr = RS.lamp_all_thresholds(ser)
    out = []
    for r in rows:
        st = RS.lamp_states(t, ser, r["t"], thr)
        out.append((r["t"], RS.priority_from(st, r["scorer"]), RS.lamp_kind(st)))
    return out


def boot_ci(x, y, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(y):
            out.append(P.auc(x[i], y[i]))
    return np.percentile(out, [2.5, 97.5])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=",".join(LAMP + HAND))
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--lead", type=float, default=0.3)
    a = ap.parse_args()

    stems = a.bouts.split(",")
    circular = sorted(set(P.SELECTED_ON) & set(stems))
    if circular:
        print(f"CIRCULAR -- bout(s) {','.join(circular)} chose the feature.")
        return 1

    T, D, pri, kinds, src = [], [], [], [], []
    for stem in stems:
        probs = np.load(LAB / P.CACHE[stem])
        rows = rows_for(stem)
        n = sum(1 for _, p, _ in rows if p)
        print(f"bout {stem:>2}: {len(rows):>3} halts, {n:>3} with a priority label "
              f"({'hand' if stem in HAND else 'lamp'})")
        for t, p, k in rows:
            T.append((stem, t, p or "none")); D.append(probs)
            pri.append(p or ""); kinds.append(k); src.append(stem)

    pri, kinds, src = np.array(pri), np.array(kinds), np.array(src)
    X, names, ok = P.build(T, D, a.lead)
    x_all = X[names.index(P.PREREG)]
    po, ko, so = pri[ok], kinds[ok], src[ok]

    for tag, m in (("all priority", po != ""),
                   ("two-colour only", (po != "") & (ko == "two_colour"))):
        x, y = x_all[m], po[m] == "left"
        v = P.auc(x, y)
        _, pf, _ = P.maxstat_p(x[None, :], y, a.perm)
        one = pf[0] / 2 if v > 0.5 else 1.0 - pf[0] / 2
        lo, hi = boot_ci(x, y)
        print(f"\n=== POOLED, {tag} ===")
        print(f"  n={m.sum()} ({int(y.sum())}L/{int((~y).sum())}R)  AUC {v:.2f}  "
              f"one-sided p {one:.4f}  95% CI [{lo:.2f}, {hi:.2f}]")
        print(f"  x2 for the two registered features: p {min(1.0, one * 2):.4f}")
        print(f"  VERDICT: {'CONFIRMED' if one < 0.05 else 'not confirmed'}")

    print("\n--- leave-one-bout-out, all priority ---")
    m = po != ""
    for stem in stems:
        k = m & (so != stem)
        x, y = x_all[k], po[k] == "left"
        print(f"  without bout {stem:>2}: n={k.sum():<3} AUC {P.auc(x, y):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
