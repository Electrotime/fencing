"""Run the registered priority tests on hand-supplied contested-halt tables."""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import exp_touch_probe as P

LAB = PROJECT / "data" / "labels"


def secs(s):
    m, x = s.split(":")
    return int(m) * 60 + float(x)


def read_contested(stem):
    """(time, priority, kind) per row; kind is two_colour / mixed / both_off."""
    rows = []
    for line in (LAB / f"bout{stem}_contested.tsv").read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("Score"):
            continue
        f = line.split("\t")
        if len(f) < 5 or not f[1].strip():
            continue
        left, right, won = f[2].strip().lower(), f[3].strip().lower(), f[4].strip().lower()
        off = ("off" in left) + ("off" in right)
        kind = "both_off" if off == 2 else "mixed" if off == 1 else "two_colour"
        rows.append((secs(f[1].strip()), won if won in ("left", "right") else None, kind))
    return rows


def run(tag, x, y, perm):
    v = P.auc(x, y)
    _, pf, _ = P.maxstat_p(x[None, :], y, perm)
    one = pf[0] / 2 if v > 0.5 else 1.0 - pf[0] / 2
    print(f"  {tag:<46} n={len(y):<3} ({int(y.sum())}L/{int((~y).sum())}R)  "
          f"AUC {v:.2f}   p {one:.4f}   {'CONFIRMED' if one < 0.05 else 'not confirmed'}")
    return v, one


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default="8,9")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--lead", type=float, default=0.3)
    a = ap.parse_args()

    T, D, kinds, pri = [], [], [], []
    for stem in a.bouts.split(","):
        probs = np.load(LAB / P.CACHE[stem])
        rows = read_contested(stem)
        print(f"bout {stem}: {len(rows)} contested halts, "
              f"{sum(1 for _, p, _ in rows if p)} with a priority label")
        for u, p, k in rows:
            T.append((stem, u, p or "none")); D.append(probs)
            kinds.append(k); pri.append(p or "")
    kinds, pri = np.array(kinds), np.array(pri)

    X, names, ok = P.build(T, D, a.lead)
    if P.ONSET_PREREG not in names:
        print("onset feature missing -- coverage gap"); return 1
    ko, po = kinds[ok], pri[ok]
    amp, ons = X[names.index(P.PREREG)], X[names.index(P.ONSET_PREREG)]

    print(f"\n=== REGISTERED TESTS (two of them; read a borderline p against that) ===")
    m2 = (po != "") & (ko == "two_colour")
    run(f"1. {P.PREREG} [two-colour]", amp[m2], po[m2] == "left", a.perm)
    mall = po != ""
    run(f"2. {P.ONSET_PREREG} [all priority]", ons[mall], po[mall] == "left", a.perm)

    print("\n--- secondary, not the registered pairing ---")
    run(f"   amplitude on all priority halts", amp[mall], po[mall] == "left", a.perm)
    run(f"   onset on two-colour only", ons[m2], po[m2] == "left", a.perm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
