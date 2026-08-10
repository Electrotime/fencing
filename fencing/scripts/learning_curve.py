"""Does labelling MORE footage still pay? Measure it instead of guessing.

Aaron has labelled ~1000 s of fencer-time across four bouts and the obvious next
question is whether a fifth bout is worth hours of his time. That is answerable
from data already in hand: train on increasing amounts of it and see whether
held-out accuracy is still climbing at the point we have reached.

DESIGN DETAIL THAT MATTERS: the fraction is taken as a CONTIGUOUS SLICE OF TIME,
not as a random sample of windows. Labelling less footage means covering less of
the match, not sampling sparsely across all of it -- and because neighbouring
windows share 92% of their frames, a random 25% sample still covers essentially
the whole bout and would flatter the curve badly. `--random-subsample` is provided
to demonstrate exactly that difference rather than to be used.

Held-out bout is never trained on, at any fraction.

usage: py -3 scripts/learning_curve.py --holdout 1 [--seeds 2] [--random-subsample]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, _pick_device
from train_continuous import (TensorWindows, clip_dataset_arrays, evaluate,
                              load_bouts, train_once)

FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]


def time_slice(bout, frac, rng=None):
    """First `frac` of the bout's timeline, or a random window sample if rng given."""
    n = len(bout["y"])
    if frac >= 1.0:
        return bout
    k = int(round(n * frac))
    if k == 0:
        return None
    if rng is not None:
        idx = rng.choice(n, size=k, replace=False)
        idx.sort()
    else:
        idx = np.arange(k)      # windows are emitted in time order
    return {kk: v[idx] for kk, v in bout.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--random-subsample", action="store_true")
    a = ap.parse_args()

    bouts = load_bouts(a.stride)
    if a.holdout not in bouts:
        print(f"no extracted bout {a.holdout!r}; have {sorted(bouts)}")
        return 1
    train_keys = [k for k in bouts if k != a.holdout]
    device = _pick_device()
    cX, cA, cL, cY = clip_dataset_arrays()

    print(f"held out bout {a.holdout} ({len(bouts[a.holdout]['eval']['y'])} eval windows)")
    print(f"training pool: clips({len(cY)}) + bouts {train_keys}")
    print(f"fraction is a {'RANDOM WINDOW SAMPLE' if a.random_subsample else 'CONTIGUOUS TIME SLICE'}"
          f" of each training bout\n", flush=True)
    print(f"{'frac':>6}{'cont. windows':>15}{'total':>8}{'held-out acc':>14}{'+-':>7}")

    rows = []
    for frac in FRACTIONS:
        rng = np.random.default_rng(0) if a.random_subsample else None
        parts = [time_slice(bouts[k]["train"], frac, rng) for k in train_keys]
        parts = [p for p in parts if p is not None]
        if parts:
            X = np.concatenate([cX] + [p["X"] for p in parts])
            A = np.concatenate([cA] + [p["agg"] for p in parts])
            L = np.concatenate([cL] + [p["lengths"] for p in parts])
            Y = np.concatenate([cY] + [p["y"] for p in parts])
            n_cont = sum(len(p["y"]) for p in parts)
        else:
            X, A, L, Y, n_cont = cX, cA, cL, cY, 0

        accs = []
        for s in range(a.seeds):
            m = train_once(TensorWindows(X, A, L, Y), None, a.epochs, 42 + s, device)
            acc, per = evaluate(m, bouts[a.holdout]["eval"], device, apply_prior=False)
            accs.append(acc)
        rows.append((frac, n_cont, len(Y), float(np.mean(accs)), float(np.std(accs)), per))
        print(f"{frac:>6.2f}{n_cont:>15}{len(Y):>8}{np.mean(accs):>13.1%}"
              f"{np.std(accs):>7.1%}", flush=True)

    print("\n=== is the curve still climbing? ===")
    last_two = rows[-1][3] - rows[-2][3]
    mid = rows[-2][3] - rows[len(rows) // 2][3]
    print(f"  0.75 -> 1.00 : {last_two:+.1%}   (the marginal value of the LAST 25%)")
    print(f"  0.50 -> 0.75 : {mid:+.1%}")
    print(f"  0.00 -> 1.00 : {rows[-1][3] - rows[0][3]:+.1%}")
    print()
    if last_two > 0.02:
        print("  STILL CLIMBING -- another labelled bout is likely worth the hours.")
    elif last_two > 0.005:
        print("  FLATTENING -- more data helps a little; weigh it against other work.")
    else:
        print("  FLAT -- more of the SAME KIND of footage is not the bottleneck.")
        print("  Prefer different venues/framings, or a different model change.")
    print("\n  per-class recall at full data: " + "  ".join(
        f"{c}={rows[-1][5][c][2]:.0%}" for c in CLASS_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
