"""Does normalising crouch per video fix cross-venue transfer?"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, N_AGG_WIDE, _pick_device
from exp_opponent import train_eval, with_opponent

CONT = PROJECT / "data" / "train_continuous"
CR = 5                     # crouch column inside the own-feature block
OPP_CR = 6 + CR            # and inside the opponent block of the wide vector


COLSETS = {"crouch": (CR, OPP_CR),
           "posture": (1, CR, 6 + 1, OPP_CR),          # stance + crouch
           "motion": (0, 3, 6 + 0, 6 + 3),             # net_forward + total_travel
           "all": tuple(range(6)) + tuple(range(6, 12))}


def normalise(wide, mode, cols=(CR, OPP_CR)):
    """Rewrite selected agg columns of one bout using only that bout's own stats."""
    w = wide.copy()
    if mode == "none":
        return w
    for col in cols:
        v = w[:, col]
        # opponent block is zero where no opponent was tracked; leave those alone
        m = v != 0 if col >= 6 else np.ones(len(v), bool)
        if m.sum() < 10:
            continue
        if mode == "median":
            w[m, col] = v[m] - np.median(v[m])
        elif mode == "zscore":
            iqr = np.subtract(*np.percentile(v[m], [75, 25]))
            w[m, col] = (v[m] - np.median(v[m])) / (iqr if iqr > 1e-6 else 1.0)
        elif mode == "p90":
            # centre on the most-crouched moments: "how close to this fencer's own
            # fencing stance", which is mix-independent in a way the median is not
            w[m, col] = v[m] - np.percentile(v[m], 90)
    return w


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="7")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--modes", default="none,median,p90,zscore")
    ap.add_argument("--columns", default="crouch", choices=sorted(COLSETS))
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    if a.holdout not in stems:
        print(f"no extracted bout {a.holdout!r}; have {stems}")
        return 1
    train = [s for s in stems if s != a.holdout]
    raw = {s: with_opponent(CONT / f"{s}.npz")[0] for s in stems}

    device = _pick_device()
    ev_y = raw[a.holdout]["y"]
    print(f"held out bout {a.holdout}: {len(ev_y)} windows, "
          f"{int((ev_y == CLASS_NAMES.index('parry')).sum())} parry; "
          f"training on {train}")
    print(f"\n  {'crouch':<10}{'overall':>10}{'seed sd':>10}"
          + "".join(f"{c[:7]:>9}" for c in CLASS_NAMES))

    for mode in a.modes.split(","):
        X = np.concatenate([raw[s]["X"][::a.stride] for s in train])
        A = np.concatenate([normalise(raw[s]["wide"], mode, COLSETS[a.columns])[::a.stride] for s in train])
        L = np.concatenate([raw[s]["lengths"][::a.stride] for s in train])
        Y = np.concatenate([raw[s]["y"][::a.stride] for s in train])
        ev = raw[a.holdout]
        accs, recs = train_eval(X, A, L, Y, ev["X"], normalise(ev["wide"], mode, COLSETS[a.columns]),
                                ev["lengths"], ev["y"], N_AGG_WIDE, a.seeds,
                                a.epochs, device, a.pool)
        rec = "".join(f"{np.mean([r[c] for r in recs]):>9.0%}" for c in CLASS_NAMES)
        print(f"  {mode:<10}{np.mean(accs):>10.1%}{np.std(accs):>10.1%}{rec}", flush=True)

    print("\nColumns after seed sd are RECALL per class. `none` is the shipped feature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
