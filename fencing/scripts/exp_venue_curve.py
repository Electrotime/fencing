"""How much labelled footage does a NEW venue cost?"""
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
BUFFER = 10.0          # seconds dropped either side of the split, against overlap leak


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout", default="7")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--sizes", default="125,250,500,1000,2000")
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    if a.bout not in stems:
        print(f"no extracted bout {a.bout!r}; have {stems}")
        return 1
    others = [s for s in stems if s != a.bout]

    tgt = with_opponent(CONT / f"{a.bout}.npz")[0]
    t = np.load(CONT / f"{a.bout}.npz")["time"]
    mid = float(np.median(t))
    pool = np.flatnonzero(t < mid - BUFFER)
    test = np.flatnonzero(t > mid + BUFFER)
    pool = pool[np.argsort(t[pool])]          # time order, so a prefix is contiguous

    ev = (tgt["X"][test], tgt["wide"][test], tgt["lengths"][test], tgt["y"][test])
    device = _pick_device()
    print(f"bout {a.bout}: split at {mid:.0f}s, {len(pool)} train-pool / {len(test)} "
          f"test windows ({int((tgt['y'][test] == CLASS_NAMES.index('parry')).sum())} "
          f"parry in test)")

    o = {s: with_opponent(CONT / f"{s}.npz")[0] for s in others}
    oX = np.concatenate([o[s]["X"][::a.stride] for s in others])
    oA = np.concatenate([o[s]["wide"][::a.stride] for s in others])
    oL = np.concatenate([o[s]["lengths"][::a.stride] for s in others])
    oY = np.concatenate([o[s]["y"][::a.stride] for s in others])
    print(f"other venues: bouts {others}, {len(oY)} windows\n")

    sizes = [0] + [int(x) for x in a.sizes.split(",")]
    print(f"  {'target windows':>15}{'footage':>10}{'venue C only':>15}"
          f"{'+ other venues':>17}")
    for n in sizes:
        idx = pool[:n] if n else pool[:0]
        span = (t[idx].max() - t[idx].min()) if n else 0.0
        row = f"  {n:>15}{span:>9.0f}s"
        for tag in ("only", "plus"):
            if tag == "only" and n == 0:
                row += f"{'-':>15}"
                continue
            if n:
                X, A, L, Y = (tgt["X"][idx], tgt["wide"][idx],
                              tgt["lengths"][idx], tgt["y"][idx])
            else:
                X = A = L = Y = None
            if tag == "plus":
                X = np.concatenate([oX] + ([X] if n else []))
                A = np.concatenate([oA] + ([A] if n else []))
                L = np.concatenate([oL] + ([L] if n else []))
                Y = np.concatenate([oY] + ([Y] if n else []))
            accs, _, _ = train_eval(X, A, L, Y, *ev, N_AGG_WIDE, a.seeds, a.epochs,
                                 device, a.pool)
            row += f"{np.mean(accs):>13.1%}±{np.std(accs):.0%}"
        print(row, flush=True)

    print("\nThe 0-window row is the cross-venue baseline: every other venue, none of "
          "this one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
