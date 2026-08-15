"""Does the LSTM actually need 2 seconds of sequence? Never tested."""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, _pick_device
from train_continuous import (TensorWindows, clip_dataset_arrays, evaluate,
                              load_bouts, train_once)


def tail(X, lengths, k):
    """Newest k real frames, re-aligned to the front, zero-padded to k."""
    n = len(X)
    out = np.zeros((n, k, X.shape[2]), dtype=X.dtype)
    new_len = np.minimum(lengths, k).astype(np.int64)
    for i in range(n):
        L, m = int(lengths[i]), int(new_len[i])
        if m > 0:
            out[i, :m] = X[i, L - m:L]
    return out, new_len


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--lengths", default="15,25,35,45,60")
    a = ap.parse_args()

    ks = [int(x) for x in a.lengths.split(",")]
    bouts = load_bouts(a.stride)
    tr = [k for k in bouts if k != a.holdout]
    cX, cA, cL, cY = clip_dataset_arrays()
    X = np.concatenate([cX] + [bouts[k]["train"]["X"] for k in tr])
    A = np.concatenate([cA] + [bouts[k]["train"]["agg"] for k in tr])
    L = np.concatenate([cL] + [bouts[k]["train"]["lengths"] for k in tr])
    Y = np.concatenate([cY] + [bouts[k]["train"]["y"] for k in tr])
    ev = bouts[a.holdout]["eval"]

    device = _pick_device()
    print(f"held out bout {a.holdout} ({len(ev['y'])} eval windows), pool={a.pool}")
    print(f"median real length in training data: {int(np.median(L))} frames "
          f"({np.median(L) / 29.97:.2f}s)\n")
    print(f"{'frames':>7}{'seconds':>9}{'overall':>9}{'+-':>7}"
          + "".join(f"{c[:7]:>9}" for c in CLASS_NAMES))

    for k in ks:
        Xk, Lk = tail(X, L, k)
        eXk, eLk = tail(ev["X"], ev["lengths"], k)
        ek = dict(X=eXk, agg=ev["agg"], lengths=eLk, y=ev["y"])
        accs, recs = [], []
        for s in range(a.seeds):
            m = train_once(TensorWindows(Xk, A, Lk, Y), None, a.epochs, 42 + s,
                           device, pool=a.pool)
            acc, per = evaluate(m, ek, device, apply_prior=False)
            accs.append(acc); recs.append(per)
        line = f"{k:>7}{k / 29.97:>9.2f}{np.mean(accs):>8.1%}{np.std(accs):>7.1%}"
        for c in CLASS_NAMES:
            line += f"{np.nanmean([r[c][2] for r in recs]):>9.0%}"
        print(line, flush=True)

    print("\n(columns are per-class RECALL. 60 = the current shipped window.)")
    print("agg still covers the full 2 s in every row -- see the docstring; this")
    print("isolates the SEQUENCE length, not the whole pipeline's window.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
