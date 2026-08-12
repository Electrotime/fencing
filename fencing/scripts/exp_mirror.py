"""Left-right mirroring as free training data. Invariance PROVEN, not assumed.

The action model never sees pixels -- `_normalize_sequence` hip-centres and
divides by body height before anything reaches the LSTM -- so the Roboflow-style
crop/zoom augmentation that helped the blade detector would be normalised away
here. Mirroring is different: it changes which leg leads, which arm extends and
which way the fencer faces, none of which normalisation removes.

THE TRAP: `forward` is `world_vel * nose_dir`. Mirror the keypoints and nose_dir
flips; fail to mirror the motion track and world_vel does not, so every advance
becomes a retreat. That is the exact direction-inversion bug that cost 11 points
this morning, and it would be invisible -- the data would just be quietly wrong.

THE CLAIM: under a TRUE mirror (scene and motion together) both terms flip, so
`forward` is unchanged and every other engineered feature is symmetric too. If
that holds, the cached `agg` vectors can be reused as-is and only X needs
mirroring -- which matters, because the motion tracks were never cached.

test_invariance() checks that claim against the real _engineered_features rather
than trusting the argument. It runs first and aborts on failure.

usage: py -3 scripts/exp_mirror.py [--holdout 1] [--seeds 2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, _engineered_features, _pick_device
from train_continuous import (TensorWindows, clip_dataset_arrays, evaluate,
                              load_bouts, train_once)

# MediaPipe Pose left/right pairs. 0 (nose) and nothing else is unpaired.
PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
         (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30),
         (31, 32)]


def mirror_landmarks(kp):
    """(..., 33, 4) -> mirrored. Negate x AND swap the left/right landmarks.

    Doing only one of the two is worse than doing neither: negating x without
    swapping gives a fencer whose left leg is on the right side of their body.
    """
    out = kp.copy()
    out[..., 0] = -out[..., 0]
    for a, b in PAIRS:
        out[..., [a, b], :] = out[..., [b, a], :]
    return out


def mirror_flat(X):
    """(N, T, 132) flattened windows -> mirrored, same shape."""
    n, t, _ = X.shape
    return mirror_landmarks(X.reshape(n, t, 33, 4)).reshape(n, t, 132)


def test_invariance(tol=1e-5):
    """Every engineered feature must be unchanged by a true mirror."""
    rng = np.random.default_rng(0)
    worst, checked = 0.0, 0
    for trial in range(200):
        n = int(rng.integers(8, 60))
        kp = rng.normal(0, 0.4, size=(n, 33, 4)).astype(np.float32)
        kp[:, :, 3] = 1.0                       # visibility
        # a plausible motion track: hip-x drifting across frame, camera panning
        hip = np.cumsum(rng.normal(0.004, 0.002, size=n)).astype(np.float32) + 0.5
        pan = rng.normal(1.5, 0.5, size=n).astype(np.float32)
        motion = np.stack([pan, hip], axis=1)

        base = _engineered_features(kp, motion)
        # TRUE mirror: flip the scene AND the motion track that describes it
        m_kp = mirror_landmarks(kp)
        m_motion = np.stack([-pan, 1.0 - hip], axis=1).astype(np.float32)
        mirrored = _engineered_features(m_kp, m_motion)

        d = float(np.max(np.abs(base - mirrored)))
        worst = max(worst, d)
        checked += 1
    return worst, checked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    # default to what actually ships -- the first run of this experiment
    # used mean pooling, which is no longer the deployed configuration
    ap.add_argument("--pool", default="last", choices=("mean", "max", "last"))
    a = ap.parse_args()

    worst, n = test_invariance()
    print(f"invariance check: {n} random windows, worst feature delta {worst:.2e}")
    if worst > 1e-4:
        print("FAILED -- a true mirror changes the engineered features, so the")
        print("cached `agg` cannot be reused and mirroring X alone would poison")
        print("the training set. Do not proceed; recompute agg from mirrored")
        print("motion instead (which means re-running extraction).")
        return 1
    print("PASSED -- agg is mirror-invariant, so only X needs mirroring.\n")

    bouts = load_bouts(a.stride)
    tr = [k for k in bouts if k != a.holdout]
    cX, cA, cL, cY = clip_dataset_arrays()
    X = np.concatenate([cX] + [bouts[k]["train"]["X"] for k in tr])
    A = np.concatenate([cA] + [bouts[k]["train"]["agg"] for k in tr])
    L = np.concatenate([cL] + [bouts[k]["train"]["lengths"] for k in tr])
    Y = np.concatenate([cY] + [bouts[k]["train"]["y"] for k in tr])

    aug = dict(
        X=np.concatenate([X, mirror_flat(X)]), agg=np.concatenate([A, A]),
        L=np.concatenate([L, L]), Y=np.concatenate([Y, Y]))

    device = _pick_device()
    print(f"held out bout {a.holdout}; baseline {len(Y)} windows, "
          f"mirrored {len(aug['Y'])}\n")
    print(f"{'setup':<22}{'overall':>9}{'+-':>7}" + "".join(f"{c[:7]:>9}" for c in CLASS_NAMES))

    for name, (xx, aa, ll, yy) in (("baseline", (X, A, L, Y)),
                                   ("+ mirrored", (aug["X"], aug["agg"], aug["L"], aug["Y"]))):
        accs, recs = [], []
        for s in range(a.seeds):
            m = train_once(TensorWindows(xx, aa, ll, yy), None, a.epochs, 42 + s,
                           device, pool=a.pool)
            acc, per = evaluate(m, bouts[a.holdout]["eval"], device, apply_prior=False)
            accs.append(acc); recs.append(per)
        line = f"{name:<22}{np.mean(accs):>8.1%}{np.std(accs):>7.1%}"
        for c in CLASS_NAMES:
            line += f"{np.nanmean([r[c][2] for r in recs]):>9.0%}"
        print(line, flush=True)
    print("\n(columns are per-class RECALL on the held-out bout)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
