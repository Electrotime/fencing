"""Elbow-opening RATE as a 7th feature, with a shuffled control. See CLAUDE.md."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, _pick_device
from exp_mirror import mirror_flat
from exp_mirror_venue import fit, predict
from exp_opponent import with_opponent

CONT = PROJECT / "data" / "train_continuous"
SL, SR, EL, ER, WL, WR = 11, 12, 13, 14, 15, 16
RECENT = 8            # frames, ~0.27s -- the span the AUC sweep favoured


def _angle(a, b, c):
    v1, v2 = a - b, c - b
    n = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return np.degrees(np.arccos(np.clip((v1 * v2).sum(-1) / np.where(n == 0, 1, n), -1, 1)))


def ext_rate(stem):
    """Fastest elbow opening in the newest RECENT frames, per window."""
    d = np.load(CONT / f"{stem}.npz")
    X = d["X"].reshape(len(d["X"]), 60, 33, 4)[..., :2]
    ln = d["lengths"]
    e = np.maximum(_angle(X[:, :, SL], X[:, :, EL], X[:, :, WL]),
                   _angle(X[:, :, SR], X[:, :, ER], X[:, :, WR]))
    out = np.zeros(len(e), dtype=np.float32)
    for i in range(len(e)):
        seq = e[i, :ln[i]]
        if len(seq) > 1:
            de = np.diff(seq)[-RECENT:]
            out[i] = de.max()
    return np.clip(out / 30.0, -2.0, 2.0)      # degrees/frame -> roughly unit scale


def widen(stem, d, col):
    """[own(7) | opponent(7) | present(1)], paired exactly as with_opponent does."""
    z = np.load(CONT / f"{stem}.npz")
    slot, time = z["slot"].astype(str), z["time"]
    own = np.concatenate([d["agg"], col[:, None]], axis=1)
    index = {(s, float(t)): i for i, (s, t) in enumerate(zip(slot, time))}
    opp = np.zeros_like(own)
    present = np.zeros((len(own), 1), dtype=np.float32)
    for i, (s, t) in enumerate(zip(slot, time)):
        j = index.get(("B" if s == "A" else "A", float(t)))
        if j is not None:
            opp[i] = own[j]
            present[i] = 1.0
    return np.concatenate([own, opp, present], axis=1).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    raw = {s: with_opponent(CONT / f"{s}.npz")[0] for s in stems}
    col = {s: ext_rate(s) for s in stems}
    rng = np.random.default_rng(0)
    shuf = {s: rng.permutation(col[s]) for s in stems}
    device = _pick_device()
    rows = {}

    for holdout in stems:
        train = [s for s in stems if s != holdout]
        rows[holdout] = {}
        for arm in ("baseline", "extrate", "shuffled"):
            src = col if arm == "extrate" else shuf
            def A_of(s, arm=arm, src=src):
                return raw[s]["wide"] if arm == "baseline" else widen(s, raw[s], src[s])
            X = np.concatenate([raw[s]["X"][::a.stride] for s in train])
            A = np.concatenate([A_of(s)[::a.stride] for s in train])
            L = np.concatenate([raw[s]["lengths"][::a.stride] for s in train])
            Y = np.concatenate([raw[s]["y"][::a.stride] for s in train])
            X = np.concatenate([X, mirror_flat(X)])
            A, L, Y = np.concatenate([A, A]), np.concatenate([L, L]), np.concatenate([Y, Y])
            ev = raw[holdout]
            accs = []
            for s in range(a.seeds):
                m = fit(X, A, L, Y, 42 + s, a.epochs, device, a.pool, n_agg=A.shape[1])
                p = predict(m, ev["X"], A_of(holdout), ev["lengths"], device)
                accs.append(float((p == ev["y"]).mean()))
            rows[holdout][arm] = (float(np.mean(accs)), float(np.std(accs)))
        print(f"  bout {holdout} done", flush=True)

    print(f"\n  {'bout':<6}{'baseline':>13}{'+ext rate':>13}{'+shuffled':>13}"
          f"{'real-shuf':>11}")
    d = []
    for h, r in rows.items():
        b, e, s = r["baseline"], r["extrate"], r["shuffled"]
        d.append(e[0] - s[0])
        print(f"  {h:<6}{b[0]:>12.1%}{e[0]:>13.1%}{s[0]:>13.1%}"
              f"{100 * (e[0] - s[0]):>+11.2f}")
    print(f"\n  mean real-minus-shuffled {100 * np.mean(d):+.2f} pts, "
          f"worst {100 * min(d):+.2f}, best {100 * max(d):+.2f}")
    print("  The shuffled column has the same distribution and widens the head "
          "identically,\n  so only the difference is information rather than capacity.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
