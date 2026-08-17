"""Binary parry head vs the shipped promoter, coverage-matched."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import (CLASS_NAMES, DROPOUT, HIDDEN_SIZE, N_AGG_WIDE,
                              WEIGHT_DECAY, _pick_device)
from exp_mirror import mirror_flat
from exp_opponent import WideAggLSTM, with_opponent
from train_continuous import TensorWindows

CONT = PROJECT / "data" / "train_continuous"
PARRY_I = CLASS_NAMES.index("parry")
LUNGE_I = CLASS_NAMES.index("lunge")
MIN_PARRY = 50        # bouts 1 and 2 hold 8 and 15 parry windows; recall there is noise

# shipped gate, mirrored from demo_video
PARRY_OPP_LUNGE_MIN = 0.20
PARRY_PROMOTE_MIN = 0.15
PARRY_PROMOTE_OPP_MIN = 0.60


class BinaryHead(WideAggLSTM):
    """Same trunk, two outputs."""

    def __init__(self, n_agg, pool="last"):
        super().__init__(n_agg, pool)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + n_agg, 64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64, 2))


def fit(cls, X, A, L, Y, seed, epochs, device, pool, weight=None):
    torch.manual_seed(seed)
    model = cls(N_AGG_WIDE, pool).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss(weight=weight)
    dl = DataLoader(TensorWindows(X, A, L, Y), batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, ab, nb, yb in dl:
            opt.zero_grad()
            lossf(model(xb.to(device), ab.to(device), nb.to(device)),
                  yb.to(device)).backward()
            opt.step()
    model.eval()
    return model


def probs_of(model, X, A, L, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(torch.softmax(
                model(torch.from_numpy(X[i:i + 256]).to(device),
                      torch.from_numpy(A[i:i + 256]).to(device),
                      torch.from_numpy(L[i:i + 256]).to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def opponent_lunge(probs, slot, time):
    """P(lunge) of the other fencer at the same timestamp."""
    index = {(s, float(t)): i for i, (s, t) in enumerate(zip(slot, time))}
    out = np.zeros(len(time), dtype=np.float32)
    for i, (s, t) in enumerate(zip(slot, time)):
        j = index.get(("B" if s == "A" else "A", float(t)))
        if j is not None:
            out[i] = probs[j, LUNGE_I]
    return out


def score(pred, y):
    tp = int(((pred == PARRY_I) & (y == PARRY_I)).sum())
    fp = int(((pred == PARRY_I) & (y != PARRY_I)).sum())
    n = int((y == PARRY_I).sum())
    return (float((pred == y).mean()),
            tp / (tp + fp) if tp + fp else float("nan"),
            tp / n if n else float("nan"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    raw = {s: with_opponent(CONT / f"{s}.npz")[0] for s in stems}
    device = _pick_device()
    rows = {}

    for holdout in stems:
        train = [s for s in stems if s != holdout]
        X = np.concatenate([raw[s]["X"][::a.stride] for s in train])
        A = np.concatenate([raw[s]["wide"][::a.stride] for s in train])
        L = np.concatenate([raw[s]["lengths"][::a.stride] for s in train])
        Y = np.concatenate([raw[s]["y"][::a.stride] for s in train])
        X = np.concatenate([X, mirror_flat(X)])
        A, L, Y = np.concatenate([A, A]), np.concatenate([L, L]), np.concatenate([Y, Y])
        Yb = (Y == PARRY_I).astype(np.int64)
        w = torch.tensor([1.0, float((Yb == 0).sum() / max((Yb == 1).sum(), 1))],
                         device=device)

        ev = raw[holdout]
        d = np.load(CONT / f"{holdout}.npz")
        y = ev["y"]
        if int((y == PARRY_I).sum()) < MIN_PARRY:
            print(f"  bout {holdout} skipped ({int((y == PARRY_I).sum())} parry windows)")
            continue
        acc = {k: [] for k in ("six", "shipped", "twohead")}
        for s in range(a.seeds):
            six = fit(WideAggLSTM, X, A, L, Y, 42 + s, a.epochs, device, a.pool)
            binh = fit(BinaryHead, X, A, L, Yb, 42 + s, a.epochs, device, a.pool, w)
            p6 = probs_of(six, ev["X"], ev["wide"], ev["lengths"], device)
            pb = probs_of(binh, ev["X"], ev["wide"], ev["lengths"], device)[:, 1]
            opp = opponent_lunge(p6, d["slot"].astype(str), d["time"])

            base = p6.argmax(axis=1)
            alt = p6.copy()
            alt[:, PARRY_I] = -1.0
            demoted = np.where((base == PARRY_I) & (opp < PARRY_OPP_LUNGE_MIN),
                               alt.argmax(axis=1), base)
            acc["six"].append(score(demoted, y))

            eligible = (base != PARRY_I) & (opp >= PARRY_PROMOTE_OPP_MIN)
            hit = eligible & (p6[:, PARRY_I] >= PARRY_PROMOTE_MIN)
            acc["shipped"].append(score(np.where(hit, PARRY_I, demoted), y))

            k = int(hit.sum())                 # coverage-matched, so no test-set tuning
            cand = np.flatnonzero(eligible)
            top = cand[np.argsort(-pb[cand])[:k]] if k else np.array([], dtype=int)
            th = np.zeros(len(y), dtype=bool)
            th[top] = True
            acc["twohead"].append(score(np.where(th, PARRY_I, demoted), y))
        rows[holdout] = {k: np.mean(v, axis=0) for k, v in acc.items()}
        print(f"  bout {holdout} done", flush=True)

    print(f"\n  {'bout':<6}{'arm':<10}{'overall':>10}{'parry P':>10}{'parry R':>10}")
    for holdout, r in rows.items():
        for k, nm in (("six", "six-way"), ("shipped", "shipped"), ("twohead", "two-head")):
            o, p, rc = r[k]
            print(f"  {holdout:<6}{nm:<10}{o:>9.1%}{p:>10.0%}{rc:>10.0%}")
    print(f"\n  {'MEAN':<6}{'arm':<10}{'overall':>10}{'parry P':>10}{'parry R':>10}")
    for k, nm in (("six", "six-way"), ("shipped", "shipped"), ("twohead", "two-head")):
        m = np.nanmean([r[k] for r in rows.values()], axis=0)
        print(f"  {'':<6}{nm:<10}{m[0]:>9.1%}{m[1]:>10.0%}{m[2]:>10.0%}")
    print("\nPromotion counts are matched, so parry P and R move only if the two-head "
          "picks better windows.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
