"""Parry-targeted training interventions, screened on the parry-dense bouts. See CLAUDE.md."""
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

from src.action_model import CLASS_NAMES, N_AGG_WIDE, WEIGHT_DECAY, _pick_device
from exp_mirror import mirror_flat
from exp_opponent import WideAggLSTM, with_opponent
from train_continuous import TensorWindows

CONT = PROJECT / "data" / "train_continuous"
PARRY_I = CLASS_NAMES.index("parry")
ARMS = ("baseline", "weight_inv", "weight_sqrt", "focal2", "focal4",
        "oversample4", "oversample8", "short25", "short35", "short45")


def tail(X, lengths, k):
    """Newest k real frames, re-aligned to the front, zero-padded to k."""
    out = np.zeros((len(X), k, X.shape[2]), dtype=X.dtype)
    new_len = np.minimum(lengths, k).astype(np.int64)
    for i in range(len(X)):
        L, m = int(lengths[i]), int(new_len[i])
        if m > 0:
            out[i, :m] = X[i, L - m:L]
    return out, new_len


class FocalLoss(nn.Module):
    """Cross-entropy scaled by (1-p)^gamma so easy windows stop dominating."""

    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, y):
        logp = torch.log_softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1)
        return (-((1 - logp.exp()) ** self.gamma) * logp).mean()


def build_arm(arm, X, A, L, Y):
    """Return (X, A, L, Y, loss, n_frames) for one intervention."""
    loss = nn.CrossEntropyLoss()
    if arm.startswith("short"):
        k = int(arm[5:])
        X, L = tail(X, L, k)
        return X, A, L, Y, loss, k
    if arm.startswith("oversample"):
        rep = int(arm[10:]) - 1
        idx = np.flatnonzero(Y == PARRY_I)
        if rep > 0 and len(idx):
            extra = np.tile(idx, rep)
            X, A = np.concatenate([X, X[extra]]), np.concatenate([A, A[extra]])
            L, Y = np.concatenate([L, L[extra]]), np.concatenate([Y, Y[extra]])
        return X, A, L, Y, loss, X.shape[1]
    if arm.startswith("focal"):
        return X, A, L, Y, FocalLoss(float(arm[5:])), X.shape[1]
    if arm.startswith("weight"):
        counts = np.bincount(Y, minlength=len(CLASS_NAMES)).astype(np.float64)
        counts[counts == 0] = 1.0
        w = counts.sum() / counts
        if arm == "weight_sqrt":
            w = np.sqrt(w)
        w = w / w.mean()
        return X, A, L, Y, nn.CrossEntropyLoss(
            weight=torch.tensor(w, dtype=torch.float32)), X.shape[1]
    return X, A, L, Y, loss, X.shape[1]


def fit(X, A, L, Y, loss, seed, epochs, device, pool):
    torch.manual_seed(seed)
    model = WideAggLSTM(N_AGG_WIDE, pool).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    if hasattr(loss, "weight") and loss.weight is not None:
        loss.weight = loss.weight.to(device)
    dl = DataLoader(TensorWindows(X, A, L, Y), batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, ab, nb, yb in dl:
            opt.zero_grad()
            loss(model(xb.to(device), ab.to(device), nb.to(device)),
                 yb.to(device)).backward()
            opt.step()
    model.eval()
    return model


def predict(model, X, A, L, device):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model(torch.from_numpy(X[i:i + 256]).to(device),
                             torch.from_numpy(A[i:i + 256]).to(device),
                             torch.from_numpy(L[i:i + 256]).to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def pr(pred, y, c):
    tp = int(((pred == c) & (y == c)).sum())
    np_ = int((pred == c).sum())
    nt = int((y == c).sum())
    return (tp / np_ if np_ else float("nan"), tp / nt if nt else float("nan"))


def _self_test():
    x = np.arange(2 * 6 * 1, dtype=np.float32).reshape(2, 6, 1)
    L = np.array([6, 3])
    t, nl = tail(x, L, 2)
    assert list(nl) == [2, 2], nl
    assert t[0, 0, 0] == 4 and t[0, 1, 0] == 5, t[0]      # newest two of six
    assert t[1, 0, 0] == 7 and t[1, 1, 0] == 8, t[1]      # newest two of three real
    t3, nl3 = tail(x, np.array([2, 6]), 4)
    assert list(nl3) == [2, 4], nl3
    lg = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    y = torch.tensor([0, 1])
    assert FocalLoss(2.0)(lg, y).item() < nn.CrossEntropyLoss()(lg, y).item() + 1e-6
    hard = torch.tensor([[0.1, 0.0]])
    yh = torch.tensor([0])
    assert FocalLoss(0.0)(hard, yh).item() == \
        torch.nn.functional.cross_entropy(hard, yh).item()   # gamma 0 == plain CE
    Xd = np.zeros((10, 4, 1), np.float32); Ad = np.zeros((10, 13), np.float32)
    Ld = np.full(10, 4); Yd = np.array([PARRY_I] + [0] * 9)
    _, _, _, Yo, _, _ = build_arm("oversample4", Xd, Ad, Ld, Yd)
    assert int((Yo == PARRY_I).sum()) == 4, int((Yo == PARRY_I).sum())
    print("exp_parry_push self-test: ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="baseline,weight_sqrt,focal2,oversample4,short25,short35")
    ap.add_argument("--holdouts", default="4,7")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return 0
    _self_test()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    raw = {s: with_opponent(CONT / f"{s}.npz")[0] for s in stems}
    device = _pick_device()
    arms = a.arms.split(",")
    rows = {}

    for holdout in a.holdouts.split(","):
        train = [s for s in stems if s != holdout]
        bX = np.concatenate([raw[s]["X"][::a.stride] for s in train])
        bA = np.concatenate([raw[s]["wide"][::a.stride] for s in train])
        bL = np.concatenate([raw[s]["lengths"][::a.stride] for s in train])
        bY = np.concatenate([raw[s]["y"][::a.stride] for s in train])
        bX = np.concatenate([bX, mirror_flat(bX)])
        bA, bL, bY = (np.concatenate([bA, bA]), np.concatenate([bL, bL]),
                      np.concatenate([bY, bY]))
        ev = raw[holdout]
        for arm in arms:
            X, A, L, Y, loss, k = build_arm(arm, bX, bA, bL, bY)
            eX, eL = ((ev["X"], ev["lengths"]) if k == bX.shape[1]
                      else tail(ev["X"], ev["lengths"], k))
            accs, pp, rr = [], [], []
            for s in range(a.seeds):
                m = fit(X, A, L, Y, loss, 42 + s, a.epochs, device, a.pool)
                p = predict(m, eX, ev["wide"], eL, device)
                accs.append(float((p == ev["y"]).mean()))
                q, r = pr(p, ev["y"], PARRY_I)
                pp.append(q); rr.append(r)
            rows[(holdout, arm)] = (np.mean(accs), np.nanmean(pp), np.nanmean(rr))
            print(f"  bout {holdout} {arm:<12} acc {np.mean(accs):.1%}  "
                  f"parry P {np.nanmean(pp):.0%} R {np.nanmean(rr):.0%}", flush=True)

    print(f"\n  {'arm':<14}" + "".join(f"{'b' + h + ' acc':>10}{'P':>6}{'R':>6}"
                                       for h in a.holdouts.split(",")) + f"{'mean R':>9}")
    for arm in arms:
        line = f"  {arm:<14}"
        rs = []
        for h in a.holdouts.split(","):
            acc, q, r = rows[(h, arm)]
            rs.append(r)
            line += f"{acc:>10.1%}{q:>6.0%}{r:>6.0%}"
        print(line + f"{np.nanmean(rs):>9.0%}")
    print("\nParry P and R are the target; overall accuracy is the veto.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
