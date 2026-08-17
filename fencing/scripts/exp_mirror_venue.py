"""Does mirror augmentation rescue the fencer the model cannot read?"""
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


def fit(X, A, L, Y, seed, epochs, device, pool, n_agg=N_AGG_WIDE):
    torch.manual_seed(seed)
    model = WideAggLSTM(n_agg, pool).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(TensorWindows(X, A, L, Y), batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, ab, nb, yb in dl:
            opt.zero_grad()
            lossf(model(xb.to(device), ab.to(device), nb.to(device)),
                  yb.to(device)).backward()
            opt.step()
    return model


def predict(model, X, A, L, device):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 256):
            out.append(model(torch.from_numpy(X[i:i + 256]).to(device),
                             torch.from_numpy(A[i:i + 256]).to(device),
                             torch.from_numpy(L[i:i + 256]).to(device)).argmax(1).cpu())
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="7")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    train = [s for s in stems if s != a.holdout]
    raw = {s: with_opponent(CONT / f"{s}.npz")[0] for s in stems}

    X = np.concatenate([raw[s]["X"][::a.stride] for s in train])
    A = np.concatenate([raw[s]["wide"][::a.stride] for s in train])
    L = np.concatenate([raw[s]["lengths"][::a.stride] for s in train])
    Y = np.concatenate([raw[s]["y"][::a.stride] for s in train])

    ev = raw[a.holdout]
    slot = np.load(CONT / f"{a.holdout}.npz")["slot"].astype(str)
    device = _pick_device()
    ai = CLASS_NAMES.index("advance")
    print(f"held out bout {a.holdout}; training on {train} ({len(Y)} windows)\n")
    print(f"  {'arm':<12}{'overall':>10}{'fencer A':>11}{'fencer B':>11}"
          f"{'B advance R':>13}")

    # `duplicated` is the control that decides this: same window count, same gradient
    # steps, same everything -- but the copies are identical instead of mirrored. If
    # it matches `mirrored`, the gain was optimisation budget, not handedness.
    for tag in ("baseline", "duplicated", "mirrored"):
        if tag == "mirrored":
            # aggregates are mirror-invariant, so only the pose sequence flips
            tX = np.concatenate([X, mirror_flat(X)])
            tA, tL, tY = np.concatenate([A, A]), np.concatenate([L, L]), np.concatenate([Y, Y])
        elif tag == "duplicated":
            tX = np.concatenate([X, X])
            tA, tL, tY = np.concatenate([A, A]), np.concatenate([L, L]), np.concatenate([Y, Y])
        else:
            tX, tA, tL, tY = X, A, L, Y
        accs, aA, aB, aR = [], [], [], []
        for s in range(a.seeds):
            m = fit(tX, tA, tL, tY, 42 + s, a.epochs, device, a.pool)
            p = predict(m, ev["X"], ev["wide"], ev["lengths"], device)
            ok = p == ev["y"]
            accs.append(ok.mean())
            aA.append(ok[slot == "A"].mean())
            aB.append(ok[slot == "B"].mean())
            mb = (slot == "B") & (ev["y"] == ai)
            aR.append((p[mb] == ai).mean() if mb.sum() else np.nan)
        print(f"  {tag:<12}{np.mean(accs):>9.1%}{np.mean(aA):>11.1%}"
              f"{np.mean(aB):>11.1%}{np.mean(aR):>13.1%}", flush=True)

    print("\nFencer B is the one whose arm configuration never appears in a training "
          "B slot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
