"""How should the LSTM output be reduced over time? Measure, don't argue.

The shipped model mean-pools across the window. Twice this project has claimed
that this is why transient classes fail (lunge 24% recall, and Aaron's field note
that slow movement gets called `neutral` -- both are things a 60-frame average
would do). The first claim was retracted: it rested on a per-frame model that
turned out to be a degenerate lunge predictor, firing lunge on 60% of bout 1's
windows against a 2% true share. So the hypothesis has never actually been tested
with a sound model and a proper split. This does that.

Variants, all sharing the same trunk and training recipe so only the reduction
differs:

  mean     current: masked average over real frames
  max      masked max -- a spike anywhere in the window survives, which is what a
           0.7 s lunge inside a 2 s window IS
  meanmax  both concatenated: sustained AND transient evidence together
  attn     learned attention over timesteps, masked softmax
  last     the final real timestep, no aggregation

EVERY VARIANT MUST MASK THE PADDING. Clip length is strongly class-correlated
(lunge/parry ~24 frames = 60% padding, advance 46, retreat 48, sliced
neutral/walking 0%), so pooling over the zeros lets the model read "how much of
this window is padding" as a class cue. That artifact previously scored WELL on
validation and then collapsed on video -- see ActionLSTM's docstring. An unmasked
max would be the worst offender of all, since zeros beat negative activations.

usage: py -3 scripts/exp_pooling.py [--holdout 1] [--seeds 2]
"""
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

from src.action_model import (CLASS_NAMES, DROPOUT, HIDDEN_SIZE, INPUT_SIZE,
                              N_AGG_FEATURES, NUM_CLASSES, WEIGHT_DECAY, _pick_device)
from train_continuous import TensorWindows, clip_dataset_arrays, load_bouts

VARIANTS = ["mean", "max", "meanmax", "attn", "last"]


def time_mask(x, lengths):
    """(B, T, 1) float mask, 1.0 on real frames. lengths is never None here --
    extraction always records it, and silently treating None as "all real" is how
    padding leaks back in."""
    steps = torch.arange(x.shape[1], device=x.device)[None, :]
    return (steps < lengths[:, None].to(x.device)).unsqueeze(-1).to(x.dtype)


class PoolLSTM(nn.Module):
    def __init__(self, mode="mean"):
        super().__init__()
        self.mode = mode
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        if mode == "attn":
            self.score = nn.Linear(HIDDEN_SIZE, 1)
        width = HIDDEN_SIZE * (2 if mode == "meanmax" else 1)
        self.head = nn.Sequential(
            nn.Linear(width + N_AGG_FEATURES, 64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64, NUM_CLASSES))

    def pool(self, out, lengths):
        m = time_mask(out, lengths)
        if self.mode == "mean":
            return (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        if self.mode == "max":
            # -inf on padding so zeros can never win the max
            return out.masked_fill(m == 0, float("-inf")).max(1).values
        if self.mode == "meanmax":
            mean = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
            mx = out.masked_fill(m == 0, float("-inf")).max(1).values
            return torch.cat([mean, mx], dim=-1)
        if self.mode == "attn":
            s = self.score(out).masked_fill(m == 0, float("-inf"))
            return (torch.softmax(s, dim=1) * out).sum(1)
        if self.mode == "last":
            idx = (lengths.to(out.device) - 1).clamp(min=0)
            return out[torch.arange(out.shape[0], device=out.device), idx]
        raise ValueError(self.mode)

    def forward(self, x, agg, lengths):
        out, _ = self.lstm(x)
        return self.head(torch.cat([self.pool(out, lengths), agg], dim=-1))


def train(ds, mode, epochs, seed, device):
    torch.manual_seed(seed)
    m = PoolLSTM(mode).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    m.train()
    for _ in range(epochs):
        for xb, ab, nb, yb in dl:
            opt.zero_grad()
            lossf(m(xb.to(device), ab.to(device), nb.to(device)), yb.to(device)).backward()
            opt.step()
    return m


@torch.no_grad()
def score(m, arrs, device):
    m.eval()
    pred = m(torch.from_numpy(arrs["X"]).to(device),
             torch.from_numpy(arrs["agg"]).to(device),
             torch.from_numpy(arrs["lengths"]).to(device)).argmax(1).cpu().numpy()
    y = arrs["y"]
    rec = {}
    for c in range(NUM_CLASSES):
        n = int((y == c).sum())
        rec[CLASS_NAMES[c]] = (int(((pred == c) & (y == c)).sum()) / n) if n else float("nan")
    return float((pred == y).mean()), rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    a = ap.parse_args()

    bouts = load_bouts(a.stride)
    tr = [k for k in bouts if k != a.holdout]
    cX, cA, cL, cY = clip_dataset_arrays()
    X = np.concatenate([cX] + [bouts[k]["train"]["X"] for k in tr])
    A = np.concatenate([cA] + [bouts[k]["train"]["agg"] for k in tr])
    L = np.concatenate([cL] + [bouts[k]["train"]["lengths"] for k in tr])
    Y = np.concatenate([cY] + [bouts[k]["train"]["y"] for k in tr])
    ds = TensorWindows(X, A, L, Y)
    device = _pick_device()
    print(f"held out bout {a.holdout}; train on clips + bouts {tr} = {len(Y)} windows")
    print(f"{a.seeds} seeds x {a.epochs} epochs, no class weighting, no prior\n", flush=True)

    hdr = f"{'pooling':<10}{'overall':>9}{'+-':>7}"
    for c in CLASS_NAMES:
        hdr += f"{c[:7]:>9}"
    print(hdr)
    best = None
    for v in VARIANTS:
        accs, recs = [], []
        for s in range(a.seeds):
            m = train(ds, v, a.epochs, 42 + s, device)
            acc, rec = score(m, bouts[a.holdout]["eval"], device)
            accs.append(acc); recs.append(rec)
        mean_rec = {c: float(np.nanmean([r[c] for r in recs])) for c in CLASS_NAMES}
        line = f"{v:<10}{np.mean(accs):>8.1%}{np.std(accs):>7.1%}"
        for c in CLASS_NAMES:
            line += f"{mean_rec[c]:>9.0%}"
        print(line, flush=True)
        if best is None or np.mean(accs) > best[1]:
            best = (v, float(np.mean(accs)))
    print(f"\nbest overall: {best[0]} at {best[1]:.1%}  (current shipped reduction is `mean`)")
    print("Per-class recall is what matters here, not just overall -- the question")
    print("was whether transient (lunge) and quiet (neutral) classes are being")
    print("destroyed by averaging, so look at those columns before the total.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
