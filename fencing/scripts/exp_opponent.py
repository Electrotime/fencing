"""Give each fencer the OPPONENT's state as input. Fencing is interactive."""
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
                              NUM_CLASSES, WEIGHT_DECAY, _pick_device)
from train_continuous import TensorWindows, clip_dataset_arrays, train_once, evaluate

CONT = PROJECT / "data" / "train_continuous"


class WideAggLSTM(nn.Module):
    """ActionLSTM with a wider agg block. Trunk and pooling identical so the
    comparison isolates the extra input, not capacity in the recurrent part."""

    def __init__(self, n_agg, pool="last"):
        super().__init__()
        self.pool_mode = pool
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + n_agg, 64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64, NUM_CLASSES))

    def forward(self, x, agg, lengths):
        out, _ = self.lstm(x)
        steps = torch.arange(x.shape[1], device=x.device)[None, :]
        m = (steps < lengths[:, None].to(x.device)).unsqueeze(-1).to(out.dtype)
        if self.pool_mode == "mean":
            pooled = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        elif self.pool_mode == "max":
            pooled = out.masked_fill(m == 0, float("-inf")).max(1).values
        else:
            idx = (lengths.to(out.device) - 1).clamp(min=0)
            pooled = out[torch.arange(out.shape[0], device=out.device), idx]
        return self.head(torch.cat([pooled, agg], dim=-1))


def with_opponent(npz_path, key="agg"):
    """Load a bout and widen agg to [own(6) | opponent(6) | present(1)]."""
    d = np.load(npz_path)
    agg, slot, time = d[key], d["slot"], d["time"]
    # exact float times come straight from idx/fps for both slots on the same
    # frame, so an exact key match is safe here -- no tolerance needed
    index = {(str(s), float(t)): i for i, (s, t) in enumerate(zip(slot, time))}
    opp = np.zeros_like(agg)
    present = np.zeros((len(agg), 1), dtype=np.float32)
    for i, (s, t) in enumerate(zip(slot, time)):
        j = index.get(("B" if str(s) == "A" else "A", float(t)))
        if j is not None:
            opp[i] = agg[j]
            present[i] = 1.0
    wide = np.concatenate([agg, opp, present], axis=1).astype(np.float32)
    return dict(X=d["X"], agg=agg, wide=wide, lengths=d["lengths"], y=d["y"]), \
        float(present.mean())


def train_eval(X, A, L, Y, ev_X, ev_A, ev_L, ev_Y, n_agg, seeds, epochs, device, pool):
    accs, recs = [], []
    for s in range(seeds):
        torch.manual_seed(42 + s)
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
        model.eval()
        with torch.no_grad():
            pred = model(torch.from_numpy(ev_X).to(device),
                         torch.from_numpy(ev_A).to(device),
                         torch.from_numpy(ev_L).to(device)).argmax(1).cpu().numpy()
        accs.append(float((pred == ev_Y).mean()))
        recs.append({CLASS_NAMES[c]: (int(((pred == c) & (ev_Y == c)).sum()) / n
                                      if (n := int((ev_Y == c).sum())) else float("nan"))
                     for c in range(NUM_CLASSES)})
    return accs, recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="1")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    a = ap.parse_args()

    bouts, cover = {}, {}
    for p in sorted(CONT.glob("*.npz")):
        bouts[p.stem], cover[p.stem] = with_opponent(p)
    print("opponent present on: " + "  ".join(f"{k}={v:.0%}" for k, v in cover.items()))

    tr = [k for k in bouts if k != a.holdout]
    ev = bouts[a.holdout]
    th = lambda k, f: bouts[k][f][::a.stride]
    cX, cA, cL, cY = clip_dataset_arrays()
    # clips have no opponent: zeros + presence flag 0
    cW = np.concatenate([cA, np.zeros_like(cA), np.zeros((len(cA), 1), np.float32)],
                        axis=1).astype(np.float32)

    device = _pick_device()
    print(f"held out bout {a.holdout} ({len(ev['y'])} eval windows), pool={a.pool}\n")
    print(f"{'setup':<26}{'overall':>9}{'+-':>7}" + "".join(f"{c[:7]:>9}" for c in CLASS_NAMES))

    cat = lambda *xs: np.concatenate(xs)
    runs = [
        ("clips+cont, own only", cat(cX, *[th(k, "X") for k in tr]),
         cat(cA, *[th(k, "agg") for k in tr]), 6, ev["agg"]),
        ("clips+cont, + opponent", cat(cX, *[th(k, "X") for k in tr]),
         cat(cW, *[th(k, "wide") for k in tr]), 13, ev["wide"]),
        # no clips: the arm that cannot use "all-zero opponent" as a source tell
        ("cont only, own only", cat(*[th(k, "X") for k in tr]),
         cat(*[th(k, "agg") for k in tr]), 6, ev["agg"]),
        ("cont only, + opponent", cat(*[th(k, "X") for k in tr]),
         cat(*[th(k, "wide") for k in tr]), 13, ev["wide"]),
    ]
    for name, X, A, n_agg, ev_A in runs:
        L = (cat(cL, *[th(k, "lengths") for k in tr]) if "clips" in name
             else cat(*[th(k, "lengths") for k in tr]))
        Y = (cat(cY, *[th(k, "y") for k in tr]) if "clips" in name
             else cat(*[th(k, "y") for k in tr]))
        accs, recs = train_eval(X, A, L, Y, ev["X"], ev_A, ev["lengths"], ev["y"],
                                n_agg, a.seeds, a.epochs, device, a.pool)
        line = f"{name:<26}{np.mean(accs):>8.1%}{np.std(accs):>7.1%}"
        for c in CLASS_NAMES:
            line += f"{np.nanmean([r[c] for r in recs]):>9.0%}"
        print(line, flush=True)
    print("\nCompare the two PAIRS separately. The `cont only` pair is the clean")
    print("test: no clips, so an all-zero opponent block cannot double as a")
    print("marker for which corpus a window came from.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
