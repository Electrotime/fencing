"""Train on clips + CONTINUOUS windows, validated leave-one-bout-out.

Every accuracy number in this project so far describes a model trained only on
hand-cut clips. This is the first training run that sees continuous footage.

Two things are being tested, and they are separable:

  1. DOES CONTINUOUS DATA HELP AT ALL?  clips-only vs clips+continuous, same
     recipe otherwise.
  2. CAN THE POST-HOC PRIOR BE RETIRED?  The shipped model uses inverse-frequency
     class weighting, which drives its effective prior to UNIFORM, and then
     demo_video multiplies CLASS_PRIOR back in to undo that (19.0% -> 41.9% when
     it was introduced). Continuous windows arrive at their NATURAL frequencies,
     so training on them unweighted should give the model a correct prior
     directly, and the post-hoc correction becomes unnecessary. Both weighting
     schemes are run so the question is answered rather than assumed.

VALIDATION IS BY BOUT, never by window. Windows are sampled every 5 frames from a
60-frame span, so neighbours share 92% of their frames -- a random split would put
near-duplicates on both sides and report a fantasy. Held-out bout = held-out
match, which is also the question that matters: does this work on footage it has
never seen?

Evaluation runs on the cached tensors from extract_continuous.py, so it is
instant. Only extraction is expensive, and that is cached.

usage: py -3 scripts/train_continuous.py [--epochs 80] [--seeds 3]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.action_model import (CLASS_NAMES, NUM_CLASSES, WEIGHT_DECAY, ActionLSTM,
                              FencingDataset, _pick_device)

CONT_DIR = PROJECT / "data" / "train_continuous"
KEYPOINTS = PROJECT / "data" / "keypoints"


class TensorWindows(Dataset):
    """Pre-extracted continuous windows, in FencingDataset's 4-tuple shape."""

    def __init__(self, X, agg, lengths, y):
        self.X, self.agg, self.lengths, self.y = X, agg, lengths, y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]), torch.from_numpy(self.agg[i]),
                int(self.lengths[i]), int(self.y[i]))


def load_bouts(stride=1):
    """Extracted bouts, optionally subsampled for TRAINING cost.

    Windows are emitted every PREDICT_EVERY=5 frames from a 60-frame span, so
    neighbours share 92% of their frames. Taking every 3rd still leaves 75%
    overlap -- almost no information is lost and training cost drops 3x. Applied
    to training data only; evaluation always uses every window, so the held-out
    numbers stay comparable to evaluate_labels.
    """
    out = {}
    for p in sorted(CONT_DIR.glob("*.npz")):
        d = np.load(p)
        full = dict(X=d["X"], agg=d["agg"], lengths=d["lengths"], y=d["y"])
        thin = {k: v[::stride] for k, v in full.items()}
        out[p.stem] = {"eval": full, "train": thin}
    return out


def clip_dataset_arrays():
    """FencingDataset as flat arrays so it can be concatenated with bout windows."""
    ds = FencingDataset(KEYPOINTS)
    X, A, L, Y = [], [], [], []
    for i in range(len(ds)):
        f, g, n, lab = ds[i]
        X.append(np.asarray(f, dtype=np.float32))
        A.append(np.asarray(g, dtype=np.float32))
        L.append(int(n)); Y.append(int(lab))
    return (np.stack(X), np.stack(A), np.array(L, dtype=np.int64),
            np.array(Y, dtype=np.int64))


def train_once(train_ds, weights, epochs, seed, device):
    torch.manual_seed(seed)
    model = ActionLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
    lossf = torch.nn.CrossEntropyLoss(
        weight=None if weights is None else torch.tensor(weights, dtype=torch.float32,
                                                         device=device))
    dl = DataLoader(train_ds, batch_size=32, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, ab, nb, yb in dl:
            opt.zero_grad()
            out = model(xb.to(device), ab.to(device), nb.to(device))
            loss = lossf(out, yb.to(device))
            loss.backward()
            opt.step()
    return model


# Shipped prior, applied to inverse-frequency models exactly as demo_video does.
# Comparing an inv-freq model RAW against a natural-prior model is not a fair
# test: inv-freq drives the effective prior to uniform on purpose and the
# correction is put back at inference. Judge each recipe as it would actually run.
SHIPPED_PRIOR = np.array([0.184, 0.045, 0.017, 0.122, 0.230, 0.401], dtype=np.float64)


@torch.no_grad()
def evaluate(model, arrs, device, apply_prior=False):
    model.eval()
    X = torch.from_numpy(arrs["X"]).to(device)
    A = torch.from_numpy(arrs["agg"]).to(device)
    L = torch.from_numpy(arrs["lengths"]).to(device)
    logits = model(X, A, L)
    if apply_prior:
        p = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float64)
        q = p * (SHIPPED_PRIOR / (1.0 / NUM_CLASSES))[None, :]
        pred = (q / q.sum(1, keepdims=True)).argmax(1)
    else:
        pred = logits.argmax(1).cpu().numpy()
    y = arrs["y"]
    per = {}
    for c in range(NUM_CLASSES):
        n = int((y == c).sum()); pp = int((pred == c).sum())
        tp = int(((pred == c) & (y == c)).sum())
        per[CLASS_NAMES[c]] = (n, tp / pp if pp else float("nan"),
                               tp / n if n else float("nan"))
    return float((pred == y).mean()), per


def inv_freq_weights(y):
    """Inverse-frequency weights -- the current recipe. Drives the effective prior
    to uniform, which is why demo_video has to multiply CLASS_PRIOR back in."""
    freq = np.array([max(1, int((y == c).sum())) for c in range(NUM_CLASSES)])
    w = freq.sum() / (NUM_CLASSES * freq)
    return w.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--stride", type=int, default=3,
                    help="subsample TRAINING windows; neighbours overlap 92%%")
    a = ap.parse_args()

    bouts = load_bouts(a.stride)
    if not bouts:
        print(f"no extracted bouts in {CONT_DIR} -- run extract_continuous.py first")
        return 1
    device = _pick_device()
    cX, cA, cL, cY = clip_dataset_arrays()
    print(f"clips: {len(cY)} windows | continuous (eval): "
          + ", ".join(f"{k}={len(v['eval']['y'])}" for k, v in bouts.items()))
    print(f"training stride {a.stride} -> "
          + ", ".join(f"{k}={len(v['train']['y'])}" for k, v in bouts.items()))
    print(f"device {device}, {a.epochs} epochs, {a.seeds} seeds, "
          f"{len(bouts) * 3 * a.seeds} training runs\n", flush=True)

    rows = []
    for held in bouts:
        tr = [k for k in bouts if k != held]
        bX = np.concatenate([bouts[k]["train"]["X"] for k in tr])
        bA = np.concatenate([bouts[k]["train"]["agg"] for k in tr])
        bL = np.concatenate([bouts[k]["train"]["lengths"] for k in tr])
        bY = np.concatenate([bouts[k]["train"]["y"] for k in tr])

        cc = lambda *xs: np.concatenate(xs)
        both = (cc(cX, bX), cc(cA, bA), cc(cL, bL), cc(cY, bY))
        # (training arrays, inverse-frequency weighting?, apply CLASS_PRIOR at eval?)
        # Each recipe is judged the way it would actually be deployed.
        setups = {
            "clips only, inv-freq+prior": (cX, cA, cL, cY, True, True),
            "clips+cont, inv-freq+prior": (*both, True, True),
            "clips+cont, natural (no prior)": (*both, False, False),
        }
        for name, (X, A, L, Y, weighted, prior) in setups.items():
            accs, per = [], None
            for s in range(a.seeds):
                m = train_once(TensorWindows(X, A, L, Y),
                               inv_freq_weights(Y) if weighted else None,
                               a.epochs, 42 + s, device)
                # evaluate on EVERY window of the held-out bout, never the thinned
                # set, so these numbers line up with evaluate_labels
                acc, per = evaluate(m, bouts[held]["eval"], device, apply_prior=prior)
                accs.append(acc)
            rows.append((held, name, float(np.mean(accs)), float(np.std(accs)), per))
            print(f"  held-out {held:<6} {name:<32} {np.mean(accs):.1%} "
                  f"(+-{np.std(accs):.1%} over {a.seeds} seeds)", flush=True)
        print()

    print("=== summary: mean over held-out bouts ===")
    for name in ("clips only, inv-freq+prior", "clips+cont, inv-freq+prior",
                 "clips+cont, natural (no prior)"):
        v = [r[2] for r in rows if r[1] == name]
        print(f"  {name:<32}{np.mean(v):.1%}")
    print("\nEach recipe is scored as it would be DEPLOYED: inv-freq models get")
    print("CLASS_PRIOR multiplied in (as demo_video does); the natural-prior model")
    print("gets nothing, because training on real frequencies is the point of it.")
    print("\nPer-bout detail (held-out bout = held-out MATCH; windows overlap 92%,")
    print("so a random split would leak near-duplicates and report a fantasy):")
    for held, name, mean, sd, per in rows:
        if "natural" not in name:
            continue
        print(f"  bout {held} (natural): " + "  ".join(
            f"{c}={per[c][2]:.0%}" for c in CLASS_NAMES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
