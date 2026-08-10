"""Train the shippable checkpoint on clips + continuous windows.

Two modes, because "the model we ship" and "the number we can honestly quote" are
different objects:

  --holdout N   train on clips + every bout EXCEPT N. Used to VERIFY that the
                offline tensor evaluation in train_continuous.py matches a real
                end-to-end evaluate_labels run on bout N. Nothing is shipped from
                this; it exists to catch train/serve disagreement.

  --ship        train on clips + ALL bouts. Strictly more data, so it should be at
                least as good -- but it cannot then be honestly scored on those
                bouts. The honest generalisation estimate remains the
                leave-one-bout-out figure (60.4%), already measured.

NO BEST-EPOCH SELECTION, deliberately. load_action_model's docstring records that
picking checkpoints by validation accuracy reliably lands on lunge-heavy models
(seed 8 -> 52% lunge against a 42% average) and that validation accuracy and demo
behaviour are anti-correlated here. Fixed epochs plus seed ensembling avoids
choosing on a metric known to mislead.

NO CLASS WEIGHTING, also deliberately. Continuous windows arrive at their natural
frequencies; training on those gives the model a correct prior directly, and
measured identically (60.4%) to inverse-frequency weighting plus the post-hoc
CLASS_PRIOR. Set APPLY_CLASS_PRIOR=False in demo_video when using this checkpoint
-- applying it on top would correct a prior that is already right.

usage:
  py -3 scripts/train_shipping.py --holdout 1 --out models/verify_h1.pth
  py -3 scripts/train_shipping.py --ship --out models/action_cont.pth
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, WEIGHT_DECAY, ActionLSTM, _pick_device
from train_continuous import TensorWindows, clip_dataset_arrays, load_bouts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None, help="bout stem to exclude, e.g. 1")
    ap.add_argument("--ship", action="store_true", help="train on every bout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3)
    # A checkpoint's pooling mode is part of its identity: all modes share the same
    # parameter shapes, so loading with the wrong one succeeds silently and behaves
    # wrong. Whatever is used here must match demo_video.POOL_MODE.
    ap.add_argument("--pool", default="last", choices=ActionLSTM.POOL_MODES)
    a = ap.parse_args()
    if bool(a.holdout) == bool(a.ship):
        print("pick exactly one of --holdout N or --ship")
        return 2

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bouts = load_bouts(a.stride)
    use = [k for k in bouts if k != a.holdout] if a.holdout else list(bouts)
    if a.holdout and a.holdout not in bouts:
        print(f"no extracted bout {a.holdout!r}; have {sorted(bouts)}")
        return 1

    cX, cA, cL, cY = clip_dataset_arrays()
    X = np.concatenate([cX] + [bouts[k]["train"]["X"] for k in use])
    A = np.concatenate([cA] + [bouts[k]["train"]["agg"] for k in use])
    L = np.concatenate([cL] + [bouts[k]["train"]["lengths"] for k in use])
    Y = np.concatenate([cY] + [bouts[k]["train"]["y"] for k in use])

    device = _pick_device()
    print(f"training on clips({len(cY)}) + bouts {use} = {len(Y)} windows")
    c = Counter(Y.tolist())
    print("  class mix: " + "  ".join(
        f"{n}={c.get(i, 0)}({c.get(i, 0) / len(Y):.0%})" for i, n in enumerate(CLASS_NAMES)))
    print(f"  {a.members} members x {a.epochs} epochs, pool={a.pool}, "
          f"no class weighting, no best-epoch selection", flush=True)

    ds = TensorWindows(X, A, L, Y)
    for m in range(a.members):
        torch.manual_seed(42 + m)
        model = ActionLSTM(pool=a.pool).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
        lossf = torch.nn.CrossEntropyLoss()
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        model.train()
        for _ in range(a.epochs):
            for xb, ab, nb, yb in dl:
                opt.zero_grad()
                lossf(model(xb.to(device), ab.to(device), nb.to(device)),
                      yb.to(device)).backward()
                opt.step()
        p = out.with_name(f"{out.stem}.m{m}.pth")
        torch.save(model.state_dict(), p)
        print(f"  saved {p.name}", flush=True)

    # a bare copy at the stem too, so load_action_model works even if the members
    # are ever moved away
    torch.save(model.state_dict(), out)
    print(f"saved {out.name} (+ {a.members} ensemble members)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
