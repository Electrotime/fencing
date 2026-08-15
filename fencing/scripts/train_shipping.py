"""Train the shippable checkpoint on clips + continuous windows."""
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
from train_continuous import (CONT_DIR, TensorWindows, clip_dataset_arrays,
                              load_bouts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default=None, help="bout stem to exclude, e.g. 1")
    ap.add_argument("--ship", action="store_true", help="train on every bout")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--members", type=int, default=4)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last", choices=ActionLSTM.POOL_MODES)
    ap.add_argument("--opponent", action="store_true")
    ap.add_argument("--no-clips", action="store_true")
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

    use_clips = not (a.no_clips or a.opponent)
    if a.opponent:
        from exp_opponent import with_opponent
        wide = {k: with_opponent(CONT_DIR / f"{k}.npz")[0] for k in use}
        X = np.concatenate([wide[k]["X"][::a.stride] for k in use])
        A = np.concatenate([wide[k]["wide"][::a.stride] for k in use])
        L = np.concatenate([wide[k]["lengths"][::a.stride] for k in use])
        Y = np.concatenate([wide[k]["y"][::a.stride] for k in use])
        cY = []
    else:
        cX, cA, cL, cY = clip_dataset_arrays()
        pre = ([cX], [cA], [cL], [cY]) if use_clips else ([], [], [], [])
        if not use_clips:
            cY = []
        X = np.concatenate(pre[0] + [bouts[k]["train"]["X"] for k in use])
        A = np.concatenate(pre[1] + [bouts[k]["train"]["agg"] for k in use])
        L = np.concatenate(pre[2] + [bouts[k]["train"]["lengths"] for k in use])
        Y = np.concatenate(pre[3] + [bouts[k]["train"]["y"] for k in use])

    device = _pick_device()
    print(f"training on clips({len(cY)}) + bouts {use} = {len(Y)} windows, "
          f"agg width {A.shape[1]}" + ("  [OPPONENT]" if a.opponent else ""))
    c = Counter(Y.tolist())
    print("  class mix: " + "  ".join(
        f"{n}={c.get(i, 0)}({c.get(i, 0) / len(Y):.0%})" for i, n in enumerate(CLASS_NAMES)))
    print(f"  {a.members} members x {a.epochs} epochs, pool={a.pool}, "
          f"no class weighting, no best-epoch selection", flush=True)

    ds = TensorWindows(X, A, L, Y)
    for m in range(a.members):
        torch.manual_seed(42 + m)
        model = ActionLSTM(pool=a.pool, n_agg=A.shape[1]).to(device)
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
