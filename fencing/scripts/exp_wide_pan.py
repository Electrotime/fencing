"""Wide pan strips (0.44) vs shipped (0.22), leave-one-bout-out. See CLAUDE.md."""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, N_AGG_WIDE, _pick_device
from exp_mirror import mirror_flat
from exp_mirror_venue import fit, predict
from exp_opponent import with_opponent

CONT = PROJECT / "data" / "train_continuous"
ADV, WALK = CLASS_NAMES.index("advance"), CLASS_NAMES.index("walking")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--mirror", action="store_true", default=True)
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    missing = [s for s in stems if "agg_wide" not in np.load(CONT / f"{s}.npz").files]
    if missing:
        print(f"no agg_wide in {missing} -- re-extract those bouts first")
        return 1

    raw = {k: {s: with_opponent(CONT / f"{s}.npz", k)[0] for s in stems}
           for k in ("agg", "agg_wide")}
    device = _pick_device()
    rows = {}

    for holdout in stems:
        train = [s for s in stems if s != holdout]
        rows[holdout] = {}
        for key in ("agg", "agg_wide"):
            r = raw[key]
            X = np.concatenate([r[s]["X"][::a.stride] for s in train])
            A = np.concatenate([r[s]["wide"][::a.stride] for s in train])
            L = np.concatenate([r[s]["lengths"][::a.stride] for s in train])
            Y = np.concatenate([r[s]["y"][::a.stride] for s in train])
            if a.mirror:
                X = np.concatenate([X, mirror_flat(X)])
                A, L, Y = np.concatenate([A, A]), np.concatenate([L, L]), np.concatenate([Y, Y])
            ev = r[holdout]
            accs, advr, a2w = [], [], []
            for s in range(a.seeds):
                m = fit(X, A, L, Y, 42 + s, a.epochs, device, a.pool)
                p = predict(m, ev["X"], ev["wide"], ev["lengths"], device)
                accs.append(float((p == ev["y"]).mean()))
                n_adv = int((ev["y"] == ADV).sum())
                if n_adv:
                    advr.append(float(((p == ADV) & (ev["y"] == ADV)).sum()) / n_adv)
                    a2w.append(float(((p == WALK) & (ev["y"] == ADV)).sum()) / n_adv)
            rows[holdout][key] = (np.mean(accs), np.std(accs),
                                  np.mean(advr) if advr else np.nan,
                                  np.mean(a2w) if a2w else np.nan)
        print(f"  bout {holdout} done", flush=True)

    print(f"\n  {'bout':<6}{'narrow':>16}{'wide':>16}{'delta':>8}"
          f"{'adv R n/w':>14}{'adv->walk n/w':>16}")
    dl = []
    for h, r in rows.items():
        n, w = r["agg"], r["agg_wide"]
        dl.append(w[0] - n[0])
        print(f"  {h:<6}{n[0]:>11.1%}±{n[1]:<4.1%}{w[0]:>11.1%}±{w[1]:<4.1%}"
              f"{100 * (w[0] - n[0]):>+8.2f}"
              f"{n[2]:>7.0%}/{w[2]:<6.0%}{n[3]:>8.0%}/{w[3]:<7.0%}")
    print(f"\n  mean delta {100 * np.mean(dl):+.2f} pts, "
          f"worst bout {100 * min(dl):+.2f}, best {100 * max(dl):+.2f}")
    print("  ship only if the mean is positive and no bout regresses beyond seed noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
