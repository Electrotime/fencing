"""Does adding bout 7 (third venue, 204 parry windows) help? Matched A/B."""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, N_AGG_WIDE, _pick_device
from exp_opponent import train_eval, with_opponent

CONT = PROJECT / "data" / "train_continuous"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="4")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--extra", default="7", help="bout added in arm B")
    ap.add_argument("--mirror", action="store_true", help="match the shipped recipe")
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz"))
    for need in (a.holdout, a.extra):
        if need not in stems:
            print(f"no extracted bout {need!r}; have {stems}")
            return 1
    base = [s for s in stems if s not in (a.holdout, a.extra)]
    bouts = {s: with_opponent(CONT / f"{s}.npz") for s in stems}

    ev = bouts[a.holdout]
    ev_d = np.load(CONT / f"{a.holdout}.npz")

    def arm(use):
        X = np.concatenate([bouts[s][0]["X"][::a.stride] for s in use])
        A = np.concatenate([bouts[s][0]["wide"][::a.stride] for s in use])
        L = np.concatenate([bouts[s][0]["lengths"][::a.stride] for s in use])
        Y = np.concatenate([bouts[s][0]["y"][::a.stride] for s in use])
        if a.mirror:
            from exp_mirror import mirror_flat
            X = np.concatenate([X, mirror_flat(X)])
            A, L, Y = np.concatenate([A, A]), np.concatenate([L, L]), np.concatenate([Y, Y])
        return X, A, L, Y

    device = _pick_device()
    print(f"held out bout {a.holdout} ({len(ev_d['y'])} windows, "
          f"{int((ev_d['y'] == CLASS_NAMES.index('parry')).sum())} parry)")
    results = {}
    for tag, use in (("without", base), ("with " + a.extra, base + [a.extra])):
        X, A, L, Y = arm(use)
        n_par = int((Y == CLASS_NAMES.index("parry")).sum())
        accs, recs, precs = train_eval(X, A, L, Y, ev[0]["X"], ev[0]["wide"],
                                ev[0]["lengths"], ev[0]["y"], N_AGG_WIDE,
                                a.seeds, a.epochs, device, a.pool)
        results[tag] = (accs, recs, precs)
        print(f"  {tag:<10} bouts {use}  {len(Y)} windows ({n_par} parry): "
              f"{np.mean(accs):.1%} +-{np.std(accs):.1%}", flush=True)

    (ba, br, bp), (wa, wr, wp) = results["without"], results["with " + a.extra]
    print(f"\n=== held-out bout {a.holdout} ===")
    print(f"  overall   {np.mean(ba):.1%} -> {np.mean(wa):.1%}  "
          f"({100 * (np.mean(wa) - np.mean(ba)):+.2f} pts, seed sd "
          f"{100 * np.std(ba):.2f}/{100 * np.std(wa):.2f} pts)")
    print(f"  {'class':<10}{'recall base':>13}{'+b' + a.extra:>8}{'d':>8}"
          f"{'prec base':>12}{'+b' + a.extra:>8}{'d':>8}")
    for c in CLASS_NAMES:
        b, w = np.mean([r[c] for r in br]), np.mean([r[c] for r in wr])
        bq, wq = np.nanmean([r[c] for r in bp]), np.nanmean([r[c] for r in wp])
        star = "  <-- target" if c == "parry" else ""
        print(f"  {c:<10}{b:>13.0%}{w:>8.0%}{w - b:>+8.1%}"
              f"{bq:>12.0%}{wq:>8.0%}{wq - bq:>+8.1%}{star}")
    print("\nRead the seed sd before the delta.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
