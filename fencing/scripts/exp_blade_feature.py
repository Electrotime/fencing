"""Does blade/torso motion energy help as a 7th ENGINEERED FEATURE? Measure, don't assume.

Blade energy separates hand-cut parry INTERVALS well -- pooled AUC 0.79 over 70
parries (`pool_blade_energy.py`). That is not the question the model faces. The model
sees fixed 2 s WINDOWS, and measured on those the same feature is chance:

    blade/torso p90 over the window, parry vs non-blade
      span 2.00 s   bout3 0.34  bout4 0.49  bout5 0.43   pooled 0.53
      span 0.35 s   bout3 0.51  bout4 0.65  bout5 0.57   pooled 0.66

The gap is DILUTION: a parry lasts ~0.6 s, so a p90 over 2 s is mostly reporting
whatever else was in the window. Shortening the span recovers a good part of it,
monotonically, which is the same reasoning that made `last` pooling beat `mean`.
BLADE_SPAN is therefore 0.35 s, not the full window -- deliberately a different
temporal footprint from the other six features.

Expectations are low and stated up front so they cannot be revised afterwards:
0.66 pooled is weak, bout 3 sits at chance at every span, and `stance_ratio` had the
best AUC ever tried here (0.91) and still made the model WORSE. AUC says a signal
exists, never that the model lacks it.

WHY THIS RUNS OFFLINE. Blade energy needs PIXELS, which `_engineered_features` never
sees -- it takes keypoints and a motion track. Wiring it into the live pipeline means
a third per-frame deque on FencerTrack and changes to extraction, the demo and the
scorer. That is a lot of plumbing to install before knowing whether it pays, so this
joins two existing caches by (slot, time) instead -- exactly how exp_opponent.py
tested opponent context before it was built.

  data/train_continuous/<stem>.npz   windows: X, agg, lengths, y, time, slot
  data/labels/<stem>_blade.npz       per frame: blade, torso, time, slot

Layout with --opponent (the shipped recipe): [own(7) | opponent(7) | present(1)] = 15,
against the current 13. The opponent's blade energy is included on purpose -- a parry
is a response to THEIR extension, which is what the parry gate exploits.

usage: py -3 scripts/exp_blade_feature.py --holdout 4 [--seeds 2] [--span 0.35]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES, N_AGG_FEATURES, _pick_device
from exp_opponent import WideAggLSTM, train_eval  # noqa: F401  (harness reused verbatim)

CONT = PROJECT / "data" / "train_continuous"
LAB = PROJECT / "data" / "labels"
BLADE_SPAN = 0.35        # seconds back from the window's newest frame; see docstring
MIN_FRAMES = 2           # fewer than this in the span -> feature undefined
NEUTRAL_FILL = 1.0       # "blade no hotter than torso"; measured to be needed on <1%


def blade_feature(stem, times, slots, span=BLADE_SPAN):
    """p90 of blade/torso over the last `span` seconds of each window.

    Returns (values, n_missing). Joined by (slot, time) rather than recomputed, so
    this cannot drift from what pool_blade_energy.py scored.
    """
    cache = LAB / f"{stem}_blade.npz"
    if not cache.exists():
        raise SystemExit(f"no {cache.name}; run scripts/blade_energy.py on bout {stem}")
    e = np.load(cache)
    et, es = e["time"], e["slot"].astype(str)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = e["blade"] / np.maximum(e["torso"], 1e-6)

    out = np.full(len(times), np.nan, dtype=np.float32)
    for sl in ("A", "B"):
        m = es == sl
        t_s, r_s = et[m], ratio[m]
        order = np.argsort(t_s)
        t_s, r_s = t_s[order], r_s[order]
        wm = np.flatnonzero(slots.astype(str) == sl)
        if len(wm) == 0:
            continue
        lo = np.searchsorted(t_s, times[wm] - span, side="left")
        hi = np.searchsorted(t_s, times[wm], side="right")
        for k, i in enumerate(wm):
            v = r_s[lo[k]:hi[k]]
            v = v[np.isfinite(v)]
            if len(v) >= MIN_FRAMES:
                out[i] = np.percentile(v, 90)
    missing = int(np.isnan(out).sum())
    out[np.isnan(out)] = NEUTRAL_FILL
    return out, missing


def load_bout(stem, span):
    """Windows plus the blade column, and the 7-wide own-feature block."""
    d = np.load(CONT / f"{stem}.npz")
    bf, missing = blade_feature(stem, d["time"], d["slot"], span)
    own7 = np.concatenate([d["agg"], bf[:, None]], axis=1).astype(np.float32)
    return dict(X=d["X"], agg6=d["agg"], agg7=own7, lengths=d["lengths"],
                y=d["y"], time=d["time"], slot=d["slot"]), missing


def widen(bout, use_blade):
    """[own | opponent | present] with the opponent matched on (slot, time).

    Re-implemented rather than calling exp_opponent.with_opponent because that reads
    the npz off disk and would not see the appended blade column. The pairing rule is
    identical: exact float time match, safe because both slots' times come from the
    same idx/fps on the same frame.
    """
    own = bout["agg7"] if use_blade else bout["agg6"]
    slot, time = bout["slot"].astype(str), bout["time"]
    index = {(s, float(t)): i for i, (s, t) in enumerate(zip(slot, time))}
    opp = np.zeros_like(own)
    present = np.zeros((len(own), 1), dtype=np.float32)
    for i, (s, t) in enumerate(zip(slot, time)):
        j = index.get(("B" if s == "A" else "A", float(t)))
        if j is not None:
            opp[i] = own[j]
            present[i] = 1.0
    return np.concatenate([own, opp, present], axis=1).astype(np.float32), float(present.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="4")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last")
    ap.add_argument("--span", type=float, default=BLADE_SPAN)
    # THE CONTROL THAT DECIDES THIS. Adding a 7th feature also makes the head's first
    # Linear wider, so "accuracy went up" can mean capacity rather than information.
    # --shuffle permutes the blade column WITHIN each bout: identical marginal
    # distribution, identical parameter count, zero alignment to the window it
    # describes. If the gain survives that, it was never the blade.
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()

    stems = sorted(p.stem for p in CONT.glob("*.npz")
                   if (LAB / f"{p.stem}_blade.npz").exists())
    if a.holdout not in stems:
        print(f"holdout {a.holdout!r} has no blade cache; have {stems}")
        return 1
    train = [s for s in stems if s != a.holdout]

    bouts, miss_tot, n_tot = {}, 0, 0
    for s in stems:
        bouts[s], miss = load_bout(s, a.span)
        if a.shuffle:
            rng = np.random.default_rng(1234)
            col = bouts[s]["agg7"][:, -1].copy()
            rng.shuffle(col)
            bouts[s]["agg7"][:, -1] = col
        miss_tot += miss
        n_tot += len(bouts[s]["y"])
    print(f"bouts {stems}, holding out {a.holdout}; blade span {a.span:.2f}s"
          + ("   [SHUFFLED CONTROL -- feature carries no information]" if a.shuffle else ""))
    print(f"  feature undefined on {miss_tot}/{n_tot} windows "
          f"({miss_tot / n_tot:.2%}), filled with {NEUTRAL_FILL}")

    device = _pick_device()
    results = {}
    for use_blade in (False, True):
        n_agg = (N_AGG_FEATURES + int(use_blade)) * 2 + 1
        tr = [widen(bouts[s], use_blade)[0] for s in train]
        ev_A, cov = widen(bouts[a.holdout], use_blade)
        X = np.concatenate([bouts[s]["X"][::a.stride] for s in train])
        A = np.concatenate([t[::a.stride] for t in tr])
        L = np.concatenate([bouts[s]["lengths"][::a.stride] for s in train])
        Y = np.concatenate([bouts[s]["y"][::a.stride] for s in train])
        ev = bouts[a.holdout]
        accs, recs = train_eval(X, A, L, Y, ev["X"], ev_A, ev["lengths"], ev["y"],
                                n_agg, a.seeds, a.epochs, device, a.pool)
        results[use_blade] = (accs, recs)
        tag = "WITH blade" if use_blade else "baseline  "
        print(f"  {tag} (n_agg={n_agg:2d}, opponent on {cov:.0%}): "
              f"{np.mean(accs):.1%} +-{np.std(accs):.1%}", flush=True)

    b_acc, b_rec = results[False]
    w_acc, w_rec = results[True]
    print(f"\n=== held-out bout {a.holdout} ===")
    # points, not a fraction -- the first version of this line printed "+0.0 pts"
    # for a real +1.6, which is the kind of thing that gets a result waved through
    print(f"  overall   {np.mean(b_acc):.1%} -> {np.mean(w_acc):.1%}  "
          f"({100 * (np.mean(w_acc) - np.mean(b_acc)):+.2f} pts, seed sd "
          f"{100 * np.std(b_acc):.2f}/{100 * np.std(w_acc):.2f} pts)")
    print(f"  {'class':<10}{'recall base':>13}{'+blade':>10}{'delta':>9}")
    for c in CLASS_NAMES:
        b = np.mean([r[c] for r in b_rec])
        w = np.mean([r[c] for r in w_rec])
        star = "  <-- the target" if c == "parry" else ""
        print(f"  {c:<10}{b:>13.0%}{w:>10.0%}{w - b:>+9.1%}{star}")
    print("\nRead the SEED SD before the delta. This project rejected mirroring at "
          "+0.5 pts\nagainst +-1.5 noise, and one bout is one bout -- run both "
          "holdouts before believing it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
