"""Go/no-go for the touch predictor: does the action probability at halt time separate left/right?"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES
import check_touches as CT
import read_scoreboard as RS

LAB = PROJECT / "data" / "labels"
CACHE = {s: f"{s}_probs_mirror.npz" for s in ("1","4","5","6","7","8","9","10","11","12")}
BACKS = (1.0, 2.0, 3.0, 4.0)
POOLS = ("max", "mean")

# FAILED on bout 4, 2026-08-20: AUC 0.52, one-sided p 0.42. Kept as the record.
DEAD_PREREG = "advance max (A-B) @1s"

# Replacement, registered 2026-08-20 on bouts 4+7, to be tested on 5 or 6. Averaging
# every lookback removes the choice that killed the first one -- advance led both
# searches but peaked at 1s on bout 7 and 4s on bout 4.
PREREG = "advance max (A-B) mean-lookback"
SELECTED_ON = ("7", "4")

# Registered 2026-08-20 on bouts 4+7. Foil priority goes to whoever was attacking,
# so two-lamp touches are the ones a pose feature could decide; single-lamp touches
# mix counter-attacks, ripostes and lines, and showed nothing (AUC 0.57).
PREREG_LIGHTS = "both"

# incompleteness, not bad rows: the surviving rows stay usable
# Registered 2026-08-22 from Aaron's statement of the right-of-way rule, BEFORE
# 8_probs_mirror.npz and 9_probs_mirror.npz existed -- the pose extraction for
# bouts 8 and 9 was still running. Priority goes to whoever went forward FIRST,
# which is a question of ORDER; every feature tried so far pools by max or mean
# and is order-blind. One-sided: A's advance mass earlier -> A holds priority.
# Not yet covered: the parry transfer rule (a defender's parry takes priority
# back, unless the attacker parries simultaneously, which is a beat).
ONSET_PREREG = "advance onset lead (A first) @4s"

# Registered 2026-08-27, BEFORE any frame-level cache existed for bouts 5,6,8,9,10.
# Why a retry: the window model reports a 2 s window at every timestep, and its
# advance probability decays with a 1.58 s autocorrelation time. Foil lockout is
# 0.30 s, so the instrument is ~5x too coarse to time an onset at all. This asks
# whether per-frame resolution rescues the SAME statistic, unchanged: time-centroid
# of advance probability over [t-4.0, t+0.3], B minus A, one-sided, A earlier -> A
# holds priority. Same halts, same direction. Only the probability source changes.
# Third registered feature, so the family correction becomes x3.
# SUPERSEDED 2026-08-27, before any run: models/action_frame.pth expects 6 agg
# features (head 134) and the pipeline now feeds 13 (141). It predates wide_agg by
# three weeks and nothing in the repo trains a replacement. Registered source is
# now the CURRENT model over a 25-frame window (0.83 s) instead of 60 (2.00 s).
# Statistic, direction, halts and subset all unchanged. Trades blur for noise: the
# model was trained at 60 frames, so 25 is off-distribution. That is the cost of
# the only resolution test available without retraining.
ONSET_SHORT_PREREG = "advance onset lead (A first) @4s [25-frame window]"

ADVISORY = ("checksum cannot run", "checksum seeded from here", "a row is probably missing")


def ranks(x):
    """Average ranks, ties shared."""
    order = np.argsort(x, kind="mergesort")
    s = np.asarray(x, dtype=float)[order]
    r = np.empty(len(x), dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def auc(scores, pos):
    """Mann-Whitney AUC of `scores` against boolean `pos`; 0.5 is chance."""
    pos = np.asarray(pos, dtype=bool)
    n1 = int(pos.sum())
    if n1 == 0 or n1 == len(pos):
        return float("nan")
    return float((ranks(scores)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(pos) - n1)))


def maxstat_p(X, pos, n_perm=20000, seed=0):
    """Per-feature and family-wise p-values from one label-shuffling null.

    Ranks are invariant to label permutation, so the whole null is a matmul.
    The family-wise value uses the max |AUC-0.5| across features, which is the
    statistic the eyeball actually applies when scanning the table.
    """
    pos = np.asarray(pos, dtype=bool)
    n, n1 = len(pos), int(pos.sum())
    R = np.stack([ranks(x) for x in X])
    denom, corr = n1 * (n - n1), n1 * (n1 + 1) / 2.0
    obs = np.abs((R @ pos - corr) / denom - 0.5)

    rng = np.random.default_rng(seed)
    M = np.stack([rng.permutation(pos) for _ in range(n_perm)], axis=1)
    null = np.abs((R @ M - corr) / denom - 0.5)
    per_feat = (null >= obs[:, None]).sum(axis=1) + 1
    fam = (null.max(axis=0) >= obs.max()) .sum() + 1
    return obs, per_feat / (n_perm + 1), fam / (n_perm + 1)


def pooled(d, t, back, lead):
    """Per-slot max and mean of each class probability over [t-back, t+lead]."""
    slot, time, probs = d["slot"].astype(str), d["time"], d["probs"]
    out = {}
    for s in ("A", "B"):
        m = (slot == s) & (time >= t - back) & (time <= t + lead)
        if not m.any():
            return None
        out[s] = (probs[m].max(axis=0), probs[m].mean(axis=0))
    return out


ADV_I = CLASS_NAMES.index("advance")


def onset_centroid(d, t, back=4.0, lead=0.3):
    """Time-centroid of each fencer's advance probability, B minus A.

    Right of way goes to whoever went forward FIRST, so the deciding quantity is
    ORDER, which a max over a window cannot see -- it only knows who advanced
    harder. The centroid is threshold-free and needs no tuning: if A's advance
    mass sits earlier, centroid_A is smaller and this comes out positive.
    The longest window is used because only it can contain the start.
    """
    slot, time, probs = d["slot"].astype(str), d["time"], d["probs"]
    out = {}
    for s in ("A", "B"):
        m = (slot == s) & (time >= t - back) & (time <= t + lead)
        if not m.any():
            return None
        w = probs[m, ADV_I].astype(float)
        tt = time[m].astype(float)
        out[s] = float((tt * w).sum() / w.sum()) if w.sum() > 0 else float(tt.mean())
    return out["B"] - out["A"]


def lights_for(stem, times):
    """left / right / both / none per halt, read off the lamp indicator."""
    lay = RS.LAYOUT.get(stem, {})
    cache = LAB / f"{stem}_lamp.npz"
    if "lamp" not in lay or not cache.exists():
        return np.array(["?"] * len(times))
    t, ser = RS.lamp_series("", lay["lamp"], 0.1, cache)
    thr = RS.lamp_thresholds(ser)
    return np.array([RS.lamps_at(t, ser, u, thr)[0] for u in times])


def load(stem):
    src = LAB / f"bout{stem}_touches.tsv"
    if not src.exists():
        src = LAB / f"bout{stem}_touches.csv"
    problems, rows = CT.check(src)
    hard = [m for _, m in problems if not any(k in m for k in ADVISORY)]
    if hard:
        raise SystemExit(f"{src.name} has unresolved problems: {hard}")
    advisory = len(problems) - len(hard)
    if advisory:
        print(f"  ({advisory} advisory: halts known missing, so this bout is "
              f"incomplete but not wrong)")
    return src, [(r["t"], r["scorer"]) for r in rows]


def build(T, D, lead):
    """Feature matrix over halts covered at every lookback, plus feature names."""
    feats = {b: [pooled(D[i], t, b, lead) for i, (_, t, _) in enumerate(T)] for b in BACKS}
    ok = np.ones(len(T), dtype=bool)
    for b in BACKS:
        ok &= np.array([f is not None for f in feats[b]])
    X, names = [], []
    for b in BACKS:
        for ci, cname in enumerate(CLASS_NAMES):
            for pi, pname in enumerate(POOLS):
                X.append(np.array([f["A"][pi][ci] - f["B"][pi][ci]
                                   for f, k in zip(feats[b], ok) if k]))
                names.append(f"{cname} {pname} (A-B) @{b:.0f}s")
    adv = [i for i, nm in enumerate(names) if nm.startswith("advance max")]
    X.append(np.mean([X[i] for i in adv], axis=0))
    names.append("advance max (A-B) mean-lookback")
    onset = [onset_centroid(D[j], t, max(BACKS), lead)
             for j, (_, t, _) in enumerate(T)]
    ok2 = ok & np.array([o is not None for o in onset])
    if ok2.sum() == ok.sum():
        X.append(np.array([o for o, k in zip(onset, ok) if k]))
        names.append(ONSET_PREREG)
    return np.stack(X), names, ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default="7")
    ap.add_argument("--perm", type=int, default=20000)
    ap.add_argument("--lead", type=float, default=0.3)
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--lights", default="all", choices=("all", "both", "single"))
    ap.add_argument("--prereg", action="store_true",
                    help="test only the pre-registered feature, one-sided")
    a = ap.parse_args()

    T, D = [], []
    for stem in a.bouts.split(","):
        src, touches = load(stem)
        d = np.load(LAB / CACHE[stem])
        print(f"{src.name}: {len(touches)} halts from {CACHE[stem]}")
        for t, sc in touches:
            T.append((stem, t, sc))
            D.append(d)

    LIGHTS = np.concatenate([lights_for(b, [u for bb, u, _ in T if bb == b])
                             for b in dict.fromkeys(b for b, _, _ in T)])
    labels = np.array([sc for _, _, sc in T])
    print(f"  left {(labels == 'left').sum()}  right {(labels == 'right').sum()}  "
          f"none {(labels == 'none').sum()}")

    X, names, ok = build(T, D, a.lead)
    dec = ok & (labels != "none")
    Xd = X[:, (labels[ok] != "none")]
    y = labels[dec] == "left"

    if a.lights != "all":
        want = {"both": {"both"}, "single": {"left", "right"}}[a.lights]
        keep_l = np.array([l in want for l in LIGHTS])
        dec = dec & keep_l
        Xd = X[:, (labels[ok] != "none") & keep_l[ok]]
        y = labels[dec] == "left"
        print(f"  restricted to {a.lights}-lamp halts: {dec.sum()} decided")

    if a.prereg:
        i = names.index(PREREG)
        v = auc(Xd[i], y)
        _, pf, _ = maxstat_p(Xd[i:i + 1], y, a.perm)
        one_sided = pf[0] / 2 if v > 0.5 else 1.0 - pf[0] / 2
        if a.lights != PREREG_LIGHTS:
            print(f"  NOTE: registered subset is --lights {PREREG_LIGHTS}, "
                  f"running --lights {a.lights}")
        print()
        print(f"=== PRE-REGISTERED: {PREREG} on {a.lights}-lamp halts, "
              f"one-sided, {dec.sum()} decided touches ===")
        print(f"  AUC {v:.2f}   one-sided p {one_sided:.4f}")
        overlap = sorted(set(SELECTED_ON) & set(a.bouts.split(",")))
        if overlap:
            print(f"  CIRCULAR -- bout(s) {','.join(overlap)} are where this "
                  f"feature was chosen. This is not a confirmation at any p "
                  f"value. Run on a bout held out from the search.")
            return 1
        print("  VERDICT: " + ("confirmed" if one_sided < 0.05 else "not confirmed"))
        return 0

    excl = ("mean-lookback", "onset lead")
    keep = [k for k, nm in enumerate(names) if not any(e in nm for e in excl)]
    Xd, X, names = Xd[keep], X[keep], [names[k] for k in keep]
    print(f"\n=== LEFT vs RIGHT: {len(names)} features, {dec.sum()} decided touches "
          f"({y.sum()} left / {(~y).sum()} right) ===")
    obs, pf, fam = maxstat_p(Xd, y, a.perm)
    order = np.argsort(obs)[::-1]
    print(f"  {'feature':<30}{'AUC':>7}{'raw p':>9}")
    for i in order[:a.show]:
        v = auc(Xd[i], y)
        print(f"  {names[i]:<30}{v:>7.2f}{pf[i]:>9.4f}")
    print(f"\n  family-wise p over all {len(names)} features: {fam:.4f}")
    print("  VERDICT: " + ("a real effect survives the search"
                           if fam < 0.05 else
                           "the best feature is what a search this wide finds in noise"))

    print(f"\n=== DECIDED vs NONE: {ok.sum()} halts "
          f"({(labels[ok] != 'none').sum()} decided / {(labels[ok] == 'none').sum()} none) ===")
    obs2, pf2, fam2 = maxstat_p(np.abs(X), labels[ok] != "none", a.perm)
    order2 = np.argsort(obs2)[::-1]
    print(f"  {'feature':<30}{'AUC':>7}{'raw p':>9}")
    for i in order2[:a.show]:
        print(f"  {'|' + names[i] + '|':<30}"
              f"{auc(np.abs(X[i]), labels[ok] != 'none'):>7.2f}{pf2[i]:>9.4f}")
    print(f"\n  family-wise p: {fam2:.4f}")
    return 0


def _self_test():
    assert abs(auc([1, 2, 3, 4], [0, 0, 1, 1]) - 1.0) < 1e-9
    assert abs(auc([4, 3, 2, 1], [0, 0, 1, 1]) - 0.0) < 1e-9
    assert abs(auc([1, 1, 1, 1], [0, 0, 1, 1]) - 0.5) < 1e-9

    # a planted signal must beat the family-wise threshold; pure noise must not
    rng = np.random.default_rng(0)
    y = np.array([0] * 20 + [1] * 20, dtype=bool)
    sig = np.stack([rng.normal(size=40) + 2.0 * y] + [rng.normal(size=40) for _ in range(40)])
    _, _, fam = maxstat_p(sig, y, 2000)
    assert fam < 0.01, fam
    noise = rng.normal(size=(48, 40))
    _, _, famn = maxstat_p(noise, y, 2000)
    assert famn > 0.05, famn

    # the family-wise value must be no smaller than the best raw p
    _, pf, fam2 = maxstat_p(noise, y, 2000)
    assert fam2 >= pf.min() - 1e-12, (fam2, pf.min())
    print("exp_touch_probe self-test: ok")


if __name__ == "__main__":
    _self_test()
    raise SystemExit(main())
