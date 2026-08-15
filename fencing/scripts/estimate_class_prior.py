"""Fix `lunge` over-prediction by correcting the class prior. Honestly split."""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT = Path(r"c:\Users\aaron\OneDrive\Documents\GitHub\fencing\fencing")
sys.path.insert(0, str(PROJECT))
from src.action_model import CLASS_NAMES

LABELS = PROJECT / "data" / "labels" / "bout1_intervals.csv"
CACHE = PROJECT / "data" / "labels" / "bout1_probs.npz"
SLOT_OF = {"left": "A", "right": "B"}
K = len(CLASS_NAMES)

truth = defaultdict(list)
with open(LABELS) as f:
    for row in csv.DictReader(r for r in f if not r.startswith("#")):
        truth[SLOT_OF[row["fencer"]]].append(
            (float(row["start"]), float(row["end"]), row["label"].strip()))


def truth_at(slot, t):
    for s, e, lab in truth[slot]:
        if s <= t < e:
            return lab
    return None


d = np.load(CACHE)
rows = []
for slot, t, p in zip(d["slot"], d["time"], d["probs"]):
    g = truth_at(str(slot), float(t))
    if g is not None:
        rows.append((float(t), CLASS_NAMES.index(g), p))
T = np.array([r[0] for r in rows])
Y = np.array([r[1] for r in rows])
P = np.stack([r[2] for r in rows])
print(f"{len(rows)} windows with ground truth\n")

# the model was trained with inverse-frequency weighting -> effective prior uniform
TRAIN_PRIOR = np.full(K, 1.0 / K)


def acc(P_, Y_):
    return float(np.mean(P_.argmax(1) == Y_))


def per_class(P_, Y_):
    pred = P_.argmax(1)
    out = []
    for c in range(K):
        n = int(np.sum(Y_ == c))
        tp = int(np.sum((pred == c) & (Y_ == c)))
        pp = int(np.sum(pred == c))
        out.append((CLASS_NAMES[c], n, pp, tp / pp if pp else float("nan"),
                    tp / n if n else float("nan")))
    return out


def apply_prior(P_, target):
    q = P_ * (target / TRAIN_PRIOR)[None, :]
    return q / q.sum(1, keepdims=True)


def em_prior(P_, iters=200):
    """Saerens et al.: estimate target prior from unlabelled predictions alone."""
    pri = TRAIN_PRIOR.copy()
    for _ in range(iters):
        q = P_ * (pri / TRAIN_PRIOR)[None, :]
        q /= q.sum(1, keepdims=True)
        new = q.mean(0)
        if np.max(np.abs(new - pri)) < 1e-9:
            break
        pri = new
    return pri


mid = np.median(T)
halves = {"first": T < mid, "second": T >= mid}

print(f"baseline (no correction): overall {acc(P, Y):.1%}")
print(f"   {'class':<9}{'n_true':>8}{'n_pred':>8}{'prec':>7}{'recall':>8}")
for name, n, pp, pr, rc in per_class(P, Y):
    print(f"   {name:<9}{n:>8}{pp:>8}{pr:>7.0%}{rc:>8.0%}")

print("\n=== prior fitted on one half, scored on the OTHER (no cheating) ===")
for fit, test in (("first", "second"), ("second", "first")):
    fm, tm = halves[fit], halves[test]
    target = np.bincount(Y[fm], minlength=K).astype(float)
    target /= target.sum()
    before, after = acc(P[tm], Y[tm]), acc(apply_prior(P[tm], target), Y[tm])
    print(f"  fit {fit:<6} -> test {test:<6}: {before:.1%} -> {after:.1%}  "
          f"({after - before:+.1%})")

print("\n=== EM prior (needs NO labels — usable in production) ===")
pri = em_prior(P)
print("   estimated: " + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, pri)))
true_pri = np.bincount(Y, minlength=K) / len(Y)
print("   actual   : " + "  ".join(f"{c}={v:.3f}" for c, v in zip(CLASS_NAMES, true_pri)))
Pem = apply_prior(P, pri)
print(f"   overall {acc(P, Y):.1%} -> {acc(Pem, Y):.1%}")
print(f"   {'class':<9}{'n_true':>8}{'n_pred':>8}{'prec':>7}{'recall':>8}")
for name, n, pp, pr, rc in per_class(Pem, Y):
    print(f"   {name:<9}{n:>8}{pp:>8}{pr:>7.0%}{rc:>8.0%}")
