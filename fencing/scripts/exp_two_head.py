"""Two heads -- footwork and blade -- instead of one six-way choice.

This is what the two-track schema was FOR, and it has never been built. The
argument: `parry` has to out-compete five footwork classes to be emitted at all,
while physically it co-occurs with them (63 labelled parries: 34 retreat, 7
neutral, 5 advance underneath). A separate blade head only has to beat `none` and
`extension`, so a parry no longer has to be a better explanation than the retreat
it is happening during.

LABEL AVAILABILITY IS THE HARD PART, and fabricating the missing half would
quietly poison this:

  bouts 3, 4   two-track -> BOTH tracks known
  bouts 1, 2   single-track -> a `parry` row gives blade=parry and says NOTHING
               about the footwork; any other row gives footwork, and blade is
               UNKNOWN, not `none` -- the old schema forced one label, so a parry
               during a retreat had to be written `retreat`
  clips        a parry clip gives blade=parry, footwork unknown; other clips give
               footwork, blade unknown (a clip cut as `advance` may well contain
               an extension)

So both heads take a masked loss: -1 means "not labelled", contributes nothing.
Negative `none` evidence comes from bouts 3-4, which record it explicitly (bout 4
alone has ~185 blade=none rows), so the blade head is not starved of negatives.

Scored three ways, because they answer different questions:
  footwork  5-way accuracy (parry removed from the footwork vocabulary)
  blade     parry precision/recall -- the point of the exercise
  collapsed blade-priority back to one label, using the SAME rule evaluate_labels
            uses, so it is directly comparable to the six-way model

usage: py -3 scripts/exp_two_head.py --holdout 3 [--seeds 2]
"""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import (CLASS_NAMES, DROPOUT, HIDDEN_SIZE, INPUT_SIZE,
                              N_AGG_FEATURES, N_AGG_WIDE, WEIGHT_DECAY, _pick_device)
from exp_opponent import with_opponent
from train_continuous import clip_dataset_arrays

CONT = PROJECT / "data" / "train_continuous"
LABELS = PROJECT / "data" / "labels"
FOOT = ["advance", "lunge", "retreat", "neutral", "walking"]   # no parry
BLADE = ["none", "parry", "extension"]
F_IX = {n: i for i, n in enumerate(FOOT)}
B_IX = {n: i for i, n in enumerate(BLADE)}
CSV_FOR = {"1": "bout1_intervals.csv", "2": "bout2_intervals.csv",
           "3": "bout3_intervals_2track.csv", "4": "bout4_intervals_2track.csv",
           # bout 5 is two-track, so BOTH tracks are known for all 144 intervals.
           # It is the test of this experiment's own prediction: holding out bout 4
           # previously left just 189 blade labels in training (parry lamp at
           # chance), and bout 5 feeds exactly that starved arm.
           "5": "bout5_intervals_2track.csv"}


def bout_labels(stem):
    """slot -> [(start, end, footwork_idx, blade_idx)], -1 where unknown."""
    path = LABELS / CSV_FOR[stem]
    out = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(r for r in f if not r.startswith("#"))
        two = "footwork" in (rdr.fieldnames or [])
        for r in rdr:
            slot = {"left": "A", "right": "B"}[r["fencer"]]
            if two:
                fi = F_IX.get(r["footwork"].strip(), -1)
                bi = B_IX.get(r["blade"].strip(), -1)
            else:
                lab = r["label"].strip()
                if lab == "parry":
                    fi, bi = -1, B_IX["parry"]        # footwork was never recorded
                elif lab == "extension":
                    fi, bi = -1, B_IX["extension"]
                else:
                    fi, bi = F_IX.get(lab, -1), -1    # NOT `none`: a parry may be hidden
            out[slot].append((float(r["start"]), float(r["end"]), fi, bi))
    return out


def load_bout(stem, stride=1, opponent=False):
    d = np.load(CONT / f"{stem}.npz")
    lab = bout_labels(stem)
    fw = np.full(len(d["y"]), -1, dtype=np.int64)
    bl = np.full(len(d["y"]), -1, dtype=np.int64)
    for i, (s, t) in enumerate(zip(d["slot"], d["time"])):
        for a, b, fi, bi in lab[str(s)]:
            if a <= float(t) < b:
                fw[i], bl[i] = fi, bi
                break
    # The shipped single-head model reads the OPPONENT's engineered features, worth
    # +2.9 pts there. Without this the two-head arm is handicapped against the model
    # it is being compared to, and a parry is precisely the action whose stimulus is
    # on the other fencer -- 34 of 63 parries happen during a retreat because the
    # opponent is attacking.
    agg = with_opponent(CONT / f"{stem}.npz")[0]["wide"] if opponent else d["agg"]
    full = dict(X=d["X"], agg=agg, lengths=d["lengths"], y=d["y"], fw=fw, bl=bl)
    return {"eval": full, "train": {k: v[::stride] for k, v in full.items()}}


class TwoHeadData(Dataset):
    def __init__(self, X, A, L, F, B):
        self.X, self.A, self.L, self.F, self.B = X, A, L, F, B

    def __len__(self):
        return len(self.F)

    def __getitem__(self, i):
        return (torch.from_numpy(self.X[i]), torch.from_numpy(self.A[i]),
                int(self.L[i]), int(self.F[i]), int(self.B[i]))


class TwoHead(nn.Module):
    """Shared trunk, two heads. Trunk identical to ActionLSTM so the comparison
    isolates the head structure rather than capacity."""

    def __init__(self, pool="last", n_agg=N_AGG_FEATURES):
        super().__init__()
        self.pool_mode = pool
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        mk = lambda n: nn.Sequential(
            nn.Linear(HIDDEN_SIZE + n_agg, 64), nn.ReLU(),
            nn.Dropout(DROPOUT), nn.Linear(64, n))
        self.foot, self.blade = mk(len(FOOT)), mk(len(BLADE))

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
        z = torch.cat([pooled, agg], dim=-1)
        return self.foot(z), self.blade(z)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="3")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--pool", default="last", choices=("mean", "max", "last"))
    # Sweep the blade head's decision threshold. As TWO INDICATORS a false parry
    # only lights a lamp wrongly -- it no longer overwrites a correct footwork
    # call -- so the usable operating point is a precision/recall trade, not the
    # argmax. Aaron's framing; the collapsed metric could not see this.
    ap.add_argument("--thresholds", default="0.25,0.4,0.5,0.6,0.75,0.9")
    # Keep this fraction of BLADE labels, masking the rest to -1. Footwork labels
    # are untouched, so this isolates "how much blade supervision" from every other
    # variable -- the question being whether labelling more parries would actually
    # make the lamp usable, or whether it plateaus at chance.
    ap.add_argument("--blade-frac", type=float, default=1.0)
    # Give both heads the opponent's engineered features, as the shipped single-head
    # model does. IMPLIES NO CLIPS: a clip is a single-fencer file, so its opponent
    # block is all zeros and perfectly correlated with "came from a clip" -- measured
    # harmful (-3.4 on bout 1) for the single head, and there is no reason it would
    # behave differently here.
    ap.add_argument("--opponent", action="store_true")
    a = ap.parse_args()

    bouts = {s: load_bout(s, a.stride, a.opponent)
             for s in CSV_FOR if (CONT / f"{s}.npz").exists()}
    tr = [k for k in bouts if k != a.holdout]
    n_agg = N_AGG_WIDE if a.opponent else N_AGG_FEATURES

    parts_X, parts_A, parts_L, parts_F, parts_B = [], [], [], [], []
    if not a.opponent:
        cX, cA, cL, cY = clip_dataset_arrays()
        # clips: parry -> blade only; everything else -> footwork only
        parts_X, parts_A, parts_L = [cX], [cA], [cL]
        parts_F = [np.array([F_IX.get(CLASS_NAMES[c], -1) for c in cY], dtype=np.int64)]
        parts_B = [np.array([B_IX["parry"] if CLASS_NAMES[c] == "parry" else -1
                             for c in cY], dtype=np.int64)]

    X = np.concatenate(parts_X + [bouts[k]["train"]["X"] for k in tr])
    A = np.concatenate(parts_A + [bouts[k]["train"]["agg"] for k in tr])
    L = np.concatenate(parts_L + [bouts[k]["train"]["lengths"] for k in tr])
    F = np.concatenate(parts_F + [bouts[k]["train"]["fw"] for k in tr])
    B = np.concatenate(parts_B + [bouts[k]["train"]["bl"] for k in tr])

    if a.blade_frac < 1.0:
        rng = np.random.default_rng(0)
        known = np.flatnonzero(B >= 0)
        drop = rng.choice(known, size=int(round(len(known) * (1 - a.blade_frac))),
                          replace=False)
        B = B.copy()
        B[drop] = -1

    device = _pick_device()
    print(f"held out bout {a.holdout}; train on "
          f"{'bouts' if a.opponent else 'clips + bouts'} {tr} = {len(F)} windows"
          f"{'  [OPPONENT, no clips]' if a.opponent else ''}")
    print(f"  footwork labelled on {int((F >= 0).sum())}, blade labelled on "
          f"{int((B >= 0).sum())}")
    print("  blade mix: " + "  ".join(
        f"{n}={int((B == i).sum())}" for i, n in enumerate(BLADE)) + f"  unknown={int((B < 0).sum())}")
    ev = bouts[a.holdout]["eval"]
    print(f"  held-out has {int((ev['bl'] == B_IX['parry']).sum())} parry windows "
          f"of {len(ev['y'])}\n", flush=True)

    thresholds = [float(x) for x in a.thresholds.split(",")]

    def _pr(truth, pred):
        """(precision, recall, fraction-of-windows-lit) for one threshold."""
        tp = int((truth & pred).sum())
        return (tp / max(int(pred.sum()), 1),
                tp / max(int(truth.sum()), 1),
                float(pred.mean()))

    ds = TwoHeadData(X, A, L, F, B)
    accs, precs, recs, foots, sweeps = [], [], [], [], []
    for s in range(a.seeds):
        torch.manual_seed(42 + s)
        m = TwoHead(a.pool, n_agg).to(device)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=WEIGHT_DECAY)
        # ignore_index=-1 is what makes partially-labelled sources usable
        lf = nn.CrossEntropyLoss(ignore_index=-1)
        dl = DataLoader(ds, batch_size=32, shuffle=True)
        m.train()
        for _ in range(a.epochs):
            for xb, ab, nb, fb, bb in dl:
                opt.zero_grad()
                fo, bo = m(xb.to(device), ab.to(device), nb.to(device))
                (lf(fo, fb.to(device)) + lf(bo, bb.to(device))).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            fo, bo = m(torch.from_numpy(ev["X"]).to(device),
                       torch.from_numpy(ev["agg"]).to(device),
                       torch.from_numpy(ev["lengths"]).to(device))
        fp = fo.argmax(1).cpu().numpy()
        bprob = torch.softmax(bo, dim=1)[:, B_IX["parry"]].cpu().numpy()
        bp = bo.argmax(1).cpu().numpy()

        mf = ev["fw"] >= 0
        foots.append(float((fp[mf] == ev["fw"][mf]).mean()))
        truth_parry = ev["bl"] == B_IX["parry"]
        sweeps.append([(t, *_pr(truth_parry, bprob >= t)) for t in thresholds])
        tp = int((truth_parry & (bp == B_IX["parry"])).sum())
        precs.append(tp / max(int((bp == B_IX["parry"]).sum()), 1))
        recs.append(tp / max(int(truth_parry.sum()), 1))
        # collapsed EXACTLY as evaluate_labels does: blade wins only if emittable
        coll = np.array([CLASS_NAMES.index("parry") if bp[i] == B_IX["parry"]
                         else CLASS_NAMES.index(FOOT[fp[i]]) for i in range(len(fp))])
        accs.append(float((coll == ev["y"]).mean()))

    print("=== AS TWO INDEPENDENT INDICATORS (Aaron's framing) ===")
    print("A wrong parry lamp costs a wrong parry lamp; it no longer overwrites a")
    print("correct footwork call, so these two lines are the ones that matter.\n")
    print(f"  {'footwork 5-way accuracy':<28}{np.mean(foots):.1%}")
    print(f"  {'parry @argmax':<28}prec {np.mean(precs):.0%}  rec {np.mean(recs):.0%}")
    print(f"\n  parry lamp, by decision threshold "
          f"({int(truth_parry.sum())} true parry windows):")
    print(f"    {'thresh':>7}{'precision':>11}{'recall':>9}{'lamp on':>10}")
    for i, t in enumerate(thresholds):
        pr = float(np.mean([s[i][1] for s in sweeps]))
        rc = float(np.mean([s[i][2] for s in sweeps]))
        on = float(np.mean([s[i][3] for s in sweeps]))
        print(f"    {t:>7.2f}{pr:>11.0%}{rc:>9.0%}{on:>10.1%}")

    print(f"\n=== collapsed to ONE label, for reference only ===")
    print(f"  {'collapsed 6-way accuracy':<28}{np.mean(accs):.1%} (+-{np.std(accs):.1%})")
    print("  This is the metric that made two-head look like a failure, and it is")
    print("  the WRONG metric for a two-indicator display: it charges every false")
    print("  parry the full cost of a destroyed footwork call.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
