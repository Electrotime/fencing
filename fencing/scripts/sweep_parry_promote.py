"""Can opponent context RAISE parry recall, not just cut false parries?

The shipped gate is one-directional. `_apply_parry_gate` only ever deletes: if the
model called parry and the opponent is not lunging, demote to the footwork runner-up.
It lifted precision 29% -> 55% on held-out bout 4 and cost recall 19% -> 17%, because
deleting is all it can do. Parry recall is therefore capped by how often parry wins
the argmax outright, no matter how obvious the context.

The co-occurrence Aaron pointed out ("lunge and parry are usually together") is
symmetric, and only half of it is being used. 76% of labelled parries have the
opponent lunging. So the untried direction: when parry LOST the argmax but sits at a
decent probability AND the opponent is unmistakably attacking, promote it.

    veto      argmax == parry  and  opp lunge <  T   ->  demote   (shipped)
    PROMOTE   argmax != parry  and  own parry >= P
                                and  opp lunge >= Q  ->  promote  (this script)

THE CONTROL THAT DECIDES THIS. Promoting on `own parry >= P` alone would also raise
recall -- it is just a lower decision threshold, and any lower threshold buys recall
with precision. So every promoter is scored against an OWN-ONLY control matched on
the NUMBER OF PROMOTIONS: take the same count of windows, ranked by own parry
probability, ignoring the opponent entirely. Same recall budget spent, same class,
opponent context the only difference. If the opponent-conditioned promoter does not
beat that, the opponent adds nothing and this closes as a negative -- exactly how the
shuffled-feature control closed blade energy.

PRE-REGISTERED, so it cannot be revised after seeing the table:
  * tune on bout 5, confirm on bout 4. Two venues, two different held-out models.
  * criterion = parry F1 on the tuning bout.
  * VETO: overall accuracy must not fall on the tuning bout. Parry is ~2% of windows;
    a rule that trades a point of overall accuracy for parry recall is a bad trade and
    the project has already rejected bigger gains than that on noise grounds.
  * the winner must beat its matched own-only control, or it closes as a negative.

Held-out caches only -- a cache from a model trained on the bout it is scored on is
circular. 4_probs_gate.npz is action_opp5 on bout 4 (in training) and is NOT usable
here; it exists only for the gate's label-invariance check.

usage: py -3 scripts/sweep_parry_promote.py [--tune 5] [--confirm 4] [--self-test]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.action_model import CLASS_NAMES
from src.labels import load_intervals, UNSCORABLE

LAB = PROJECT / "data" / "labels"
LUNGE_I, PARRY_I = CLASS_NAMES.index("lunge"), CLASS_NAMES.index("parry")

# (probability cache, label file, what the model was trained on) -- the third column is
# the audit trail that says the cache is not circular.
HELDOUT = {
    "4": ("4_probs_heldb5.npz", "bout4_intervals_2track.csv", "verify_h4_b5 (bouts 1,2,3,5)"),
    "5": ("5_probs_held.npz",   "bout5_intervals_2track.csv", "action_opp (bouts 1-4)"),
    "1": ("1_probs_heldb5.npz", "bout1_intervals.csv",        "verify_h1_b5 (bouts 2,3,4,5)"),
}

# shipped gate, mirrored from demo_video so this scores the real starting point
PARRY_OPP_LUNGE_MIN = 0.20


def load(stem):
    """Scorable windows for one bout: probs, opponent probs, truth label.

    Slots are paired by exact time, which is safe because both are predicted on the
    same frame index at the same fps -- the demo's gate relies on the same thing.
    A window whose opponent has no distribution gets opp lunge 0.0, matching
    _apply_parry_gate's conservative reading of "nobody visible to parry".
    """
    cache, csv_name, provenance = HELDOUT[stem]
    d = np.load(LAB / cache)
    slot, time, probs = d["slot"].astype(str), d["time"], d["probs"]

    index = {(s, float(t)): i for i, (s, t) in enumerate(zip(slot, time))}
    opp_lunge = np.zeros(len(time), dtype=np.float32)
    for i, (s, t) in enumerate(zip(slot, time)):
        j = index.get(("B" if s == "A" else "A", float(t)))
        if j is not None:
            opp_lunge[i] = probs[j, LUNGE_I]

    truth, _ = load_intervals(LAB / csv_name)

    def truth_at(s, t):
        for st, en, lab in truth.get(s, []):
            if st <= t < en:
                return lab
        return None

    keep, y = [], []
    for i, (s, t) in enumerate(zip(slot, time)):
        lab = truth_at(s, float(t))
        if lab is None or lab in UNSCORABLE:
            continue
        keep.append(i)
        y.append(lab)
    keep = np.array(keep, dtype=int)
    return dict(probs=probs[keep], opp_lunge=opp_lunge[keep],
                y=np.array(y), provenance=provenance, n_all=len(time))


def decide(probs, opp_lunge, promote_min=None, opp_min=None, own_only=False):
    """Predicted label per window: argmax -> shipped veto -> optional promoter.

    own_only ignores opp_lunge in the PROMOTER (the veto is untouched) -- that is the
    matched control, not a shipping candidate.
    """
    pred_i = probs.argmax(axis=1)

    # --- shipped veto: an argmax parry needs an attacking opponent ---
    alt = probs.copy()
    alt[:, PARRY_I] = -1.0
    runner_up = alt.argmax(axis=1)
    demote = (pred_i == PARRY_I) & (opp_lunge < PARRY_OPP_LUNGE_MIN)
    pred_i = np.where(demote, runner_up, pred_i)

    # --- promoter: parry lost the argmax but the context says otherwise ---
    # Eligibility is `argmax != parry`, NOT `pred != parry`. A window the veto just
    # demoted must stay demoted: the veto's whole job is deleting parries whose
    # opponent is not attacking, and re-promoting them would hand back the precision
    # it bought. The opponent-conditioned rule cannot rescue one anyway (the veto
    # fires below 0.20 and the promoter demands >= 0.30), but the own-only CONTROL
    # would -- and then the control would be spending its budget on a different
    # population, which is not a matched comparison. Same eligible set for both.
    if promote_min is not None:
        cond = (probs.argmax(axis=1) != PARRY_I) & (probs[:, PARRY_I] >= promote_min)
        if not own_only:
            cond &= opp_lunge >= opp_min
        pred_i = np.where(cond, PARRY_I, pred_i)

    return np.array(CLASS_NAMES)[pred_i]


def score(pred, y):
    tp = int(((pred == "parry") & (y == "parry")).sum())
    n_pred = int((pred == "parry").sum())
    n_true = int((y == "parry").sum())
    prec = tp / n_pred if n_pred else float("nan")
    rec = tp / n_true if n_true else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else 0.0
    return dict(acc=float((pred == y).mean()), prec=prec, rec=rec, f1=f1,
                n_pred=n_pred, n_true=n_true, tp=tp)


def control_for(bout, n_promotions):
    """Own-only promoter matched on the NUMBER of promotions, not the threshold.

    Ranks the non-parry windows by own parry probability and takes the same count, so
    the control spends an identical recall budget. Returns None if there is nothing to
    match (no promotions to compare against).
    """
    if n_promotions <= 0:
        return None
    base = decide(bout["probs"], bout["opp_lunge"])
    # same eligible set as the real promoter -- see the note in decide()
    elig = np.flatnonzero(bout["probs"].argmax(axis=1) != PARRY_I)
    if len(elig) == 0:
        return None
    p = bout["probs"][elig, PARRY_I]
    take = elig[np.argsort(p)[::-1][:n_promotions]]
    pred = base.copy()
    pred[take] = "parry"
    return score(pred, bout["y"])


def _self_test():
    """The decision path must reproduce the shipped gate exactly when off."""
    n = len(CLASS_NAMES)

    def mk(**kw):
        v = np.full(n, 0.01, dtype=np.float32)
        for k, x in kw.items():
            v[CLASS_NAMES.index(k)] = x
        return v

    # promoter OFF must equal the shipped veto
    P = np.stack([mk(parry=0.6, retreat=0.3), mk(parry=0.6, retreat=0.3),
                  mk(advance=0.5, parry=0.3), mk(advance=0.5, parry=0.3)])
    opp = np.array([0.8, 0.0, 0.8, 0.0], dtype=np.float32)
    base = decide(P, opp)
    assert list(base) == ["parry", "retreat", "advance", "advance"], list(base)

    # promoter ON: row 2 has parry 0.3 and an attacking opponent -> promoted.
    # row 3 has the same parry probability but a quiet opponent -> untouched.
    got = decide(P, opp, promote_min=0.25, opp_min=0.5)
    assert list(got) == ["parry", "retreat", "parry", "advance"], list(got)

    # the promoter must never RESCUE a parry the veto just deleted, or the two rules
    # would cancel and the gate's 55% precision would silently revert
    assert got[1] == "retreat", "veto must win over the promoter"

    # ...and that holds for the own-only control too, which is the case that would
    # otherwise quietly undo the veto and make the comparison unmatched. Row 1 has
    # parry 0.6 (well over promote_min) and was vetoed: it must STAY retreat.
    own = decide(P, opp, promote_min=0.25, opp_min=0.5, own_only=True)
    assert list(own) == ["parry", "retreat", "parry", "parry"], list(own)

    s = score(np.array(["parry", "advance"]), np.array(["parry", "parry"]))
    assert s["tp"] == 1 and s["prec"] == 1.0 and s["rec"] == 0.5, s
    print("sweep_parry_promote self-test: ok")


def report(tag, s, base):
    d_acc = 100 * (s["acc"] - base["acc"])
    print(f"  {tag:<34}{s['acc']:>8.1%}{d_acc:>+8.2f}{s['prec']:>11.0%}"
          f"{s['rec']:>9.0%}{s['f1']:>7.2f}{s['n_pred']:>8d}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", default="5")
    ap.add_argument("--confirm", default="4")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        _self_test()
        return 0
    _self_test()

    bouts = {k: load(k) for k in (a.tune, a.confirm)}
    for k, b in bouts.items():
        n_par = int((b["y"] == "parry").sum())
        print(f"bout {k}: {len(b['y'])} scorable of {b['n_all']} windows, "
              f"{n_par} parry ({n_par / len(b['y']):.1%})   [{b['provenance']}]")

    grid_p = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
    grid_q = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

    tune = bouts[a.tune]
    base = score(decide(tune["probs"], tune["opp_lunge"]), tune["y"])
    print(f"\n=== TUNING on held-out bout {a.tune} "
          f"(criterion: parry F1, veto: overall accuracy must not fall) ===")
    print(f"  {'rule':<34}{'overall':>8}{'d pts':>8}{'parry P':>11}{'R':>9}{'F1':>7}{'#pred':>8}")
    report("shipped gate (veto only)", base, base)

    best = None
    for p in grid_p:
        for q in grid_q:
            pred = decide(tune["probs"], tune["opp_lunge"], p, q)
            s = score(pred, tune["y"])
            n_prom = int((pred == "parry").sum() - base["n_pred"])
            report(f"promote p>={p:.2f}, opp>={q:.2f}", s, base)
            if s["acc"] < base["acc"]:          # pre-registered veto
                continue
            if best is None or s["f1"] > best[0]["f1"]:
                best = (s, p, q, n_prom)

    if best is None:
        print(f"\nNo promoter clears the pre-registered veto on bout {a.tune}: every "
              f"setting that\nraises parry recall costs overall accuracy. CLOSED as a "
              f"negative.")
        return 0

    s, p, q, n_prom = best
    print(f"\n  best by F1: p>={p:.2f}, opp>={q:.2f}  "
          f"({n_prom} windows promoted, F1 {base['f1']:.2f} -> {s['f1']:.2f})")

    ctrl = control_for(tune, n_prom)
    print(f"\n=== THE CONTROL: same {n_prom} promotions, ranked by own parry "
          f"probability alone ===")
    print(f"  {'rule':<34}{'overall':>8}{'d pts':>8}{'parry P':>11}{'R':>9}{'F1':>7}{'#pred':>8}")
    report(f"opponent-conditioned", s, base)
    if ctrl is not None:
        report("own-only control (matched n)", ctrl, base)
        if s["f1"] <= ctrl["f1"]:
            print("\n  The control MATCHES OR BEATS it. The gain is a lower parry "
                  "threshold, not\n  opponent context. CLOSED as a negative -- same "
                  "verdict the shuffled feature\n  control gave blade energy.")
            return 0
        print(f"\n  Opponent context beats the matched control "
              f"({s['f1']:.2f} vs {ctrl['f1']:.2f}). Proceed to confirm.")

    conf = bouts[a.confirm]
    cbase = score(decide(conf["probs"], conf["opp_lunge"]), conf["y"])
    cnew = score(decide(conf["probs"], conf["opp_lunge"], p, q), conf["y"])
    cprom = cnew["n_pred"] - cbase["n_pred"]
    cctrl = control_for(conf, cprom)
    print(f"\n=== CONFIRMING on held-out bout {a.confirm} (different venue, "
          f"different model, threshold NOT tuned here) ===")
    print(f"  {'rule':<34}{'overall':>8}{'d pts':>8}{'parry P':>11}{'R':>9}{'F1':>7}{'#pred':>8}")
    report("shipped gate (veto only)", cbase, cbase)
    report(f"promote p>={p:.2f}, opp>={q:.2f}", cnew, cbase)
    if cctrl is not None:
        report("own-only control (matched n)", cctrl, cbase)

    print("\nA rule that only works on the bout it was tuned on is a tuned constant, "
          "not a\nmechanism. The parry veto shipped because 0.2 was best-or-tied on "
          "BOTH bouts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
