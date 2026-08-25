"""Per-bout report: tempo and outcomes from the labels, priority from the lamps, actions from the model."""
import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES
import check_touches as CT
import read_scoreboard as RS

LAB = PROJECT / "data" / "labels"
SLOT = {"left": "A", "right": "B"}


def phrase_stats(rows):
    """Tempo and outcome mix. Labels only -- no model, no accuracy caveat."""
    t = np.array([r["t"] for r in rows], dtype=float)
    scoring = [r["scorer"] for r in rows if r["scorer"] in ("left", "right")]
    gaps = np.diff(t) if len(t) > 1 else np.array([])
    best = cur = 1 if scoring else 0
    for a, b in zip(scoring, scoring[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    span = (t.max() - t.min()) / 60.0 if len(t) > 1 else 0.0
    return dict(
        halts=len(rows), scoring=len(scoring),
        off_target=sum(1 for r in rows if r["note"] == "off_target"),
        left=scoring.count("left"), right=scoring.count("right"),
        gaps=gaps, longest_run=best,
        per_min=len(scoring) / span if span else float("nan"), span_min=span)


def priority_stats(stem, rows):
    """Contested-halt rate from the lamps. Returns None where no lamp box exists."""
    lay = RS.LAYOUT.get(stem, {})
    cache = LAB / f"{stem}_lamp.npz"
    if "lamp" not in lay or not cache.exists():
        return None
    t, ser = RS.lamp_series("", lay["lamp"], 0.1, cache)
    thr = RS.lamp_all_thresholds(ser)
    kinds, won = Counter(), Counter()
    for r in rows:
        st = RS.lamp_states(t, ser, r["t"], thr)
        k = RS.lamp_kind(st)
        kinds[k] += 1
        p = RS.priority_from(st, r["scorer"])
        if p:
            won[p] += 1
    return dict(kinds=kinds, won=won)


def action_context(stem, rows, back=2.0, lead=0.3):
    """What each fencer was doing before a scoring touch. MODEL OUTPUT -- ~70% accurate."""
    f = LAB / f"{stem}_probs_mirror.npz"
    if not f.exists():
        return None
    d = np.load(f)
    slot, time, probs = d["slot"].astype(str), d["time"], d["probs"]
    scorer_act, other_act = Counter(), Counter()
    for r in rows:
        if r["scorer"] not in SLOT:
            continue
        me, them = SLOT[r["scorer"]], "B" if SLOT[r["scorer"]] == "A" else "A"
        for who, tally in ((me, scorer_act), (them, other_act)):
            m = (slot == who) & (time >= r["t"] - back) & (time <= r["t"] + lead)
            if m.any():
                tally[CLASS_NAMES[int(probs[m].mean(axis=0).argmax())]] += 1
    return dict(scorer=scorer_act, other=other_act)


def show(stem):
    src = LAB / f"bout{stem}_touches.tsv"
    if not src.exists():
        print(f"bout {stem}: no touch table, skipped")
        return None
    _, rows = CT.check(src)
    p = phrase_stats(rows)
    print()
    print(f"=== BOUT {stem} -- {p['span_min']:.1f} min ===")
    print("  tempo and outcomes (from the labels, no model)")
    print(f"    halts {p['halts']}   scoring {p['scoring']}   "
          f"off-target {p['off_target']} ({p['off_target'] / p['halts']:.0%} of halts)")
    print(f"    score {p['left']}-{p['right']}   touches/min {p['per_min']:.1f}   "
          f"longest run {p['longest_run']}")
    if len(p["gaps"]):
        q = np.percentile(p["gaps"], [25, 50, 75])
        print(f"    phrase length  p25 {q[0]:.1f}s   median {q[1]:.1f}s   p75 {q[2]:.1f}s")

    pr = priority_stats(stem, rows)
    if pr:
        two, mixed = pr["kinds"]["two_colour"], pr["kinds"]["mixed"]
        contested = two + mixed
        print("  priority (from the lamps, no model)")
        white = "left_white" in RS.LAYOUT[stem]["lamp"]
        print(f"    contested halts {contested} ({contested / p['halts']:.0%})   "
              f"two-colour {two}   mixed {mixed}"
              + ("" if white else "   [no white box: mixed halts undercounted,"
                                  " so this rate is NOT comparable across bouts]"))
        if sum(pr["won"].values()):
            print(f"    priority won  left {pr['won']['left']}   right {pr['won']['right']}")

    ac = action_context(stem, rows)
    if ac and sum(ac["scorer"].values()):
        print("  action before the touch  [MODEL OUTPUT, ~70% accurate -- treat as indicative]")
        for tag, c in (("scorer", ac["scorer"]), ("opponent", ac["other"])):
            top = ", ".join(f"{k} {v}" for k, v in c.most_common(3))
            print(f"    {tag:<9} {top}")
    return p


def _self_test():
    rows = [dict(t=10.0, scorer="left", note=""), dict(t=30.0, scorer="left", note=""),
            dict(t=45.0, scorer="none", note="off_target"),
            dict(t=70.0, scorer="right", note="")]
    p = phrase_stats(rows)
    assert p["halts"] == 4 and p["scoring"] == 3, p
    assert p["off_target"] == 1 and p["left"] == 2 and p["right"] == 1, p
    assert p["longest_run"] == 2, p                      # left, left, then right
    assert abs(np.median(p["gaps"]) - 20.0) < 1e-9, p["gaps"]
    one = phrase_stats([dict(t=5.0, scorer="left", note="")])
    assert one["halts"] == 1 and len(one["gaps"]) == 0, one
    print("bout_stats self-test: ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default="4,5,6,7")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    _self_test()
    if a.self_test:
        return 0
    every = [show(s) for s in a.bouts.split(",")]
    got = [p for p in every if p]
    if len(got) > 1:
        g = np.concatenate([p["gaps"] for p in got if len(p["gaps"])])
        h, s = sum(p["halts"] for p in got), sum(p["scoring"] for p in got)
        print(f"\n=== ALL {len(got)} BOUTS ===")
        print(f"  {h} halts, {s} scoring ({s / h:.0%}), median phrase {np.median(g):.1f}s, "
              f"{(g < 15).mean():.0%} under 15s")
        S, O = Counter(), Counter()
        for s in a.bouts.split(","):
            src = LAB / f"bout{s}_touches.tsv"
            if not src.exists():
                continue
            ac = action_context(s, CT.check(src)[1])
            if ac:
                S += ac["scorer"]
                O += ac["other"]
        n = sum(S.values())
        if n:
            print(f"  in the 2s before a touch, over {n} touches [MODEL, indicative]:")
            print(f"    scorer   advance {S['advance'] / n:.0%}  "
                  f"retreat {S['retreat'] / n:.0%}")
            print(f"    opponent advance {O['advance'] / n:.0%}  "
                  f"retreat {O['retreat'] / n:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
