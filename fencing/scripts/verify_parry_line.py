"""Replay the shipped gate against the offline rule over every cached window."""
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import demo_video as D
import exp_parry_decoder as DEC
import sweep_parry_promote as S
from src.action_model import CLASS_NAMES

PARRY_I = CLASS_NAMES.index("parry")
LUNGE_I = CLASS_NAMES.index("lunge")


def offline(own, opp_lunge):
    """The rule as measured offline: one linear boundary, parry or runner-up."""
    score = D.PARRY_W_OWN * own[:, PARRY_I] + D.PARRY_W_OPP * opp_lunge
    alt = own.copy()
    alt[:, PARRY_I] = -1.0
    return np.where(score >= D.PARRY_LINE_MIN, PARRY_I, alt.argmax(axis=1))


def live(own, opp_probs):
    """The rule as the demo runs it, through the real FencerTrack objects."""
    out = np.empty(len(own), dtype=int)
    for i in range(len(own)):
        a, b = D.FencerTrack(), D.FencerTrack()
        a.probs = own[i].astype(np.float32)
        a.label = CLASS_NAMES[int(a.probs.argmax())]
        a.conf = float(a.probs.max())
        b.probs = opp_probs[i].astype(np.float32)
        b.label = CLASS_NAMES[int(b.probs.argmax())]
        tr = {"A": a, "B": b}
        D._apply_parry_gate(tr)
        out[i] = CLASS_NAMES.index(tr["A"].label)
    return out


def main() -> int:
    if not D.PARRY_DECODER:
        print("PARRY_DECODER is off; nothing to verify")
        return 1
    total = bad = 0
    print(f"  {'bout':<6}{'windows':>9}{'disagreements':>15}{'parry called':>14}")
    for stem in ("4", "7", "5", "1"):
        own, opp, seen, ylab, lag, prov = DEC.load_pairs(stem)
        o = offline(own, opp[:, LUNGE_I])
        l = live(own, opp)
        d = int((o != l).sum())
        total += len(o)
        bad += d
        print(f"  {stem:<6}{len(o):>9}{d:>15}{int((l == PARRY_I).sum()):>14}")
    print(f"\n  {total} windows, {bad} disagreements")
    if bad:
        print("  MISMATCH -- the shipped gate does not implement the measured rule")
        return 1
    print("  live gate and offline rule agree exactly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
