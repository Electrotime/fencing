"""Reading Aaron's interval label files. One definition, two consumers.

`evaluate_labels.py` and `bout_timeline.py` both parse these CSVs and collapse the
two-track schema, and the collapse rule has a history of being got wrong in subtly
different ways. That is exactly the kind of thing that gets hand-copied and then
drifts -- `wide_agg()` in action_model.py exists for the same reason -- so it lives
here once. Verified identical to evaluate_labels.py's previous inline parser on all
seven label files before the move.

(`calibrate_gate.py` keeps its own reader on purpose: it wants the union of labelled
spans across BOTH fencers with the class discarded, which is a different question.)

Two schemas are in circulation:

  single-track   fencer,start,end,label
  TWO-TRACK      fencer,start,end,footwork,blade

The two-track file is the current format; see CLAUDE.md, "THE CLASSES ARE NOT
MUTUALLY EXCLUSIVE". A fencer parries WHILE retreating, so footwork and blade are
separate columns.
"""
import csv
from collections import defaultdict
from pathlib import Path

from src.action_model import CLASS_NAMES

# left/right in the labels, A/B in the demo's slots. A is always the LEFT fencer:
# assignment is memoryless x-order because fencers never cross a piste.
SLOT_OF = {"left": "A", "right": "B"}

# `extension` is a real blade action but not one of the six classes -- it lives on
# as the arm-reach FEATURE. The model can never emit it, so windows labelled with it
# are excluded from scoring rather than counted as misses.
UNSCORABLE = {"extension"}


def collapse_two_track(footwork: str, blade: str) -> str:
    """Collapse the two tracks to the one label a six-way model can be scored on.

    Blade takes priority -- but ONLY if it is a label the model can actually emit.
    A naive blade-priority collapse defers to `extension`, and in bout 3 ten of
    fourteen lunges are written `lunge` + arm extension, so it silently deleted
    almost every lunge from a bout labelled specifically for its lunges. Fall
    through to footwork for anything unemittable.
    """
    blade = blade.strip()
    return blade if blade in CLASS_NAMES else footwork.strip()


def load_intervals(path: Path) -> tuple[dict[str, list[tuple[float, float, str]]], bool]:
    """Return ({slot: [(start, end, label)]}, two_track).

    Rows are returned in file order per slot, not sorted -- `check_labels.py` is
    what guarantees they do not overlap, and re-sorting here would hide a file that
    failed that check.
    """
    truth: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(r for r in f if not r.startswith("#"))
        two_track = "footwork" in (rdr.fieldnames or [])
        for row in rdr:
            label = (collapse_two_track(row["footwork"], row["blade"]) if two_track
                     else row["label"].strip())
            truth[SLOT_OF[row["fencer"]]].append(
                (float(row["start"]), float(row["end"]), label))
    return dict(truth), two_track


def labelled_spans(truth: dict[str, list[tuple[float, float, str]]],
                   slot: str) -> list[tuple[float, float]]:
    """Merged (start, end) spans of LABELLED time for one slot, any class.

    Used to tell "the model called an action during fencing we labelled differently"
    apart from "the model called an action over a replay" -- a distinction the
    per-window scorer cannot make, because it drops unlabelled time entirely.
    """
    spans = sorted((s, e) for s, e, _ in truth.get(slot, []))
    merged: list[tuple[float, float]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def overlap(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Seconds of overlap between two intervals; 0.0 if they are disjoint."""
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def unlabelled_gaps(truth: dict[str, list[tuple[float, float, str]]], slot: str,
                    end: float) -> list[tuple[float, float]]:
    """The (start, end) holes between labelled spans, plus head and tail.

    Why the LENGTH of a hole matters: Aaron described bout 4's gaps two different
    ways -- "if there is a gap it probably means there's no arm/blade thing" and
    "the gaps in intervals aren't gaps in fencing, it's just that the broadcast has
    other stuff". Those imply different things, and the measured gap distribution
    says both are true of different gaps: median 5.3 s (a pause between actions
    during live fencing) but 32 gaps over 15 s covering 906 of the 1190 gap-seconds
    (replays, crowd shots, graphics).

    So an unlabelled stretch cannot be treated uniformly as "not fencing". Callers
    should bucket by hole length rather than assume.
    """
    spans = labelled_spans(truth, slot)
    if not spans:
        return [(0.0, end)]
    holes = [(0.0, spans[0][0])] if spans[0][0] > 0 else []
    holes += [(spans[i][1], spans[i + 1][0]) for i in range(len(spans) - 1)]
    if spans[-1][1] < end:
        holes.append((spans[-1][1], end))
    return [(s, e) for s, e in holes if e > s]


if __name__ == "__main__":
    # smoke test: the collapse rule is the part with a history of getting this wrong
    assert collapse_two_track("retreat", "parry") == "parry"      # blade wins
    assert collapse_two_track("lunge", "extension") == "lunge"    # unemittable -> footwork
    assert collapse_two_track("advance", "none") == "advance"
    assert collapse_two_track("neutral", "") == "neutral"
    assert overlap((0, 2), (1, 3)) == 1.0
    assert overlap((0, 1), (2, 3)) == 0.0
    m = labelled_spans({"A": [(0.0, 1.0, "x"), (0.9, 2.0, "y"), (5.0, 6.0, "z")]}, "A")
    assert m == [(0.0, 2.0), (5.0, 6.0)], m
    t = {"A": [(2.0, 3.0, "x"), (8.0, 9.0, "y")]}
    assert unlabelled_gaps(t, "A", 10.0) == [(0.0, 2.0), (3.0, 8.0), (9.0, 10.0)]
    assert unlabelled_gaps(t, "A", 9.0) == [(0.0, 2.0), (3.0, 8.0)]   # no empty tail
    assert unlabelled_gaps({}, "A", 5.0) == [(0.0, 5.0)]              # nothing labelled
    print("src/labels.py: ok")
