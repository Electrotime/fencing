"""Reading Aaron's interval label files. One definition, two consumers."""
import csv
from collections import defaultdict
from pathlib import Path

from src.action_model import CLASS_NAMES

# left/right in the labels, A/B in the demo's slots. A is always the LEFT fencer:
# assignment is memoryless x-order because fencers never cross a piste.
SLOT_OF = {"left": "A", "right": "B"}

UNSCORABLE = {"extension"}


def collapse_two_track(footwork: str, blade: str) -> str:
    """Collapse the two tracks to the one label a six-way model can be scored on."""
    blade = blade.strip()
    return blade if blade in CLASS_NAMES else footwork.strip()


def load_intervals(path: Path) -> tuple[dict[str, list[tuple[float, float, str]]], bool]:
    """Return ({slot: [(start, end, label)]}, two_track)."""
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
    """Merged (start, end) spans of LABELLED time for one slot, any class."""
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
    """The (start, end) holes between labelled spans, plus head and tail."""
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
