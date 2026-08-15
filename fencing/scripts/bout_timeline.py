"""Turn the per-window prediction stream into an EVENT TIMELINE and bout statistics."""
import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from src.action_model import CLASS_NAMES
from src.labels import (labelled_spans, load_intervals, overlap, unlabelled_gaps,
                        UNSCORABLE)

FILLER_GAP = 15.0

QUIET_CLASSES = {"neutral", "walking"}
SLOT_NAME = {"A": "left", "B": "right"}
# Aggression proxy: forward intent vs yielding ground. Not a claim about who is
# winning -- retreating is a legitimate tactic and this counts actions, not touches.
FORWARD, BACKWARD = {"advance", "lunge"}, {"retreat"}


@dataclass
class Event:
    slot: str
    label: str
    start: float
    end: float
    conf: float      # mean probability of `label` across the event's windows
    peak: float      # highest single-window probability
    n: int           # number of predictions merged

    @property
    def dur(self) -> float:
        return self.end - self.start


def mmss(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:05.2f}"


def smooth_probs(times: np.ndarray, probs: np.ndarray, window: float) -> np.ndarray:
    """Average each prediction's probabilities with its neighbours within `window` s."""
    if window <= 0:
        return probs
    out = np.empty_like(probs)
    lo = np.searchsorted(times, times - window / 2, side="left")
    hi = np.searchsorted(times, times + window / 2, side="right")
    for i in range(len(times)):
        out[i] = probs[lo[i]:hi[i]].mean(axis=0)
    return out


def build_events(times: np.ndarray, probs: np.ndarray, slot: str, *,
                 max_gap: float, period: float) -> list[Event]:
    """Merge a run of same-label predictions into one event. `times` must be sorted."""
    if len(times) == 0:
        return []
    assert (np.diff(times) >= 0).all(), "sort the stream before segmenting it"
    labels = [CLASS_NAMES[i] for i in probs.argmax(axis=1)]
    confs = probs.max(axis=1)

    events: list[Event] = []
    run_start = run_last = times[0]
    run_label = labels[0]
    run_confs = [float(confs[0])]
    for t, lab, c in zip(times[1:], labels[1:], confs[1:]):
        if lab == run_label and (t - run_last) <= max_gap:
            run_last, _ = t, run_confs.append(float(c))
            continue
        events.append(Event(slot, run_label, float(run_start), float(run_last + period),
                            float(np.mean(run_confs)), float(np.max(run_confs)),
                            len(run_confs)))
        run_start = run_last = t
        run_label, run_confs = lab, [float(c)]
    events.append(Event(slot, run_label, float(run_start), float(run_last + period),
                        float(np.mean(run_confs)), float(np.max(run_confs)),
                        len(run_confs)))
    return events


def classify_event(ev: Event, truth: dict, spans: dict,
                   gaps: dict | None = None) -> str:
    """matched / wrong-class / unscorable / filler / in-pause."""
    best, best_ov = None, 0.0
    for s, e, lab in truth.get(ev.slot, []):
        ov = overlap((ev.start, ev.end), (s, e))
        if ov > best_ov:
            best, best_ov = lab, ov
    if best is None:
        if any(overlap((ev.start, ev.end), sp) > 0 for sp in spans.get(ev.slot, [])):
            return "wrong-class"
        hole = max((e - s for s, e in gaps.get(ev.slot, [])
                    if overlap((ev.start, ev.end), (s, e)) > 0), default=float("inf")) \
            if gaps else float("inf")
        return "filler" if hole >= FILLER_GAP else "in-pause"
    if best in UNSCORABLE:
        return "unscorable"
    return "matched" if best == ev.label else "wrong-class"


def score(events: list[Event], truth: dict, end: float = 0.0) -> dict:
    """Event-level precision, plus recall over ground-truth action intervals."""
    spans = {s: labelled_spans(truth, s) for s in truth}
    gaps = {s: unlabelled_gaps(truth, s, end) for s in truth} if end else {}
    buckets = Counter(classify_event(ev, truth, spans, gaps) for ev in events)
    # `in-pause` is excluded from precision entirely rather than counted either way:
    # it is unlabelled time inside live fencing, so we genuinely do not know.
    scorable = sum(v for k, v in buckets.items()
                   if k not in ("unscorable", "in-pause"))

    found = strict_found = total = 0
    per_class: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # any, strict, n
    for slot, rows in truth.items():
        for s, e, lab in rows:
            if lab in QUIET_CLASSES or lab in UNSCORABLE:
                continue
            total += 1
            covered = sum(overlap((ev.start, ev.end), (s, e)) for ev in events
                          if ev.slot == slot and ev.label == lab)
            hit = covered > 0
            strict = covered >= 0.5 * (e - s)
            found += hit
            strict_found += strict
            per_class[lab][0] += hit
            per_class[lab][1] += strict
            per_class[lab][2] += 1
    return {"buckets": buckets, "scorable": scorable,
            "precision": buckets["matched"] / scorable if scorable else float("nan"),
            "recall": found / total if total else float("nan"),
            "strict_recall": strict_found / total if total else float("nan"),
            "found": found, "strict_found": strict_found, "total": total,
            "per_class": dict(sorted(per_class.items()))}


def gate(events: list[Event], min_dur: float, min_conf: float) -> list[Event]:
    return [e for e in events if e.dur >= min_dur and e.conf >= min_conf]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache", type=Path)
    ap.add_argument("labels", type=Path, nargs="?")
    ap.add_argument("--min-dur", type=float, default=0.60)
    ap.add_argument("--min-conf", type=float, default=0.55)
    ap.add_argument("--max-gap", type=float, default=0.40)
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="seconds of probability smoothing before segmenting (0=off)")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--csv", type=Path)
    ap.add_argument("--quiet-events", action="store_true")
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()

    assert QUIET_CLASSES == {"neutral", "walking"}, "keep in step with demo_video"
    if not a.cache.exists():
        print(f"no cache at {a.cache}. Generate one with:\n"
              f"  py -3 scripts/evaluate_labels.py <video> <labels> "
              f"--model models/verify_*.pth --tag _held")
        return 1
    if a.labels and not any(k in a.cache.stem for k in ("held", "verify")):
        print("!! This cache does not look like it came from a HELD-OUT checkpoint.\n"
              "!! models/action_opp.pth trained on every labelled bout, so scoring it\n"
              "!! against one of them is circular -- it reads 91.6% on bout 1 where the\n"
              "!! honest held-out figure is 74.6%. Regenerate with:\n"
              "!!   evaluate_labels.py <video> <labels> --model models/verify_<N>.pth "
              "--tag _held\n")

    z = np.load(a.cache)
    slots, times, probs = z["slot"], z["time"], z["probs"]
    period = float(np.median(np.diff(np.sort(times[slots == "A"])))) if (slots == "A").any() else 0.17

    events: list[Event] = []
    for slot in ("A", "B"):
        m = slots == slot
        st, sp = times[m], probs[m]
        order = np.argsort(st)
        st, sp = st[order], sp[order]
        events += build_events(st, smooth_probs(st, sp, a.smooth), slot,
                               max_gap=a.max_gap, period=period)
    events.sort(key=lambda e: e.start)
    action_events = [e for e in events if a.quiet_events or e.label not in QUIET_CLASSES]

    span = float(times.max() - times.min())
    print(f"{a.cache.name}: {len(times)} predictions over {mmss(span)} "
          f"({period * 1000:.0f} ms apart), {len(events)} raw runs\n")

    truth = None
    if a.labels:
        truth, two_track = load_intervals(a.labels)
        print(f"labels: {a.labels.name} "
              f"({'two-track' if two_track else 'single-track'}), "
              f"{sum(len(v) for v in truth.values())} intervals")
        cov = sum(e - s for sl in truth for s, e in labelled_spans(truth, sl))
        print(f"labelled fencer-time {cov:.0f} s across {span * 2:.0f} s of "
              f"two-fencer video ({cov / (span * 2):.0%} coverage)\n")

    if a.sweep:
        if truth is None:
            print("--sweep needs a labels file to score against")
            return 1
        return sweep(action_events, truth, float(times.max()))

    kept = gate(action_events, a.min_dur, a.min_conf)
    print(f"gates: duration >= {a.min_dur}s, mean confidence >= {a.min_conf:.0%}  "
          f"->  {len(kept)} of {len(action_events)} action events kept\n")

    print("=== EVENT TIMELINE ===")
    print(f"{'time':>9}  {'fencer':<6} {'action':<9}{'dur':>6}{'conf':>6}{'peak':>6}"
          + ("  verdict" if truth else ""))
    spans = {s: labelled_spans(truth, s) for s in truth} if truth else {}
    shown = kept if a.top in (0, None) else kept[:a.top]
    for ev in shown:
        v = f"  {classify_event(ev, truth, spans)}" if truth else ""
        print(f"{mmss(ev.start):>9}  {SLOT_NAME[ev.slot]:<6} {ev.label:<9}"
              f"{ev.dur:>6.2f}{ev.conf:>6.0%}{ev.peak:>6.0%}{v}")
    if a.top and len(kept) > a.top:
        print(f"  ... {len(kept) - a.top} more (use --top 0 for all)")

    print("\n=== BOUT STATISTICS ===")
    minutes = span / 60.0
    for slot in ("A", "B"):
        se = [e for e in kept if e.slot == slot]
        idle = sum(e.dur for e in events
                   if e.slot == slot and e.label in QUIET_CLASSES)
        print(f"\n  fencer {slot} ({SLOT_NAME[slot]}): {len(se)} actions, "
              f"{len(se) / minutes:.1f}/min")
        counts = Counter(e.label for e in se)
        for c in sorted(counts, key=lambda k: -counts[k]):
            d = [e.dur for e in se if e.label == c]
            print(f"    {c:<9}{counts[c]:>4}  median {np.median(d):.2f}s  "
                  f"longest {max(d):.2f}s  mean conf {np.mean([e.conf for e in se if e.label == c]):.0%}")
        fwd = sum(counts[c] for c in FORWARD)
        back = sum(counts[c] for c in BACKWARD)
        if fwd + back:
            print(f"    forward {fwd} vs backward {back} "
                  f"({fwd / (fwd + back):.0%} forward)")
        print(f"    idle (neutral/walking) {idle:.0f}s")
        if truth:
            true_counts = Counter(l for _, _, l in truth.get(slot, [])
                                  if l not in QUIET_CLASSES and l not in UNSCORABLE)
            line = "  ".join(f"{c} {counts[c]}/{true_counts[c]}"
                             for c in sorted(set(counts) | set(true_counts)))
            print(f"    counted/actual: {line}")

    if truth:
        r = score(kept, truth, end=float(times.max()))
        print("\n=== HOW GOOD IS THIS TIMELINE? ===")
        b = r["buckets"]
        print(f"  {'matched':<12}{b['matched']:>5}   overlaps a true interval of the same class")
        print(f"  {'wrong-class':<12}{b['wrong-class']:>5}   real fencing, wrong action")
        print(f"  {'filler':<12}{b['filler']:>5}   in an unlabelled hole >={FILLER_GAP:.0f}s "
              f"(replay/crowd/graphic) -- ERROR")
        print(f"  {'in-pause':<12}{b['in-pause']:>5}   in a SHORT unlabelled hole "
              f"(live fencing) -- ambiguous, excluded")
        if b["unscorable"]:
            print(f"  {'unscorable':<12}{b['unscorable']:>5}   truth is `extension` "
                  f"(not an emittable class), excluded")
        print(f"\n  event precision  {r['precision']:.0%}")
        print(f"  recall (any overlap)      {r['recall']:.0%}  "
              f"({r['found']}/{r['total']} true actions touched)")
        print(f"  recall (>=50% covered)    {r['strict_recall']:.0%}  "
              f"({r['strict_found']}/{r['total']}) <- the honest one")
        print(f"  {'class':<10}{'any':>6}{'strict':>8}{'of':>5}")
        for c, (any_, strict_, t_) in r["per_class"].items():
            print(f"  {c:<10}{any_:>6}{strict_:>8}{t_:>5}")
        print(f"\n  precision counts matched / (matched + wrong-class + filler); the\n"
              f"  {b['in-pause']} in-pause events are excluded as genuinely unknown.")

    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["fencer", "start", "end", "duration", "action", "confidence",
                        "peak", "n_windows"])
            for ev in kept:
                w.writerow([SLOT_NAME[ev.slot], f"{ev.start:.2f}", f"{ev.end:.2f}",
                            f"{ev.dur:.2f}", ev.label, f"{ev.conf:.3f}",
                            f"{ev.peak:.3f}", ev.n])
        print(f"\nwrote {len(kept)} events to {a.csv}")
    return 0


def sweep(events: list[Event], truth: dict, end: float) -> int:
    """Does gating on duration and confidence actually buy anything?"""
    print("=== GATE SWEEP: does filtering remove filler without losing real events? ===")
    print(f"{'min_dur':>8}{'min_conf':>10}{'events':>8}{'prec':>7}{'rec@any':>9}"
          f"{'rec@50%':>9}{'filler':>8}{'filler%':>9}")
    base = None
    for md in (0.0, 0.25, 0.40, 0.60, 1.00, 1.50, 2.00):
        for mc in (0.0, 0.45, 0.55, 0.65, 0.80):
            k = gate(events, md, mc)
            if not k:
                continue
            r = score(k, truth, end=end)
            ph = r["buckets"]["filler"]
            print(f"{md:>8.2f}{mc:>10.2f}{len(k):>8}{r['precision']:>7.0%}"
                  f"{r['recall']:>9.0%}{r['strict_recall']:>9.0%}"
                  f"{ph:>8}{ph / len(k):>9.0%}")
            if base is None:
                base = (r["precision"], r["recall"], ph / len(k))
    if base:
        print(f"\nungated baseline: precision {base[0]:.0%}, recall {base[1]:.0%}, "
              f"{base[2]:.0%} filler")
        print("Read the SPREAD, not the best cell: if precision barely moves across the\n"
              "grid then duration and confidence carry no information about which events\n"
              "are real, and the gate is decoration.")
    return 0


def _self_test() -> int:
    """Segmentation edge cases, on hand-built streams where the answer is known."""
    idx = {c: i for i, c in enumerate(CLASS_NAMES)}

    def stream(spec):
        """spec = [(time, class)] -> (times, one-hot-ish probs)"""
        t = np.array([s[0] for s in spec], dtype=np.float32)
        p = np.full((len(spec), len(CLASS_NAMES)), 0.05, dtype=np.float32)
        for i, (_, c) in enumerate(spec):
            p[i, idx[c]] = 0.9
        return t, p

    # 1. a plain run merges into ONE event, ending one period after the last call
    t, p = stream([(0.0, "advance"), (0.17, "advance"), (0.34, "advance")])
    ev = build_events(t, p, "A", max_gap=0.4, period=0.17)
    assert len(ev) == 1 and ev[0].label == "advance", ev
    assert abs(ev[0].dur - 0.51) < 1e-4, ev[0].dur

    # 2. a label change splits, even with no time gap
    t, p = stream([(0.0, "advance"), (0.17, "lunge")])
    assert len(build_events(t, p, "A", max_gap=0.4, period=0.17)) == 2

    # 3. THE ONE THAT MATTERS: same label either side of a long untracked gap must
    #    NOT weld into one 40-second action
    t, p = stream([(0.0, "advance"), (40.0, "advance")])
    ev = build_events(t, p, "A", max_gap=0.4, period=0.17)
    assert len(ev) == 2, f"welded across a gap: {ev}"
    assert max(e.dur for e in ev) < 1.0

    # 4. smoothing must not blend across that same gap
    t, p = stream([(0.0, "advance"), (40.0, "lunge")])
    sm = smooth_probs(t, p, 1.0)
    assert np.allclose(sm, p), "smoothing leaked across a 40 s gap"
    #    ...but it DOES blend true neighbours
    t, p = stream([(0.0, "advance"), (0.17, "lunge"), (0.34, "advance")])
    sm = smooth_probs(t, p, 1.0)
    assert CLASS_NAMES[sm[1].argmax()] == "advance", "single-frame blip survived"

    # 5. gates
    evs = [Event("A", "lunge", 0, 0.3, 0.9, 0.9, 2), Event("A", "lunge", 1, 2.0, 0.4, 0.5, 6)]
    assert len(gate(evs, 0.4, 0.0)) == 1 and len(gate(evs, 0.0, 0.5)) == 1
    assert len(gate(evs, 0.0, 0.0)) == 2 and len(gate(evs, 0.4, 0.5)) == 0

    # 6. bucketing, including the long-hole / short-hole split
    truth = {"A": [(10.0, 11.0, "lunge"), (14.0, 15.0, "advance"),
                   (100.0, 101.0, "retreat")]}
    spans = {"A": labelled_spans(truth, "A")}
    gaps = {"A": unlabelled_gaps(truth, "A", 120.0)}
    C = lambda e: classify_event(e, truth, spans, gaps)
    assert C(Event("A", "lunge", 10.1, 10.9, 1, 1, 1)) == "matched"
    assert C(Event("A", "advance", 10.1, 10.9, 1, 1, 1)) == "wrong-class"
    # the 11->14 s hole is 3 s: live fencing, ambiguous
    assert C(Event("A", "lunge", 12.0, 13.0, 1, 1, 1)) == "in-pause"
    # the 15->100 s hole is 85 s: filler, and a real error
    assert C(Event("A", "lunge", 50.0, 51.0, 1, 1, 1)) == "filler"
    # a slot with no labels at all must not crash, and counts as filler
    assert C(Event("B", "lunge", 10.1, 10.9, 1, 1, 1)) == "filler"
    # without gap information every hole is treated as filler (the old behaviour)
    assert classify_event(Event("A", "lunge", 12.0, 13.0, 1, 1, 1), truth, spans) == "filler"
    # in-pause is excluded from precision, not counted against it
    r = score([Event("A", "lunge", 10.1, 10.9, 1, 1, 1),
               Event("A", "lunge", 12.0, 13.0, 1, 1, 1)], truth, end=120.0)
    assert r["buckets"]["in-pause"] == 1 and r["scorable"] == 1
    assert r["precision"] == 1.0, r["precision"]

    # 7. strict recall is not fooled by one long event straddling two intervals
    truth = {"A": [(0.0, 1.0, "advance"), (2.0, 3.0, "advance")]}
    one_long = [Event("A", "advance", 0.0, 3.0, 0.9, 0.9, 18)]
    r = score(one_long, truth)
    assert r["found"] == 2 and r["strict_found"] == 2      # it really does cover both
    half = [Event("A", "advance", 0.9, 2.1, 0.9, 0.9, 7)]  # clips each by 0.1 s
    r = score(half, truth)
    assert r["found"] == 2 and r["strict_found"] == 0, r   # touches both, covers neither

    print("bout_timeline self-test: ok")
    return 0


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        raise SystemExit(_self_test())
    raise SystemExit(main())
