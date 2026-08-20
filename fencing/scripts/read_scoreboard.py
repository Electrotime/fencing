"""Recover touch times and scorer from the broadcast scoreboard, no OCR and no hand labels."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

RAW = PROJECT / "data" / "raw_video"
LAB = PROJECT / "data" / "labels"

# `anchor` is the timer panel: on screen exactly when the scorebug is, and it never
# glows, so it settles whether the overlay is up. Without it the sponsor bar that
# slides through the same band reads as a score change. `wide` is the score pill,
# `digits` its interior in wide-relative coordinates -- the only part a touch moves.
LAYOUT = {
    "7": dict(anchor=(600, 566, 696, 618),
              wide=dict(left=(470, 560, 562, 620), right=(726, 560, 814, 620)),
              digits=dict(left=(30, 12, 80, 52), right=(12, 12, 78, 52)),
              lamp=dict(left=(470, 560, 562, 620), right=(726, 560, 814, 620))),
    # bout 4 lamp boxes were located by contrasting frames at labelled halts against
    # the rest of the video, not by eye: red peaked at (794, 908), green at (1226, 908).
    "4": dict(lamp=dict(left=(760, 890, 830, 926), right=(1192, 890, 1262, 926))),
}

# Lamp brightness does NOT transfer between broadcasts -- bout 7 peaks near 230 over a
# baseline of 2, bout 4 near 100 over 15 -- so the threshold is derived per series.
# Lamps are lit a few percent of the time, so the top of the range is the lit state.
def lamp_threshold(v, frac=0.5):
    lo, hi = np.percentile(v, [50, 99.5])
    return lo + frac * (hi - lo)

# The pill border glows with the lamp colour, which is bright in grayscale and would
# read as a score change. Digits are white and the glow is saturated, so everything
# here runs on min(B,G,R): white survives, coloured glow does not.
WHITE = 175


def sample(video, boxes, stride, cache=None):
    """Per-side min-channel crops of the score pills, every `stride` seconds."""
    if cache and Path(cache).exists():
        d = np.load(cache)
        return d["t"], {k: d[k] for k in boxes}, float(d["fps"])

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * stride)))
    times, acc, i = [], {k: [] for k in boxes}, 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                mn = f.min(axis=2)
                for s, b in boxes.items():
                    acc[s].append(mn[b[1]:b[3], b[0]:b[2]])
                times.append(i / fps)
        i += 1
    cap.release()
    t = np.array(times, dtype=np.float32)
    out = {s: np.stack(v) for s, v in acc.items()}
    if cache:
        np.savez_compressed(cache, t=t, fps=fps, **out)
    return t, out, fps


def lamp_series(video, boxes, stride=0.1, cache=None):
    """Per-side redness, greenness and whiteness of the lamp indicator over time."""
    if cache and Path(cache).exists():
        d = np.load(cache)
        return d["t"], {s: {k: d[f"{s}_{k}"] for k in ("red", "green", "white")}
                        for s in boxes}

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * stride)))
    times, acc, i = [], {s: {k: [] for k in ("red", "green", "white")} for s in boxes}, 0
    while True:
        if not cap.grab():
            break
        if i % step == 0:
            ok, f = cap.retrieve()
            if ok:
                for s, b in boxes.items():
                    c = f[b[1]:b[3], b[0]:b[2]].astype(np.int16)
                    B, G, R = c[:, :, 0], c[:, :, 1], c[:, :, 2]
                    acc[s]["red"].append(np.percentile(R - np.maximum(G, B), 99))
                    acc[s]["green"].append(np.percentile(G - np.maximum(R, B), 99))
                    acc[s]["white"].append(np.percentile(np.minimum(np.minimum(B, G), R), 99))
                times.append(i / fps)
        i += 1
    cap.release()
    t = np.array(times, dtype=np.float32)
    out = {s: {k: np.array(v, dtype=np.float32) for k, v in d.items()} for s, d in acc.items()}
    if cache:
        np.savez_compressed(cache, t=t,
                            **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
    return t, out


def lamp_thresholds(series):
    return {s: lamp_threshold(d["red" if s == "left" else "green"])
            for s, d in series.items()}


def lamps_at(t, series, t0, thr, lo=-0.3, hi=2.0):
    """Which lamps fired at one halt: left / right / both / none.

    `none` means no coloured lamp, which in foil is an off-target (white) hit. The
    white channel is NOT used for that -- it rises at every halt in some broadcasts,
    marking the stoppage rather than the kind of hit.
    """
    m = (t >= t0 + lo) & (t <= t0 + hi)
    if not m.any():
        return "none", {}
    pk = {s: {k: float(v[m].max()) for k, v in d.items()} for s, d in series.items()}
    lit = {s: pk[s]["red" if s == "left" else "green"] > thr[s] for s in series}
    if lit["left"] and lit["right"]:
        return "both", pk
    if lit["left"]:
        return "left", pk
    if lit["right"]:
        return "right", pk
    return "none", pk


def otsu(x, bins=256):
    """Threshold splitting a bimodal 1-D distribution."""
    h, edges = np.histogram(x, bins=bins)
    p = h / h.sum()
    w0 = np.cumsum(p)
    m = np.cumsum(p * ((edges[:-1] + edges[1:]) / 2))
    mt = m[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        var = (mt * w0 - m) ** 2 / (w0 * (1 - w0))
    return float(edges[1:][np.nanargmax(var)])


def presence(arr):
    """True where the overlay is on screen, from the panel's own static pixels.

    The median frame is the overlay because it is up more than half the time, and
    the pixels that barely move around it are the panel furniture. Distance to
    those is bimodal but heavily skewed -- overlay-up sits near 0 while cutaways
    spread over a wide tail -- so the split is taken in log space.
    """
    med = np.median(arr, axis=0)
    mad = np.median(np.abs(arr - med), axis=0)
    anchor = mad <= np.percentile(mad, 40)
    d = np.abs(arr[:, anchor].astype(np.int16) - med[anchor].astype(np.int16)).mean(axis=1)
    return np.log1p(d) < otsu(np.log1p(d)), d


def digit_masks(arr, box, ok):
    """White-pixel masks with the always-lit furniture removed.

    The pill has a bright specular arc inside the crop. Left in, it is most of the
    mask, so swapping one digit for another moves under 4% of pixels and no
    threshold separates a real change from compression noise.
    """
    m = arr[:, box[1]:box[3], box[0]:box[2]] > WHITE
    const = m[ok].mean(axis=0) > 0.9 if ok.any() else np.zeros(m.shape[1:], bool)
    return m & ~const


def dist(a, b):
    """Changed pixels as a fraction of the digit itself, not of the whole box."""
    return (a != b).sum() / max(int(a.sum()), int(b.sum()), 1)


def states(masks, ok, times, persist, stride, tol):
    """Committed digit changes: (time, previous mask, new mask)."""
    need = max(2, int(round(persist / stride)))
    cur, events = None, []
    run, run_t, run_n = None, None, 0
    for i in range(len(masks)):
        if not ok[i]:
            run, run_n = None, 0
            continue
        m = masks[i]
        if cur is None:
            cur = m
            continue
        if dist(m, cur) <= tol:
            run, run_n = None, 0
            continue
        if run is not None and dist(m, run) <= tol:
            run_n += 1
        else:
            run, run_t, run_n = m, times[i], 1
        if run_n >= need:
            events.append((float(run_t), cur, run))
            cur, run, run_n = run, None, 0
    return events


def decode(masks, ok, times, persist, stride, tol):
    """Times at which this side's digit reaches a state it has never held before.

    A score only ever goes up, one at a time, so a digit image the side already
    showed cannot be its new score. Requiring novelty is what separates a touch
    from the overlay flapping through garbage during a replay or a sponsor bar.
    """
    need = max(2, int(round(persist / stride)))
    idx = np.flatnonzero(ok)
    cents, lab = [], np.full(len(masks), -1)
    for i in idx:
        best, bd = -1, 9.0
        for k, c in enumerate(cents):
            d = dist(masks[i], c)
            if d < bd:
                best, bd = k, d
        if bd <= tol:
            lab[i] = best
        else:
            cents.append(masks[i])
            lab[i] = len(cents) - 1

    seen, events, cur = set(), [], None
    run_k, run_n, run_t = -1, 0, None
    for i in idx:
        k = int(lab[i])
        if k == run_k:
            run_n += 1
        else:
            run_k, run_n, run_t = k, 1, times[i]
        if run_n >= need and k != cur and k not in seen:
            if cur is not None:
                events.append(float(run_t))
            seen.add(k)
            cur = k
    return events


def read(bout, stride=0.25, persist=2.0, tol=0.35, cache=None):
    lay = LAYOUT[bout]
    boxes = dict(lay["wide"], anchor=lay["anchor"])
    t, crops, _ = sample(RAW / f"{bout}.mp4", boxes, stride, cache)
    ok, _ = presence(crops["anchor"])
    ev = []
    for side in ("left", "right"):
        digits = digit_masks(crops[side], lay["digits"][side], ok)
        ev += [(u, side) for u in decode(digits, ok, t, persist, stride, tol)]
    return sorted(ev), {"overlay": ok.mean()}, len(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout", default="7")
    ap.add_argument("--stride", type=float, default=0.25)
    ap.add_argument("--persist", type=float, default=3.0)
    ap.add_argument("--tol", type=float, default=0.15)
    ap.add_argument("--cache")
    ap.add_argument("--sweep", action="store_true")
    a = ap.parse_args()

    truth_file = LAB / f"bout{a.bout}_touches.tsv"
    truth = None
    if truth_file.exists():
        import check_touches as CT
        _, rows = CT.check(truth_file)
        truth = [(r["t"], r["scorer"]) for r in rows if r["scorer"] != "none"]

    def evaluate(ev):
        used, lags, wrong = set(), [], 0
        missed = []
        for t, sc in truth:
            best = None
            for i, (u, s) in enumerate(ev):
                if i in used or not (0 <= u - t <= 8):
                    continue
                if best is None or u < ev[best][0]:
                    best = i
            if best is None:
                missed.append((t, sc))
                continue
            used.add(best)
            lags.append(ev[best][0] - t)
            if ev[best][1] != sc:
                wrong += 1
        return lags, wrong, missed, [ev[i] for i in range(len(ev)) if i not in used]

    if a.sweep and truth:
        print(f"  {'persist':>8}{'tol':>7}{'events':>8}{'hit':>6}{'wrong':>7}"
              f"{'spur':>6}{'lag':>7}")
        for persist in (1.0, 2.0, 3.0, 4.0, 6.0):
            for tol in (0.15, 0.25, 0.40, 0.60):
                ev, cover, _ = read(a.bout, a.stride, persist, tol, a.cache)
                lags, wrong, missed, extra = evaluate(ev)
                print(f"  {persist:>8.1f}{tol:>7.2f}{len(ev):>8}"
                      f"{len(lags):>4}/{len(truth):<2}{wrong:>6}{len(extra):>6}"
                      f"{np.median(lags) if lags else float('nan'):>7.2f}")
        return 0

    ev, cover, n = read(a.bout, a.stride, a.persist, a.tol, a.cache)
    print(f"bout {a.bout}: {n} samples at {a.stride}s, "
          f"overlay up {cover['overlay']:.0%}")
    print(f"detected {len(ev)} score changes "
          f"({sum(1 for _, s in ev if s == 'left')} left / "
          f"{sum(1 for _, s in ev if s == 'right')} right)")
    if truth is None:
        for u, s in ev:
            print(f"  {u:8.2f}  {s}")
        return 0

    lags, wrong, missed, extra = evaluate(ev)
    print(f"\nvs {truth_file.name}: {len(truth)} scoring touches")
    print(f"  matched {len(lags)}/{len(truth)}   wrong side {wrong}   "
          f"spurious {len(extra)}")
    if missed:
        print("  missed: " + ", ".join(f"{t:.1f}s {s}" for t, s in missed))
    if extra:
        print("  spurious: " + ", ".join(f"{t:.1f}s {s}" for t, s in extra[:10]))
    if lags:
        lags = np.array(lags)
        print(f"  scoreboard lag: median {np.median(lags):.2f}s, "
              f"{lags.min():.2f} to {lags.max():.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
