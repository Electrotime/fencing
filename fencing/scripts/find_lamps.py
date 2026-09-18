"""Locate a broadcast's scoring lamps by contrast, so a new venue needs no hand setup."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import read_scoreboard as RS

RAW = PROJECT / "data" / "raw_video"
LAB = PROJECT / "data" / "labels"

# A contested halt lights BOTH lamps, so the same frames locate red and green at
# once. The graphic lags the touch, hence the window rather than a single frame.
LIT_FROM, LIT_TO = 0.3, 1.8
N_LIT = 4          # frames sampled per halt
N_BASE = 240       # baseline frames, spread over the whole video
AWAY = 6.0         # s a baseline frame must keep from any halt
FRAC = 0.55        # keep pixels this fraction of the way to the peak


def has_table(stem):
    """A usable contested table -- ask the parser, not the file or the bout list."""
    if not (LAB / f"bout{stem}_contested.tsv").exists():
        return False
    import exp_contested as C
    try:
        return len(C.read_contested(stem)) > 0
    except Exception:
        return False


def halt_times(stem):
    """Hand table where there is one, lamp-derived halts for the bouts with a box."""
    if has_table(stem):
        import exp_contested as C
        return [t for t, _, _ in C.read_contested(stem)]
    import find_halts as FH
    return FH.truth(stem)[0]


def _grab(cap, fps, t):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
    ok, f = cap.retrieve() if False else cap.read()
    return f if ok else None


def contrast(video, halts, n_base=N_BASE):
    """Mean frame at lit moments minus mean frame everywhere else."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) / fps

    lit, n = None, 0
    for h in halts:
        for u in np.linspace(LIT_FROM, LIT_TO, N_LIT):
            f = _grab(cap, fps, h + u)
            if f is None:
                continue
            lit = f.astype(np.float64) if lit is None else lit + f
            n += 1
    if not n:
        cap.release()
        raise SystemExit("no lit frames read")
    lit /= n

    rng = np.random.default_rng(0)
    base, m, tries = None, 0, 0
    while m < n_base and tries < n_base * 20:
        tries += 1
        u = float(rng.uniform(0, max(dur - 1, 1)))
        if any(abs(u - h) < AWAY for h in halts):
            continue
        f = _grab(cap, fps, u)
        if f is None:
            continue
        base = f.astype(np.float64) if base is None else base + f
        m += 1
    cap.release()
    return lit - base / max(m, 1), n, m


def _diff_path(stem):
    return LAB / f"{stem}_lampdiff.npz"


def diff_map(stem, video=None, refresh=False):
    """Lit-minus-baseline frame difference, cached -- the expensive half."""
    p = _diff_path(stem)
    if p.exists() and not refresh:
        d = np.load(p)
        return d["d"], [float(x) for x in d["halts"]]
    halts = halt_times(stem)
    d, n, m = contrast(video or RAW / f"{stem}.mp4", halts)
    np.savez_compressed(p, d=d.astype(np.float32), halts=np.array(halts, np.float64))
    return d.astype(np.float32), halts


def peaks(g, k=12, sup=60):
    """Top-k local maxima, non-max suppressed by `sup` pixels."""
    g = cv2.GaussianBlur(g.astype(np.float32), (0, 0), 3)
    work, out = g.copy(), []
    for _ in range(k):
        y, x = np.unravel_index(int(work.argmax()), work.shape)
        v = float(work[y, x])
        if v <= 1.0:
            break
        out.append((v, int(x), int(y)))
        work[max(0, y - sup):y + sup, max(0, x - sup):x + sup] = -1e9
    return out, g


def blob(g, x, y, frac=FRAC):
    """Bounding box of the connected region around one peak."""
    mask = (g >= frac * float(g[y, x])).astype(np.uint8)
    _, lab = cv2.connectedComponents(mask)
    ys, xs = np.where(lab == lab[y, x])
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


# Lamps sit at the same height, red left of green, one design mirrored. Scoring
# the PAIR on that beats taking each colour's global peak.
DY = 45      # px the two lamp centres may differ vertically
DX = 40      # px they must differ horizontally
SYM = 2.5    # the larger box may exceed the smaller by at most this, per axis
MIN_TWO = 8  # fewer two-light halts than this and a FAIL is not distinguishable from noise


def _sym(a, b):
    """1.0 for identically sized boxes, falling off as they diverge."""
    wa, ha = a[2] - a[0], a[3] - a[1]
    wb, hb = b[2] - b[0], b[3] - b[1]
    rw = max(wa, wb) / max(min(wa, wb), 1)
    rh = max(ha, hb) / max(min(ha, hb), 1)
    return (1.0 / rw) * (1.0 / rh), max(rw, rh)


def locate(stem, video=None, refresh=False):
    d, halts = diff_map(stem, video, refresh)
    B, G, R = d[:, :, 0], d[:, :, 1], d[:, :, 2]
    rp, rmap = peaks(R - np.maximum(G, B))
    gp, gmap = peaks(G - np.maximum(R, B))
    best = None
    for rv, rx, ry in rp:
        for gv, gx, gy in gp:
            if abs(ry - gy) > DY or gx - rx < DX:
                continue
            rb, gb = blob(rmap, rx, ry), blob(gmap, gx, gy)
            sym, worst = _sym(rb, gb)
            if worst > SYM:
                continue
            s = min(rv, gv) * sym
            if best is None or s > best[0]:
                best = (s, rb, gb, rv, gv, worst)
    if best is None:
        return dict(stem=stem, left=None, right=None, score=0.0, asym=None,
                    red_peak=rp[0][0] if rp else 0.0,
                    green_peak=gp[0][0] if gp else 0.0, halts=halts)
    s, rb, gb, rv, gv, worst = best
    return dict(stem=stem, left=rb, right=gb, score=s, asym=worst,
                red_peak=rv, green_peak=gv, halts=halts)


def two_light_times(stem):
    """Halts known to have lit BOTH lamps -- the hand table, or the known box."""
    if has_table(stem):
        return halt_times(stem)                      # contested tables are all two-light
    t, ser = RS.lamp_series("", RS.LAYOUT[stem]["lamp"], 0.1, LAB / f"{stem}_lamp.npz")
    known = RS.detect_halts(t, ser, RS.lamp_all_thresholds(ser))
    return [h["t"] for h in known if h["lights"] == "both"]


def verify(stem, boxes, halts, tol=2.5, video=None):
    """Recall is not enough: a real box also separates the sides, so known
    two-light halts have to come back two-colour.
    """
    t, ser = RS.lamp_series(str(video or RAW / f"{stem}.mp4"), boxes, 0.1,
                            LAB / f"{stem}_lampfound.npz")
    found = RS.detect_halts(t, ser, RS.lamp_all_thresholds(ser))

    def near(u):
        c = [h for h in found if abs(h["t"] - u) <= tol]
        return min(c, key=lambda h: abs(h["t"] - u)) if c else None

    hit = sum(near(u) is not None for u in halts)
    twos = [near(u) for u in two_light_times(stem)]
    seen = [h for h in twos if h is not None]
    both = sum(1 for h in seen if h["lights"] == "both")
    overlap = not (boxes["left"][2] <= boxes["right"][0]
                   or boxes["right"][2] <= boxes["left"][0])
    extra = len(found) / max(len(halts), 1)
    good = (not overlap) and hit >= 0.8 * len(halts) and both >= 0.8 * max(len(seen), 1)
    # Below MIN_TWO the two-colour rate is one halt wide: bout 12 failed at 3/5
    # with a box that matches bout 6's verified one to three pixels.
    verdict = "PASS" if good else ("INCONCLUSIVE" if len(seen) < MIN_TWO else "FAIL")
    return dict(found=len(found), hit=hit, recall=hit / max(len(halts), 1),
                both=both, n_two=len(seen), verdict=verdict,
                both_frac=both / max(len(seen), 1), overlap=overlap, extra=extra,
                ok=good)


def _self_test():
    """The pair search must prefer two aligned lamps over a brighter lone blob."""
    d = np.zeros((300, 600, 3), np.float32)
    d[140:160, 100:140, 2] = 60.0               # red lamp, left
    d[140:160, 300:340, 1] = 55.0               # green lamp, same height
    d[20:40, 500:540, 2] = 200.0                # brighter red decoy, wrong height
    B, G, R = d[:, :, 0], d[:, :, 1], d[:, :, 2]
    rp, rmap = peaks(R - np.maximum(G, B))
    gp, gmap = peaks(G - np.maximum(R, B))
    assert rp[0][1] > 450, rp[0]                # decoy IS the global red peak
    best = None
    for rv, rx, ry in rp:
        for gv, gx, gy in gp:
            if abs(ry - gy) > DY or gx - rx < DX:
                continue
            if best is None or min(rv, gv) > best[0]:
                best = (min(rv, gv), (rx, ry), (gx, gy))
    assert best is not None and 90 < best[1][0] < 150, best
    assert 130 < best[1][1] < 170, best
    print(f"  self-test ok: pair search took the aligned lamp at {best[1]}, "
          f"not the brighter decoy at ({rp[0][1]}, {rp[0][2]})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default="8,9,10,11,12,13,14")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="re-read each located box over the whole video")
    ap.add_argument("--refresh", action="store_true", help="ignore the diff cache")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    for stem in a.bouts.split(","):
        r = locate(stem, refresh=a.refresh)
        print(f"bout {stem}: {len(r['halts'])} halts")
        print(f"   left  (red)   {r['left']}   peak {r['red_peak']:5.1f}")
        asym = f"   asym {r['asym']:.2f}" if r["asym"] else ""
        print(f"   right (green) {r['right']}   peak {r['green_peak']:5.1f}{asym}")
        if a.verify and r["left"] and r["right"]:
            v = verify(stem, dict(left=r["left"], right=r["right"]), r["halts"])
            print(f"   verify: {v['verdict']:<12}  "
                  f"{v['hit']}/{len(r['halts'])} halts ({v['recall']:.0%}), "
                  f"{v['both']}/{v['n_two']} two-light read both "
                  f"({v['both_frac']:.0%}), {v['found']} detected "
                  f"({v['extra']:.2f}x)"
                  f"{', BOXES OVERLAP' if v['overlap'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
