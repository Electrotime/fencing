"""Score camera-pan estimators against known ground truth."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import demo_video as D

VID = PROJECT / "data" / "raw_video"
OUT_W, OUT_H = 320, 180          # what the pipeline feeds the estimator
CROP_FRAC = 0.88                 # crop this much of the width, leaving room to offset
# offsets in OUTPUT pixels; real |pan| p99 is ~9-11
SHIFTS = [-16.0, -8.0, -4.0, -2.0, 2.0, 4.0, 8.0, 16.0]
ROWS = slice(int(0.10 * OUT_H), int(0.75 * OUT_H))   # same band _frame_pan uses


# ---- estimators ----
# each takes two 320x180 float32 grays, returns horizontal shift in output px
_cache = {}


def _han(key, w, h):
    if key not in _cache:
        _cache[key] = cv2.createHanningWindow((w, h), cv2.CV_32F)
    return _cache[key]


def est_strips(a, b):
    return D._frame_pan(a, b, _cache.setdefault("_ship", {}))


def _border(a, b, frac):
    h, w = a.shape
    sw = max(10, int(frac * w))
    win = _han(f"b{frac}", sw, ROWS.stop - ROWS.start)
    out = []
    for x0, x1 in [(0, sw), (w - sw, w)]:
        (dx, _), r = cv2.phaseCorrelate(a[ROWS, x0:x1], b[ROWS, x0:x1], win)
        if r > D.PAN_MIN_RESPONSE:
            out.append(dx)
    return float(np.median(out)) if out else 0.0


def est_wide(a, b):
    return _border(a, b, 2 * D.PAN_STRIP_FRAC)


def est_full(a, b):
    h, w = a.shape
    win = _han("full", w, ROWS.stop - ROWS.start)
    (dx, _), r = cv2.phaseCorrelate(a[ROWS], b[ROWS], win)
    return float(dx) if r > D.PAN_MIN_RESPONSE else 0.0


def est_tiles(a, b):
    """Median over a grid of tiles; fencer tiles get outvoted by background ones."""
    h, w = a.shape
    nx, ny = 4, 2
    tw, th = w // nx, (ROWS.stop - ROWS.start) // ny
    win = _han("tile", tw, th)
    out = []
    for iy in range(ny):
        for ix in range(nx):
            y0 = ROWS.start + iy * th
            x0 = ix * tw
            pa = a[y0:y0 + th, x0:x0 + tw]
            pb = b[y0:y0 + th, x0:x0 + tw]
            (dx, _), r = cv2.phaseCorrelate(pa, pb, win)
            if r > D.PAN_MIN_RESPONSE:
                out.append(dx)
    return float(np.median(out)) if out else 0.0


ESTIMATORS = [("strips (shipped)", est_strips), ("wide strips", est_wide),
              ("full frame", est_full), ("tiles median", est_tiles)]


# ---- benchmark -------------------------------------------------------------
def crop_to(frame, x_src, cw, ch, y_src):
    patch = frame[y_src:y_src + ch, x_src:x_src + cw]
    small = cv2.resize(patch, (OUT_W, OUT_H), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


def run(stem, n_samples):
    cap = cv2.VideoCapture(str(VID / f"{stem}.mp4"))
    if not cap.isOpened():
        raise SystemExit(f"cannot open bout {stem}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cw = int(CROP_FRAC * W)
    ch = int(cw * OUT_H / OUT_W)
    if ch > H:
        ch = H
        cw = int(ch * OUT_W / OUT_H)
    y_src = (H - ch) // 2
    scale = OUT_W / cw                      # source px -> output px
    max_off = W - cw                        # room available to slide the crop
    x_base = max_off // 2

    # evenly spaced frames, skipping the first/last few seconds
    lo, hi = int(total * 0.05), int(total * 0.95)
    picks = np.linspace(lo, hi, n_samples).astype(int)

    rows = []          # (test, estimator_index, true, est)
    for fi in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, f0 = cap.read()
        if not ok:
            continue
        ok, f1 = cap.read()
        if not ok:
            continue
        for d_out in SHIFTS:
            d_src = int(round(d_out / scale))
            if abs(d_src) > x_base:
                continue
            # PURE: one frame, two crops -> true pan is exactly d_out
            a = crop_to(f0, x_base, cw, ch, y_src)
            b = crop_to(f0, x_base + d_src, cw, ch, y_src)
            true = d_src * scale
            for i, (_, fn) in enumerate(ESTIMATORS):
                rows.append(("pure", i, true, fn(a, b)))
            # REAL: consecutive frames, second slid d further along
            b0 = crop_to(f1, x_base, cw, ch, y_src)
            b1 = crop_to(f1, x_base + d_src, cw, ch, y_src)
            for i, (_, fn) in enumerate(ESTIMATORS):
                rows.append(("real", i, true, fn(a, b1) - fn(a, b0)))
    cap.release()
    return rows


def summarise(rows, test, n_est):
    print(f"\n  {'estimator':<20}{'slope':>8}{'RMSE px':>10}{'sign err':>10}"
          f"{'|d|>=8 slope':>14}{'n':>7}")
    for i in range(n_est):
        sel = [(t, e) for tst, j, t, e in rows if tst == test and j == i]
        if not sel:
            continue
        tr = np.array([t for t, _ in sel])
        es = np.array([e for _, e in sel])
        # least squares through the origin, sign-agnostic
        slope = float((tr * es).sum() / (tr * tr).sum())
        s = np.sign(slope) or 1.0
        es_al = es * s
        rmse = float(np.sqrt(((es_al - tr) ** 2).mean()))
        sign_err = float((np.sign(es_al) != np.sign(tr)).mean())
        big = np.abs(tr) >= 8
        bslope = (float((tr[big] * es_al[big]).sum() / (tr[big] ** 2).sum())
                  if big.any() else float("nan"))
        print(f"  {ESTIMATORS[i][0]:<20}{abs(slope):>8.2f}{rmse:>10.2f}"
              f"{sign_err:>10.1%}{bslope:>14.2f}{len(sel):>7}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default="4,5")
    ap.add_argument("--samples", type=int, default=150)
    a = ap.parse_args()

    allrows = []
    for stem in a.bouts.split(","):
        if not (VID / f"{stem}.mp4").exists():
            print(f"bout {stem}: no video, skipping")
            continue
        r = run(stem, a.samples)
        allrows += r
        print(f"bout {stem}: {len(r) // (2 * len(ESTIMATORS))} shifted pairs measured")

    if not allrows:
        print("nothing measured")
        return 1
    print("\n=== PURE: one frame cropped twice, true pan known exactly ===")
    summarise(allrows, "pure", len(ESTIMATORS))
    print("\n=== REAL: consecutive frames, response to an INJECTED extra pan ===")
    summarise(allrows, "real", len(ESTIMATORS))
    print("\nslope 1.00 = true magnitude. Read the |d|>=8 column hardest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
