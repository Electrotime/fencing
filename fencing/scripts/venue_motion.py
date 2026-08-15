"""Which engineered features survive a change of VENUE, and why the answer is camera work."""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import demo_video as D
from src.action_model import CLASS_NAMES, PAN_WIDTH
from src.labels import load_intervals


def auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    v = np.concatenate([pos, neg])
    order = v.argsort()
    r = np.empty(len(v), float)
    r[order] = np.arange(1, len(v) + 1)
    _, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, r)
    r = (sums / cnt)[inv]
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))

LAB = PROJECT / "data" / "labels"
VID = PROJECT / "data" / "raw_video"
CSV_FOR = {"1": "bout1_intervals.csv", "2": "bout2_intervals.csv",
           "3": "bout3_intervals_2track.csv", "4": "bout4_intervals_2track.csv",
           "5": "bout5_intervals_2track.csv", "7": "bout7_intervals_2track.csv"}
VENUE = {"1": "A", "2": "A", "3": "A", "4": "A", "5": "B", "7": "C"}
# bout 7 must be read from 7_30fps.mp4; the original is 60 fps and every window
# constant in this project is counted in frames
VIDEO_FOR = {"7": "7_30fps.mp4"}


def video_for(stem):
    return VID / VIDEO_FOR.get(stem, f"{stem}.mp4")


def pan_track(video, stride):
    """Per-frame horizontal pan estimate, in the same units the features consume."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"couldn't open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    prev, windows, out = None, {}, []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)),
                                cv2.COLOR_BGR2GRAY).astype(np.float32)
            out.append((idx / fps, D._frame_pan(prev, gray, windows)))
            prev = gray
        idx += 1
    cap.release()
    return np.array([t for t, _ in out]), np.array([p for _, p in out], np.float32), fps


def compare_estimators(stem):
    """strips (shipped) vs full-frame vs 2x-wide strips, on the same frame pairs."""
    cap = cv2.VideoCapture(str(video_for(stem)))
    prev, win, full_win = None, {}, None
    S, F, W = [], [], []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(fr, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            h, w = g.shape
            rows = slice(int(0.10 * h), int(0.75 * h))
            S.append(D._frame_pan(prev, g, win))
            if full_win is None:
                full_win = cv2.createHanningWindow((w, rows.stop - rows.start), cv2.CV_32F)
            (dxf, _), rf = cv2.phaseCorrelate(prev[rows], g[rows], full_win)
            F.append(dxf if rf > D.PAN_MIN_RESPONSE else 0.0)
            sw = max(10, int(2 * D.PAN_STRIP_FRAC * w))
            if "wide" not in win:
                win["wide"] = cv2.createHanningWindow((sw, rows.stop - rows.start),
                                                      cv2.CV_32F)
            sh = []
            for a_, b_ in [(0, sw), (w - sw, w)]:
                (dx, _), r = cv2.phaseCorrelate(prev[rows, a_:b_], g[rows, a_:b_],
                                                win["wide"])
                if r > D.PAN_MIN_RESPONSE:
                    sh.append(dx)
            W.append(float(np.median(sh)) if sh else 0.0)
        prev = g
    cap.release()

    S, F, W = np.asarray(S), np.asarray(F), np.asarray(W)
    print(f"bout {stem}: {len(S)} frame pairs\n")
    print(f"{'estimator':<10}{'med |pan|':>11}{'p90':>9}{'p99':>9}{'corr vs strips':>16}")
    for nm, v in (("strips", S), ("full", F), ("wide", W)):
        print(f"{nm:<10}{np.median(np.abs(v)):>11.3f}{np.percentile(np.abs(v), 90):>9.3f}"
              f"{np.percentile(np.abs(v), 99):>9.3f}{np.corrcoef(v, S)[0, 1]:>16.3f}")
    for q in (90, 95, 99):
        thr = np.percentile(np.abs(F), q)
        m = np.abs(F) >= thr
        print(f"\ntop {100 - q}% of full-frame motion (|full| >= {thr:.2f}, n={int(m.sum())}):")
        print(f"   full {np.abs(F[m]).mean():.3f}   strips {np.abs(S[m]).mean():.3f}"
              f"   = {np.abs(S[m]).mean() / np.abs(F[m]).mean():.0%} of full")
        print(f"   sign disagreement {float((np.sign(S[m]) != np.sign(F[m])).mean()):.1%}")
    print("\nNeither is ground truth -- a full-frame correlation can be pulled by the")
    print("fencers. But they disagree 2-4x exactly where the feature needs the pan, and")
    print("widening the strips moves them TOWARD the full-frame answer.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default="1,4,5")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--estimators", metavar="BOUT",
                    help="compare pan estimators on one bout instead")
    a = ap.parse_args()
    if a.estimators:
        return compare_estimators(a.estimators)

    print(f"{'bout':<6}{'venue':>6}{'frames':>9}{'|pan| med':>11}{'|pan| p90':>11}"
          f"{'still %':>9}{'adv |pan|':>11}{'walk |pan|':>12}{'ratio':>8}")
    rows, pan_cache = {}, {}
    for stem in a.bouts.split(","):
        video = video_for(stem)
        if not video.exists():
            print(f"{stem:<6}  (no video)")
            continue
        t, pan, fps = pan_track(video, a.stride)
        pan_cache[stem] = (t, pan, fps)
        ap_ = np.abs(pan)
        # per-class pan: how hard is the camera working during each labelled class?
        truth, _ = load_intervals(LAB / CSV_FOR[stem])
        spans = {}
        for slot in truth:
            for s, e, lab in truth[slot]:
                spans.setdefault(lab, []).append((s, e))

        def pan_during(label):
            m = np.zeros(len(t), bool)
            for s, e in spans.get(label, []):
                m |= (t >= s) & (t < e)
            return ap_[m] if m.any() else np.array([])

        adv, walk = pan_during("advance"), pan_during("walking")
        ma = float(np.median(adv)) if len(adv) else float("nan")
        mw = float(np.median(walk)) if len(walk) else float("nan")
        # "still" = pan below a tenth of a pixel at 320 wide, i.e. a locked-off camera
        still = float((ap_ < 0.1).mean())
        print(f"{stem:<6}{VENUE[stem]:>6}{len(t):>9}{np.median(ap_):>11.3f}"
              f"{np.percentile(ap_, 90):>11.3f}{still:>9.0%}{ma:>11.3f}{mw:>12.3f}"
              f"{ma / mw if mw else float('nan'):>8.2f}")
        rows[stem] = (ap_, adv, walk)

    print(f"\n(PAN_WIDTH = {PAN_WIDTH}; world_vel = diff(hip_x) - pan/PAN_WIDTH)")
    print("Venue B's camera is the busiest of the three -- but the adv/walk ratio column")
    print("does NOT line up with the total_travel AUCs (bout 1 is 1.00 at AUC 0.49, bout 5")
    print("is 1.60 at 0.34), so it does not on its own explain the inversion. The direct")
    print("test is below: does the camera moving harder actually make the model wrong?")

    for stem, cache in (("5", "5_probs_held.npz"), ("4", "4_probs_heldb5.npz")):
        if stem not in rows or not (LAB / cache).exists():
            continue
        t, pan, _ = pan_cache[stem]
        ap_ = np.abs(pan)
        d = np.load(LAB / cache)
        s, wt, P = d["slot"].astype(str), d["time"], d["probs"]
        truth, _ = load_intervals(LAB / CSV_FOR[stem])

        def truth_at(sl, x):
            for a_, b_, l in truth.get(sl, []):
                if a_ <= x < b_:
                    return l
            return None

        # mean |pan| over the 2 s window each call was made on
        lo = np.searchsorted(t, wt - 2.0, side="left")
        hi = np.searchsorted(t, wt, side="right")
        wpan = np.array([ap_[a_:b_].mean() if b_ > a_ else np.nan
                         for a_, b_ in zip(lo, hi)])
        pred = np.array(CLASS_NAMES)[P.argmax(1)]
        gt = np.array([truth_at(a_, float(b_)) for a_, b_ in zip(s, wt)])

        adv = (gt == "advance") & np.isfinite(wpan)
        ok = adv & (pred == "advance")
        bad = adv & (pred == "walking")
        if ok.sum() and bad.sum():
            print(f"\nbout {stem}, true-ADVANCE windows ({int(adv.sum())} of them):")
            print(f"  called advance correctly  n={int(ok.sum()):4d}   mean |pan| "
                  f"{wpan[ok].mean():.3f}   median {np.median(wpan[ok]):.3f}")
            print(f"  called WALKING instead    n={int(bad.sum()):4d}   mean |pan| "
                  f"{wpan[bad].mean():.3f}   median {np.median(wpan[bad]):.3f}")
            print(f"  AUC(|pan| separates the error) = {auc(wpan[bad], wpan[ok]):.2f}"
                  "   0.50 = camera motion has nothing to do with it")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
