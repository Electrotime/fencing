"""Which engineered features survive a change of VENUE, and why the answer is camera work.

`advance` -> `walking` is the largest single error in the project: 282 of bout 5's 869
true advances. The standing explanation in CLAUDE.md was the CROUCH feature -- knee angle
~140 deg fencing vs ~164 upright -- on the theory that a different camera height shifts
exactly that measurement, making it "one feature deep". That explanation is WRONG, and
the cached windows say so without needing any video:

    ADVANCE vs WALKING, AUC per feature      bout1  bout2  bout3  bout4  bout5
      net_forward   (motion)                  0.73   0.69   0.78   0.73   0.62
      stance_p90    (posture)                 0.82   0.84   0.85   0.82   0.74
      total_travel  (motion)                  0.49   0.56   0.63   0.63   0.34  <- INVERTS
      crouch        (posture)                 0.92   0.87   0.95   0.86   0.79  <- holds

Crouch is bout 5's BEST feature, and its knee-angle gap (18.7 deg) matches bout 4's
(17.2 deg) almost exactly. What collapses is the pair of MOTION features, and
total_travel does not merely weaken, it INVERTS: at bout 5 the median walking window
travels 2.46 against advance's 1.57, the reverse of every other bout.

Posture features are hip-centred and torso-normalised in pose_pipeline, so they are
scale-free and a camera-height change barely touches them. The motion features are not:

    world_vel = diff(hip_x) - pan / PAN_WIDTH

Both terms are frame-relative, and the pan correction is an ESTIMATE. So this script
measures the estimate itself. If bout 5's camera pans differently -- more, or more often
while the fencers are merely walking between touches -- then walking acquires apparent
world motion, advance loses its distinctiveness, and total_travel inverts. That is the
same failure that killed the fencing gate, whose best cue inverted between venues
because "the cue measures camera work".

Cheap on purpose: phase correlation on 320x180 grayscale, no pose and no YOLO, reusing
demo_video._frame_pan so this measures the SAME estimator the features are built on
rather than a lookalike.

RESULT. Venue B's camera pans 1.7x harder than bout 4's (median |pan| 0.664 vs 0.394)
and is still only 16% of the time against 28%. But the aggregate adv/walk RATIO printed
below is NOT the explanation -- it does not line up with the AUCs (bout 1 is 1.00 at
total_travel AUC 0.49, bout 5 is 1.60 at 0.34). The per-window causal test at the bottom
is the one that lands, and it replicates at both venues:

    true-advance windows      called correctly      called walking       AUC
      bout 5 (venue B)        n=348, med 0.843      n=282, med 1.310     0.63
      bout 4 (venue A)        n=551, med 1.453      n= 27, med 2.075     0.79

Camera pan is the mechanism; the venue only sets how often it bites (32% of bout 5's true
advances go wrong against 3.6% of bout 4's).

ONE MECHANISM RULED OUT, recorded so nobody re-runs it. `_frame_pan` returns a silent 0.0
when both strips fall below PAN_MIN_RESPONSE -- asserting "camera still" exactly when it
cannot tell, which would produce this error precisely: fencer advances, camera follows,
diff(hip_x)~0, pan wrongly 0, world_vel~0, reads as walking. Measured on bout 5 the
fallback fires on 0.2% of frames and 0.0% of both the correct and the wrong windows.

MOTION BLUR ALSO RULED OUT. If blur were degrading the pose during fast pans, pose
quality would fall as pan rises. It does not: mean landmark visibility and frozen-joint
fraction both separate high-pan from low-pan windows at AUC 0.51. On the windows that
actually go wrong the pose is if anything CLEANER than on the correct ones (visibility
0.925 vs 0.905, frozen 0.003 vs 0.008) while |pan| separates them at 0.64. The skeleton
is fine; the PAN TERM is what is wrong.

WHICH LEAVES THE ESTIMATOR, and `--estimators` shows how it fails. _frame_pan reads two
narrow border strips (PAN_STRIP_FRAC = 0.22, ~70 px of a 320 px frame). Edges are where
background lives -- but they are also where content ENTERS AND LEAVES during a pan, so
the two strips stop being a pure shift of one another, which is precisely what phase
correlation assumes. Measured on bout 5 against a full-frame correlation over the same
row band:

    frames in the top N% of full-frame motion      strips report    sign disagrees
      top 10%   (n=1730)                             47% of full        18.6%
      top  5%   (n= 865)                             38% of full        19.3%
      top  1%   (n= 173)                             23% of full        20.8%

Overall correlation between the two estimators is 0.304. Widening the strips 2x moves
the estimate toward the full-frame one, which is the direction expected if the narrow
aperture is the deficient part. Note neither is declared ground truth -- the full-frame
estimate can be pulled by the fencers themselves -- but they disagree by 2-4x exactly on
the frames the feature depends on, and one of them is badly wrong.

FIXING IT REQUIRES RE-EXTRACTION AND RETRAINING. The cached windows store normalised
keypoints but not the per-frame hip_x, so `forward` and `total_travel` cannot be
recomputed offline; changing the pan estimator changes two of the six engineered
features, which changes the model's inputs.

usage: py -3 scripts/venue_motion.py [--bouts 1,4,5] [--stride 1]
       py -3 scripts/venue_motion.py --estimators 5     # strips vs full-frame vs wide
"""
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
           "5": "bout5_intervals_2track.csv"}
VENUE = {"1": "A", "2": "A", "3": "A", "4": "A", "5": "B"}


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
    cap = cv2.VideoCapture(str(VID / f"{stem}.mp4"))
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
        video = VID / f"{stem}.mp4"
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

    # ---- the causal test ----------------------------------------------------
    # Aggregate pan is circumstantial. What matters is whether the windows the model
    # gets WRONG are the ones the camera was moving through. Joined per window against
    # a held-out probability cache, so this is the model's real behaviour, not a proxy.
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
