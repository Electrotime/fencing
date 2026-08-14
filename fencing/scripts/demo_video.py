"""Minimal end-to-end demo: run the whole stack on a video and save an annotated copy.

Per frame: YOLO finds the fencer(s) -> each one gets cropped and run through
MediaPipe pose -> sliding windows of keypoints feed the action LSTM at two
scales (a short one so brief actions like parries aren't buried, a long one
for sustained footwork) -> skeleton, boxes, labels and blade tip get drawn.
Works with one or two fencers. Offline, writes <video>_demo.mp4 next to the input.

Run from project root:  python scripts/demo_video.py path/to/video.mp4 [out.mp4]
"""

from __future__ import annotations

import sys
from collections import deque
from contextlib import ExitStack
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.action_model import (ActionFrameLSTM, ActionLSTM, CLASS_NAMES, INPUT_SIZE,
                              N_AGG_WIDE, SEQ_LEN, _engineered_features,
                              frame_logits_to_window, load_action_model, wide_agg)
from src.blade_detector import get_blade_tip, load_blade_model
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import (N_LANDMARKS, VISIBILITY_THRESHOLD,
                               _landmarks_to_array, _make_landmarker,
                               _normalize_sequence)
from src.utils import draw_action_label, draw_blade_tip, draw_skeleton

# action_opp5: continuous windows from FIVE labelled bouts (bout 5 is a second
# VENUE), `last` pooling, and each fencer given the OPPONENT's features
# (train_shipping.py --ship --opponent).
# NO CLIPS -- they are single-fencer files, so their opponent block would be all
# zeros and perfectly correlated with "came from a clip"; measured harmful (-3.4 on
# bout 1) versus +0.5/+2.1/+6.2 for continuous-only.
#
# End-to-end on a HELD-OUT bout 1, all four the same pipeline:
#   action_lstm  (clips only, mean pool, + prior)     43.4%
#   action_cont  (clips+continuous, last pool)        74.0%
#   action_opp   (4 bouts, last pool, opponent)       74.6%
#   action_opp5  (5 bouts, adds a second venue)       76.4%
# Adding bout 5 was verified as a matched A/B -- same recipe, only the training
# corpus differs -- on TWO held-out bouts, because a five-bout model cannot be
# honestly scored on any of the five:
#   held-out bout 1:  74.6% -> 76.4%  (+1.8)
#   held-out bout 4:  67.6% -> 71.3%  (+3.7, on 3822 windows)
# Every older checkpoint is kept for comparison, not deleted. Note action_lstm is
# `mean` pooling and 6-agg, so switching back needs POOL_MODE and USE_OPPONENT too.
#
# The cross-venue figure this replaces is SPENT: action_opp scored 58.1% on bout 5
# while bout 5 was unseen, and now that bout 5 is in training that measurement can
# never be repeated. A third venue is needed for the next honest one.
MODEL_PATH = PROJECT_ROOT / "models" / "action_opp5.pth"
# The checkpoint's time-reduction mode. MUST match how MODEL_PATH was trained --
# all modes share the same parameter shapes, so a mismatch loads cleanly and just
# behaves wrong. `last` measured +4 to +5 pts over `mean` on all three held-out
# bouts, with the gain concentrated in the transient and quiet classes (bout 4:
# lunge 37->56%, advance 30->43%). See ActionLSTM's docstring for the full table.
# Pre-2026-08-09 checkpoints (action_lstm.pth) are "mean".
POOL_MODE = "last"
# Opponent-aware checkpoint: agg is [own(6) | opponent(6) | present(1)] = 13 rather
# than 6, because each fencer's action is a response to the other's. Measured +2.9
# pts mean over three held-out bouts. Unlike POOL_MODE a mismatch here is LOUD --
# the head's first Linear changes shape, so load_state_dict raises.
USE_OPPONENT = True
FRAME_MODEL_PATH = PROJECT_ROOT / "models" / "action_frame.pth"  # --frame-model
BLADE_WEIGHTS = (PROJECT_ROOT / "models" / "blade_yolo" / "fencing_blade_v2"
                 / "weights" / "best.pt")

PREDICT_EVERY = 5     # frames between action predictions (labels hold in between)
MIN_REAL_FRAMES = 15  # don't guess an action until we've tracked this many frames

# multi-scale windows: a parry lasts ~12 frames, so 2s of surrounding footwork
# buries it in a single 60-frame window (measured: parry recall 3x better at
# short windows). Fast actions get judged on the short window, sustained ones
# (advance/retreat rhythm) on the long one.
WINDOW_LONG = SEQ_LEN   # 60 frames / 2 s
WINDOW_SHORT = 25       # ~0.8 s
MIN_REAL_SHORT = 12
FAST_CLASSES = {"parry"}   # brief actions the short window is allowed to override with
FAST_CONF = 0.65           # a short-window override has to be at least this sure
# quiet states: not fencing actions, so the overlay stays blank on them (this is why
# neutral/walking exist -- labels only light up when something real happens)
QUIET_CLASSES = {"neutral", "walking"}
ACTION_CONF_FLOOR = 0.50   # below this the call is too unsure to show as an action
# How often each class actually occurs in continuous footage, measured from
# data/labels/bout1_intervals.csv (787 labelled windows). The model is trained with
# inverse-frequency class weighting, which drives its effective prior to UNIFORM
# (16.7% each) -- but real fencing is nothing like uniform, and the clip corpus
# over-represents the exciting classes because clips were cut OF actions
# (lunge 10% of training windows vs 2.6% of reality, parry 10% vs 1.0%).
# Untreated, `lunge` stops behaving like a class and becomes a default: 21 true
# windows, predicted 378 times, swallowing 122 of 151 real advances.
# Correcting for it is standard label-shift: p(c|x) * target_prior / train_prior.
# Validated by fitting the prior on one half of the labelled footage and scoring
# the other (both directions): 15.7%->42.6% and 22.6%->41.2%. Per class, advance
# goes 5%->67% recall.
#
# RETIRED (2026-08-09) along with the clips-only checkpoint. All of the above was a
# patch for a training set whose class mix was an artifact of how clips were cut.
# action_cont.pth trains on continuous windows at their NATURAL frequencies, so its
# prior is already correct and multiplying CLASS_PRIOR in again would correct
# twice. Measured: inverse-frequency + post-hoc prior and natural-frequency + no
# prior both score 60.4% leave-one-bout-out, so this buys nothing and costs a
# hand-tuned constant that had to be re-estimated per venue.
#
# Set True again ONLY if MODEL_PATH is pointed back at a clips-only checkpoint.
APPLY_CLASS_PRIOR = False
# Pooled duration shares from BOTH labelled bouts (218 s). Re-derive with
# scripts/estimate_class_prior.py whenever more footage is labelled.
CLASS_PRIOR = {"advance": 0.184, "lunge": 0.045, "parry": 0.017,
               "retreat": 0.122, "neutral": 0.230, "walking": 0.401}
# Transfer is ASYMMETRIC and the prior is NOT universal. Measured:
#            bout1   bout2
#   uniform  19.2%   46.1%     bout2 needs little correction -- it is action-dense
#   bout1     50.2%*  43.3%    (*circular) transfers forward fine
#   bout2     24.4%   47.2%*   transfers BADLY backward: under-weights walking
#   pooled    45.9%   44.1%    most balanced, hence shipped
# Bout 1 is idle-heavy (walking 0.457) and bout 2 action-dense (walking 0.232), so
# a prior lifted from busy footage wrecks quiet footage. If you analyse material
# with very different action density, re-estimate. Unsupervised EM estimation does
# NOT work here (it put lunge at 0.635 vs a true 0.027) -- the prior has to come
# from labels.

# A parry is a RESPONSE: across bouts 3-5, 86% of labelled parries have the opponent
# attacking at the same moment (76% lunge+extension, 10% advance+extension). Requiring
# that at decision time roughly DOUBLES parry precision and lifts overall accuracy on
# two held-out bouts at two venues -- see _apply_parry_gate for the numbers.
PARRY_NEEDS_ATTACKER = True
PARRY_OPP_LUNGE_MIN = 0.20  # best-or-tied on both bouts; precision is flat 0.2-0.5 on
                            # bout 4 and still rising on bout 5, so this is the safe end

# The SAME co-occurrence run the other way. The veto above can only delete, so parry
# recall was capped at "how often parry wins the argmax" -- 15% on held-out bout 4.
# When parry LOST the argmax but the opponent is unmistakably lunging, promote it.
#
#   held-out bout 4   overall 72.2% -> 72.7%,  parry P 52% -> 58%,  R 15% -> 29%
#   held-out bout 5   overall 59.7% -> 59.7%,  parry P 27% -> 29%,  R 20% -> 24%
#
# Precision goes UP while recall doubles because the promoted windows are BETTER than
# the average existing parry call: 29 of 45 (64%) are true parries on bout 4.
#
# WHY IT WORKS, mechanically: 37 of those 45 were being called `retreat`. Parry-while-
# retreating is the ordinary case, the legs dominate the pose signal, and under the
# blade-priority collapse the truth is `parry` -- so the model was losing these to its
# own footwork call. The opponent's lunge is what breaks the tie. This is the same
# effect evaluate_labels.py documented from the other side ("22 retreat windows were
# called parry, and they cluster on real parries").
#
# CONTROLLED, because "promote when own parry >= P" is just a lower threshold and any
# lower threshold buys recall. Matched on the NUMBER of promotions, ranked by own parry
# probability with the opponent ignored: bout 4 gives 37% precision / 0.25 F1 against
# the opponent-conditioned 58% / 0.39. The opponent is doing the work, not the bar.
#
# HONEST WEAKNESS: bout 4 carries this. Bout 5 promoted 6 windows and bout 1 promoted 5
# -- too few to confirm or refute, and bout 1's five were 1/5 correct. Re-run
# scripts/sweep_parry_promote.py when the third venue lands. Its grid shows a broad
# plateau on bout 4 where LOWER thresholds do better still (0.10/0.30 -> 43% recall),
# deliberately not taken: bout 4 is the only bout that measures the effect, so tuning
# on it would be tuning on the confirmation set.
PARRY_PROMOTE = True
PARRY_PROMOTE_MIN = 0.15      # own parry probability, chosen on bout 5, confirmed on 4
PARRY_PROMOTE_OPP_MIN = 0.60  # opponent lunge -- far above the veto's 0.20, because
                              # admitting weaker own-evidence demands stronger context
# The blade lamp sits UNDER the footwork label rather than replacing it, in amber so it
# reads as a different kind of statement from the footwork call.
PARRY_LAMP_COLOR = (0, 165, 255)   # amber (BGR)
PARRY_LAMP_DY = 44                 # pixels below the footwork line

MAX_FROZEN_FRAC = 0.25  # skip the window if more than this share of joint steps are
                        # exactly zero, i.e. held over from the previous frame by the
                        # visibility fallback. Partial/occluded bodies were producing
                        # confident `advance` calls off frozen skeletons.
MIN_BOX_H_FRAC = 0.25 # ignore "people" shorter than this fraction of frame height.
                      # Was 0.35, which was tuned on bout 2's tight framing and silently
                      # amputated wider shots: on bout 2 the fencers run 0.30-0.60 so 0.35
                      # was harmless, but bout 1 frames them at 0.20-0.50 (median box 0.360)
                      # and 0.35 discarded 44% of REAL detections -- a slot dropped to 27%
                      # coverage and fencers went unlabelled for stretches. Calibrated, not
                      # guessed: bout 2's banner-graphic detections top out at 0.20 and its
                      # real fencers start at 0.30, so 0.25 sits in that empty gap -- every
                      # banner still rejected, 96% of bout 1's real boxes kept.
                      # (banner graphics of fencers trigger real YOLO detections --
                      # measured one at 0.30 of frame height, real fencers 0.4+)
PAN_STRIP_FRAC = 0.22
PAN_MIN_RESPONSE = 0.08
SLOT_COLORS = {"A": (0, 200, 255), "B": (255, 200, 0)}  # amber / cyan-blue (BGR)


class FencerTrack:
    """Sliding history for one fencer slot (A = left, B = right)."""

    def __init__(self) -> None:
        self.kp: deque[np.ndarray] = deque(maxlen=SEQ_LEN)      # (33, 4) frame fractions
        self.motion: deque[tuple[float, float]] = deque(maxlen=SEQ_LEN)  # (pan, hip_x)
        self.prev = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
        self.last_hip_x = 0.5
        self.label: str | None = None
        self.conf = 0.0
        self.probs: np.ndarray | None = None  # long-window distribution, for the parry gate
        # DISPLAY ONLY, set when `label` is parry: the footwork happening underneath it.
        # Deliberately separate from `label` -- evaluate_labels scores `label` against
        # blade-priority truth, so moving parry out of it would read as parry recall 0.
        self.footwork: str | None = None
        self.footwork_conf = 0.0
        self.counts: dict[str, int] = {}  # how often each label got emitted


def _frame_pan(prev_gray: np.ndarray | None, gray: np.ndarray,
               windows: dict) -> float:
    """Horizontal background shift between two frames, from L/R border strips."""
    if prev_gray is None:
        return 0.0
    h, w = gray.shape
    strip_w = max(10, int(PAN_STRIP_FRAC * w))
    rows = slice(int(0.10 * h), int(0.75 * h))  # skip broadcast graphics / scoreboard
    if "win" not in windows:
        windows["win"] = cv2.createHanningWindow((strip_w, rows.stop - rows.start), cv2.CV_32F)
    shifts = []
    for a, b in [(0, strip_w), (w - strip_w, w)]:
        (dx, _), response = cv2.phaseCorrelate(prev_gray[rows, a:b], gray[rows, a:b],
                                               windows["win"])
        if response > PAN_MIN_RESPONSE:
            shifts.append(dx)
    return float(np.median(shifts)) if shifts else 0.0


def _assign_boxes(dets: list[np.ndarray], tracks: dict[str, FencerTrack],
                  W: int) -> dict[str, np.ndarray | None]:
    """Match this frame's detections to the A/B tracks by continuity.

    Pure left/right slotting splits a fencer's history across two tracks the
    moment a second person enters or she drifts over the frame midline -- the
    action window resets and labels vanish. So: whoever has history keeps the
    box nearest their last hip position; left/right only breaks fresh starts."""
    slots: dict[str, np.ndarray | None] = {"A": None, "B": None}
    if not dets:
        return slots
    history = {s: tracks[s].last_hip_x
               for s, t in tracks.items() if any(np.any(k) for k in t.kp)}
    cxs = [float((b[0] + b[2]) / 2 / W) for b in dets]

    if len(dets) == 1:
        if history:
            slot = min(history, key=lambda s: abs(cxs[0] - history[s]))
        else:
            slot = "A" if cxs[0] < 0.5 else "B"
        slots[slot] = dets[0]
        return slots

    # Two detections: ALWAYS leftmost -> A, rightmost -> B. No history, no swap.
    #
    # FENCERS NEVER CROSS. On a piste the left fencer is left for the whole bout,
    # so the relative x-order of two boxes IS their identity -- more reliable than
    # any remembered position, and it cannot drift.
    #
    # This used to swap A/B when remembered hip positions preferred it, and that
    # was actively harmful: assignment feeds last_hip_x, which feeds the next
    # assignment, so one bad swap (a missed frame, a clinch, a silhouette in a
    # slot) PERSISTS. Measured on bout 4 -- of the 328 timestamps where the model
    # committed to a direction for both fencers, 47% were INVERTED PAIRS: correct
    # that the two oppose, wrong about which way. Those 155 windows form ~19
    # CONTIGUOUS RUNS, not scattered flicker, which is the signature of a sustained
    # mis-assignment rather than per-window noise. Slot A treats forward as
    # rightward and B as leftward, so a swap inverts advance and retreat -- the
    # single largest error in the system (556 advance/retreat confusions).
    #
    # The old docstring justified the swap by "drifting over the frame midline",
    # but that risk applies to a FIXED midline; this compares the two detections to
    # each other, so it never had the problem the swap was guarding against.
    order = np.argsort(cxs)
    slots["A"], slots["B"] = dets[order[0]], dets[order[1]]
    return slots


def _window_inputs(kp_seq: np.ndarray, motion_seq: np.ndarray, min_real: int):
    """(flat, agg, n_real) for one fencer's window, or None if it fails the gates.

    Split out of _classify_window so the OPPONENT's features come from exactly the
    same code path -- same gates, same normalisation, same engineered features. An
    opponent block built even slightly differently from an own block would be a
    train/serve mismatch hiding inside a single function.
    """
    real = np.any(kp_seq.reshape(len(kp_seq), -1) != 0, axis=1)
    if real.sum() < min_real:
        return None
    kp = kp_seq[np.argmax(real):]
    if len(kp) >= 2:
        step = np.linalg.norm(np.diff(kp[:, :, :2], axis=0), axis=2)
        if float(np.mean(step < 1e-9)) > MAX_FROZEN_FRAC:
            return None
    norm = _normalize_sequence(kp)
    agg = _engineered_features(norm, motion_seq)
    flat = norm.reshape(len(norm), -1).astype(np.float32)
    n_real = min(len(flat), SEQ_LEN)
    if len(flat) >= SEQ_LEN:
        flat = flat[:SEQ_LEN]
    else:
        flat = np.concatenate([flat, np.zeros((SEQ_LEN - len(flat), INPUT_SIZE), np.float32)])
    return flat, agg, n_real


def _classify_window(model: ActionLSTM, kp_seq: np.ndarray, motion_seq: np.ndarray,
                     min_real: int,
                     opp: tuple | None = None) -> tuple[str | None, float, np.ndarray | None]:
    """Classify one window of keypoints. Returns (label, confidence) or (None, 0).

    `opp` is the opponent's (kp_seq, motion_seq) for the SAME frames, or None. It is
    used only when USE_OPPONENT is on, i.e. when MODEL_PATH points at a 13-agg
    checkpoint. An opponent whose own window fails the gates counts as absent
    rather than as a stationary fencer -- see action_model.wide_agg.
    """
    # Gates live in _window_inputs now: too few real frames, or a skeleton mostly
    # carried forward. On the bout, 40% of `advance` calls came from windows at a
    # frame edge or >25% frozen, 14 of them more than HALF carried forward. Those
    # are not advances, they are missing data being labelled.
    got = _window_inputs(kp_seq, motion_seq, min_real)
    if got is None:
        return None, 0.0, None
    flat, agg, n_real = got

    if USE_OPPONENT:
        # The opponent goes through the SAME gates. If its window is unusable the
        # block is zeroed and the presence flag drops -- never a fabricated
        # stationary opponent, which would read as "they are holding still".
        opp_got = _window_inputs(*opp, min_real) if opp is not None else None
        agg = wide_agg(agg, None if opp_got is None else opp_got[1])

    # the short window is 25 real frames padded to 60, so 58% of it is zeros --
    # without the length the model pools across that padding and reads it as a cue
    lengths = torch.tensor([n_real])
    with torch.no_grad():
        logits = model(torch.from_numpy(flat)[None], torch.from_numpy(agg)[None], lengths)
        if logits.ndim == 3:
            # per-frame model: take the newest REAL frame, i.e. what is happening
            # now. Voting over the window would re-impose the single-label
            # assumption this model exists to avoid.
            logits = frame_logits_to_window(logits, lengths, mode="last")
        probs = torch.softmax(logits, dim=1)[0]

    if APPLY_CLASS_PRIOR:
        # train prior is uniform (inverse-frequency weighting), so dividing by it is
        # a constant and drops out of the renormalisation -- only the target matters
        w = torch.tensor([CLASS_PRIOR[c] for c in CLASS_NAMES], dtype=probs.dtype)
        probs = probs * w
        probs = probs / probs.sum().clamp(min=1e-12)

    idx = int(probs.argmax())
    # the full vector goes back too: the parry gate needs the OPPONENT's lunge
    # probability, not just their argmax label
    return CLASS_NAMES[idx], float(probs[idx]), probs.numpy()


def _predict(model: ActionLSTM, track: FencerTrack,
             opp_track: FencerTrack | None = None) -> None:
    """Multi-scale prediction: short window catches fast actions (parry), long
    window reads sustained ones. A confident fast-action hit on the short window
    overrides the long-window call; otherwise the long window decides.

    `opp_track` is the other fencer's track, needed by opponent-aware checkpoints
    (USE_OPPONENT). It defaults to None so older single-fencer callers keep
    working -- but against a 13-agg model that means every window reports "no
    opponent", which is a valid state yet throws away the whole point. Pass it.
    """
    kp_full = np.stack(track.kp)
    mot_full = np.array(track.motion, dtype=np.float32)
    if opp_track is not None and len(opp_track.kp) == len(track.kp):
        okp = np.stack(opp_track.kp)
        omot = np.array(opp_track.motion, dtype=np.float32)
    else:
        # length mismatch means the two tracks are not frame-aligned, and a
        # misaligned opponent is worse than none at all
        okp = omot = None

    def opp_slice(n):
        return None if okp is None else (okp[-n:], omot[-n:])

    long_label, long_conf, long_probs = _classify_window(
        model, kp_full[-WINDOW_LONG:], mot_full[-WINDOW_LONG:], MIN_REAL_FRAMES,
        opp_slice(WINDOW_LONG))
    short_label, short_conf, _ = _classify_window(
        model, kp_full[-WINDOW_SHORT:], mot_full[-WINDOW_SHORT:], MIN_REAL_SHORT,
        opp_slice(WINDOW_SHORT))
    track.probs = long_probs

    # The override may only ADD fast-class calls, never remove them, so it is the one
    # path that can manufacture parries. Guard: a short window may not overrule a
    # MORE confident long window (a 0.65 parry should not beat a 0.95 lunge). A real
    # parry still wins easily, since a 2 s window mostly filled by the parry is
    # exactly where the long call is weak.
    # Honest sizing, ORIGINAL (bout 1, clips-only checkpoint, measured by disabling the
    # path): 16 of fencer A's 68 parry calls and 5 of B's 22 — about a quarter.
    #
    # RE-MEASURED 2026-08-13 against `action_opp5`, because this was about to be deleted
    # as dead code. It is NOT dead, and the reasoning that said it was is worth keeping
    # as a warning: parry probability in the cached LONG-window vectors tops out at
    # 0.106, far under FAST_CONF, so the path looked unreachable. Wrong cache. The
    # override reads the SHORT window, where parry reaches 0.94 — a 0.6 s parry fills
    # most of a 0.8 s window and is diluted to nothing in a 2 s one. Same dilution that
    # sets BLADE_SPAN to 0.35 s and makes `last` pooling beat `mean`.
    #
    #   bout 1           0 fires / 1242 windows   (the bout the ORIGINAL sizing used)
    #   bout 5 300-400   2 fires / 1194 windows   both in unlabelled time
    #   bout 4  60-200   6 fires / 1674 windows   4 in labelled time, ALL FOUR true
    #                                             parries, every one retreat+parry
    #
    # So the path did not die, it MOVED: the checkpoint changed underneath it and bout 1
    # stopped exercising it. Checking one bout would have been misleading either way.
    #
    # NOT redundant with the parry promoter, which was the other reason to delete it. Of
    # those four true parries the promoter would catch only two: at t=87.75 s the
    # opponent's lunge probability is 0.001 against the promoter's 0.60 floor, so the
    # co-occurrence cue simply is not there and only the short window sees the parry.
    # The two mechanisms use different evidence — opponent context vs temporal
    # resolution — and neither subsumes the other.
    #
    # Parry over-prediction is NOT mainly this logic; it comes from the LONG window,
    # i.e. the model. The veto also cleans up after it: on bout 4 it killed exactly the
    # two fires that had no parry label and kept the four that did.
    if (short_label in FAST_CLASSES and short_conf >= FAST_CONF
            and (long_label is None or short_conf > long_conf)):
        track.label, track.conf = short_label, short_conf
    elif long_label is not None:
        track.label, track.conf = long_label, long_conf
    if track.label is not None:
        track.counts[track.label] = track.counts.get(track.label, 0) + 1


def _apply_parry_gate(tracks: dict[str, FencerTrack]) -> None:
    """Couple the two fencers' parry decisions to the OPPONENT's attack, both ways.

    Two rules over the same cue, see the constants above for the numbers:
      VETO     called parry, opponent not attacking      -> demote to the footwork
      PROMOTE  parry lost the argmax, opponent lunging   -> call parry after all

    They cannot fight each other. The veto fires below opponent-lunge 0.20 and the
    promoter demands 0.60, and promotion eligibility is `argmax != parry` rather than
    `label != parry`, so a window the veto just demoted is never handed back.
    (`track.counts` is incremented in _predict, i.e. BEFORE either rule runs, so the
    end-of-run label mix reports raw model calls -- that was already true of the veto.)

    Aaron's observation, and the labels back it hard. Across bouts 3-5, of 67
    labelled parries the opponent is simultaneously:

        lunge + extension    76%
        advance + extension  10%     -> 86% attacking in some form

    A parry is a RESPONSE; it essentially does not happen unless someone is coming
    at you. The reverse is much weaker (only 58% of lunges draw a parry), so this is
    used one-directionally: opponent-attacking gates parry, never the other way.

    Why a gate rather than a feature: the model already receives the opponent's six
    engineered features, and that is what lifted parry precision 9% -> 15%. What it
    does NOT receive is the opponent's predicted CLASS, because both fencers are
    classified independently. This couples them at decision time, which is the
    cheapest possible version of joint decoding.

    Measured on two held-out bouts at two different venues, offline on cached
    probabilities (see CLAUDE.md):

        bout 4   overall 67.4% -> 68.2%,  parry precision 18% -> 38%
        bout 5   overall 57.0% -> 58.7%,  parry precision 12% -> 27%

    Overall accuracy goes UP as well as parry precision -- the suppressed parries
    were mostly wrong, and the runner-up class is more often right. Threshold 0.2 is
    best-or-tied on both bouts, which is what a venue-independent rule looks like;
    contrast the fencing gate, whose best cue INVERTED between venues.

    Runs after BOTH tracks have predicted, so each fencer sees the other's
    distribution for the same frame.
    """
    lunge_i, parry_i = CLASS_NAMES.index("lunge"), CLASS_NAMES.index("parry")
    for slot, track in tracks.items():
        track.footwork, track.footwork_conf = None, 0.0
        if track.probs is None:
            continue
        opp = tracks.get("B" if slot == "A" else "A")
        # No opponent tracked at all -> no evidence of an attack. That is the
        # conservative reading in BOTH directions: it deletes a parry with nobody
        # visible to parry, and it never promotes one.
        opp_attack = 0.0 if opp is None or opp.probs is None else float(opp.probs[lunge_i])

        # The footwork underneath the parry, for the second indicator. Renormalised
        # over the five footwork classes -- "GIVEN this is not a parry, what are the
        # legs doing" -- because the raw runner-up sits below ACTION_CONF_FLOOR
        # almost always once parry has taken its share, and would render as `ready`
        # on nearly every parry. This is the two-head framing (5-way footwork + a
        # blade lamp) without a second head.
        alt = track.probs.copy()
        alt[parry_i] = -1.0
        fw_i = int(alt.argmax())
        rest = 1.0 - float(track.probs[parry_i])
        fw_name = CLASS_NAMES[fw_i]
        fw_conf = float(track.probs[fw_i] / rest) if rest > 1e-6 else 0.0

        if track.label == "parry":
            track.footwork, track.footwork_conf = fw_name, fw_conf
            if not PARRY_NEEDS_ATTACKER:
                continue
            if opp_attack >= PARRY_OPP_LUNGE_MIN:
                continue
            # gate rejected it: demote to the runner-up, and no lamp to light
            track.label, track.conf = fw_name, float(alt[fw_i])
            track.footwork, track.footwork_conf = None, 0.0
            continue

        # ---- the other direction: PROMOTE a parry that lost the argmax ----------
        if not PARRY_PROMOTE or int(track.probs.argmax()) == parry_i:
            continue
        if (float(track.probs[parry_i]) >= PARRY_PROMOTE_MIN
                and opp_attack >= PARRY_PROMOTE_OPP_MIN):
            track.footwork, track.footwork_conf = fw_name, fw_conf
            # Displayed confidence is the RAW parry probability, which understates the
            # call: promotions are 64% correct on bout 4 while reading 0.15-0.4. The
            # honest options were an uncalibrated combined score or the raw number,
            # and this project does not invent calibrations it has not measured.
            track.label, track.conf = "parry", float(track.probs[parry_i])


def _self_test_parry_gate() -> None:
    """The gate must fire only on parry, and only when the opponent is not attacking."""
    def mk(label, probs):
        t = FencerTrack()
        t.label, t.probs = label, np.array(probs, dtype=np.float32)
        t.conf = float(t.probs.max())
        return t

    def probs(**kw):
        v = np.full(len(CLASS_NAMES), 0.01, dtype=np.float32)
        for k, x in kw.items():
            v[CLASS_NAMES.index(k)] = x
        return v

    # opponent clearly lunging -> parry SURVIVES
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry", tr["A"].label

    # opponent NOT attacking -> parry demoted to its own runner-up
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.02))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label
    assert abs(tr["A"].conf - 0.3) < 1e-5

    # a NON-parry label is never touched, however quiet the opponent
    tr = {"A": mk("lunge", probs(lunge=0.7)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.0))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "lunge"

    # no opponent distribution at all -> treated as "nobody attacking", parry dropped
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)), "B": FencerTrack()}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    # exactly at threshold survives (>=, not >)
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.5, lunge=PARRY_OPP_LUNGE_MIN))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry"
    # ---- two-indicator display fields ----
    # A surviving parry must keep label == "parry" (evaluate_labels scores that against
    # blade-priority truth) AND expose the footwork underneath it for the second lamp.
    tr = {"A": mk("parry", probs(parry=0.5, retreat=0.3, walking=0.1)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry", "scoring label must not change"
    assert tr["A"].footwork == "retreat", tr["A"].footwork
    # renormalised over the non-parry mass: 0.3 / (1 - 0.5) = 0.6, NOT the raw 0.3.
    # Without this the footwork line sits under ACTION_CONF_FLOOR on nearly every
    # parry and the two-indicator display collapses back to one indicator.
    assert abs(tr["A"].footwork_conf - 0.6) < 1e-4, tr["A"].footwork_conf

    # a REJECTED parry lights no lamp and leaves no footwork field behind
    tr = {"A": mk("parry", probs(parry=0.5, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.0))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat" and tr["A"].footwork is None, vars(tr["A"])

    # ---- the promoter: the same cue run the other way ----
    # parry lost the argmax to retreat, but the opponent is clearly lunging -> promoted,
    # and the losing footwork becomes the second indicator
    tr = {"A": mk("retreat", probs(retreat=0.5, parry=0.25)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "parry", tr["A"].label
    assert tr["A"].footwork == "retreat", tr["A"].footwork
    assert abs(tr["A"].conf - 0.25) < 1e-5, tr["A"].conf

    # same window, quiet opponent -> untouched. This is the pair that separates the
    # promoter from "just a lower parry threshold"; the matched control that does the
    # same thing without the opponent scores 37% precision against 58%.
    tr = {"A": mk("retreat", probs(retreat=0.5, parry=0.25)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.02))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat" and tr["A"].footwork is None, vars(tr["A"])

    # own parry probability below PARRY_PROMOTE_MIN -> not promoted however hard the
    # opponent attacks. Context alone must never manufacture a parry.
    tr = {"A": mk("retreat", probs(retreat=0.7, parry=0.10)),
          "B": mk("lunge", probs(lunge=0.95))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    # THE INTERACTION THAT MUST NOT HAPPEN: a parry the veto demoted must stay demoted.
    # Here the opponent is quiet (0.0), so the veto fires; the promoter is checked on
    # argmax, not on the post-veto label, so it cannot hand the parry back. If this
    # ever flips, the veto's 29% -> 55% precision silently reverts.
    tr = {"A": mk("parry", probs(parry=0.6, retreat=0.3)),
          "B": mk("neutral", probs(neutral=0.9, lunge=0.0))}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    # no opponent distribution -> conservative in BOTH directions: never promotes
    tr = {"A": mk("retreat", probs(retreat=0.5, parry=0.25)), "B": FencerTrack()}
    _apply_parry_gate(tr)
    assert tr["A"].label == "retreat", tr["A"].label

    # a non-parry call never gets a lamp, and stale fields are cleared between frames
    tr = {"A": mk("parry", probs(parry=0.5, retreat=0.3)),
          "B": mk("lunge", probs(lunge=0.8))}
    _apply_parry_gate(tr)
    assert tr["A"].footwork == "retreat"
    tr["A"].label, tr["A"].probs = "advance", probs(advance=0.9)
    _apply_parry_gate(tr)
    assert tr["A"].footwork is None, "stale footwork survived into a non-parry frame"

    print("self-test ok: parry gate fires only on parry, only without an attacker; "
          "two-indicator fields set without touching the scoring label")


def _self_test_assign() -> None:
    """Slot assignment must be MEMORYLESS for two detections. Run with --self-test.

    Guards the single largest fix this project has had (bout 4: 43.6% -> 55.1%).
    `_assign_boxes` used to swap A/B when remembered hip positions preferred it,
    and because assignment writes the history that drives the next assignment, one
    bad swap persisted -- 47% of two-fencer direction calls came out INVERTED, in
    ~19 contiguous runs. Restoring any history-dependence here brings that back.
    """
    def box(cx, w=100):
        return np.array([cx - w / 2, 0, cx + w / 2, 300], dtype=np.float32)

    def cx_of(b, W=1000):
        return None if b is None else round(float((b[0] + b[2]) / 2 / W), 3)

    def tracks_at(pos):
        tr = {s: FencerTrack() for s in ("A", "B")}
        for s, hx in pos.items():
            tr[s].kp.append(np.ones((N_LANDMARKS, 4), dtype=np.float32))
            tr[s].last_hip_x = hx
        return tr

    dets = [box(250), box(750)]
    # the critical case: history says the fencers are the OTHER way round, which is
    # physically impossible on a piste and used to trigger a swap. x-order wins.
    out = _assign_boxes(dets, tracks_at({"A": 0.75, "B": 0.25}), 1000)
    assert (cx_of(out["A"]), cx_of(out["B"])) == (0.25, 0.75), "history overrode x-order"

    out = _assign_boxes(dets, tracks_at({"A": 0.25, "B": 0.75}), 1000)
    assert (cx_of(out["A"]), cx_of(out["B"])) == (0.25, 0.75), "agreeing history broke it"

    out = _assign_boxes(dets, {s: FencerTrack() for s in ("A", "B")}, 1000)
    assert (cx_of(out["A"]), cx_of(out["B"])) == (0.25, 0.75), "no history broke it"

    # a single detection still needs history -- x-order says nothing about WHICH
    # fencer is visible, so that path deliberately stays history-dependent
    out = _assign_boxes([box(700)], tracks_at({"A": 0.20, "B": 0.75}), 1000)
    assert out["B"] is not None and out["A"] is None, "lone box went to the wrong slot"

    print("self-test ok: two-box assignment is memoryless, lone box uses history")


def main() -> None:
    if "--self-test" in sys.argv:
        _self_test_assign()
        _self_test_parry_gate()
        return
    # --start/--end cut a segment WITHOUT re-encoding it first. The portfolio demo is
    # 60-90 s out of a 10-26 min source, and transcoding a clip through cv2 before
    # annotating it would throw away quality for no reason.
    def _opt(name, default=None):
        for i, a in enumerate(sys.argv):
            if a == name and i + 1 < len(sys.argv):
                return float(sys.argv[i + 1])
            if a.startswith(f"{name}="):
                return float(a.split("=", 1)[1])
        return default

    start_s, end_s = _opt("--start"), _opt("--end")
    _skip = {str(v) for v in (start_s, end_s) if v is not None}
    argv = [a for a in sys.argv[1:]
            if a != "--frame-model" and not a.startswith("--") and a not in _skip]
    use_frame = "--frame-model" in sys.argv
    if not argv:
        sys.exit("usage: python scripts/demo_video.py path/to/video.mp4 [out.mp4] "
                 "[--start S] [--end S] [--frame-model | --self-test]")
    video = Path(argv[0])
    if not video.exists():
        sys.exit(f"video not found: {video}")
    out_path = Path(argv[1]) if len(argv) > 1 else video.with_name(f"{video.stem}_demo.mp4")

    print("loading models...")
    person_model = load_person_model()
    if use_frame:
        # per-frame head: one call per frame, so a window can hold an advance AND
        # a lunge instead of being forced to pick one. Measured over 12 seeds:
        # bout advance 9.7% -> 14.3%, lunge 42.3% -> 31.0%, for ~3 pts of accuracy.
        if not FRAME_MODEL_PATH.exists():
            sys.exit(f"no per-frame checkpoint at {FRAME_MODEL_PATH} - train one first")
        action_model = load_action_model(FRAME_MODEL_PATH, device=torch.device("cpu"),
                                         cls=ActionFrameLSTM)
        print(f"using the per-frame model ({FRAME_MODEL_PATH.name})")
    else:
        # picks up the ensemble members if they are there, else the single checkpoint
        action_model = load_action_model(
            MODEL_PATH, device=torch.device("cpu"),
            cls=lambda: ActionLSTM(pool=POOL_MODE,
                                   n_agg=N_AGG_WIDE if USE_OPPONENT else 6))
    blade_model = load_blade_model(BLADE_WEIGHTS) if BLADE_WEIGHTS.exists() else None
    if blade_model is None:
        print("(no blade weights found, skipping the blade tip overlay)")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit(f"couldn't open {video}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Seek AFTER reading the properties. The tracks start empty either way, so the
    # first MIN_REAL_FRAMES frames of the segment are unlabelled while the window
    # fills -- start a little before the action you want to show.
    first = int(round((start_s or 0.0) * fps))
    last = int(round(end_s * fps)) if end_s is not None else total
    last = min(last, total)
    if first:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    n_frames = max(0, last - first)
    if start_s is not None or end_s is not None:
        print(f"segment {first / fps:.1f}s -> {last / fps:.1f}s "
              f"({n_frames} frames, {n_frames / fps:.1f}s)")
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    tracks = {"A": FencerTrack(), "B": FencerTrack()}
    prev_gray = None
    pan_windows: dict = {}

    with ExitStack() as stack:
        landmarkers = {
            slot: stack.enter_context(_make_landmarker(mp.tasks.vision.RunningMode.VIDEO))
            for slot in tracks
        }
        for idx in tqdm(range(n_frames), desc=video.stem[:40], unit="frame"):
            ok, frame = cap.read()
            if not ok:
                break

            gray = cv2.cvtColor(cv2.resize(frame, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            pan = _frame_pan(prev_gray, gray, pan_windows)
            prev_gray = gray

            box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=MIN_BOX_H_FRAC)
            boxes = _assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)
            timestamp = int(idx * 1000 / fps)

            drawn: dict[str, np.ndarray] = {}
            for slot, box in (("A", boxes["A"]), ("B", boxes["B"])):
                track = tracks[slot]
                kp = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
                crop = crop_box(frame, box)
                if crop is not None:
                    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    result = landmarkers[slot].detect_for_video(
                        
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp)
                    kp = _landmarks_to_array(result)
                    if np.any(kp):
                        # map crop coordinates back into full-frame fractions
                        x1, y1, x2, y2 = box
                        kp[:, 0] = (x1 + kp[:, 0] * (x2 - x1)) / W
                        kp[:, 1] = (y1 + kp[:, 1] * (y2 - y1)) / H
                        # carry occluded joints forward, same as training
                        low = kp[:, 3] < VISIBILITY_THRESHOLD

                        kp[low, :3] = track.prev[low, :3]
                        track.prev = kp.copy()
                        track.last_hip_x = float((kp[23, 0] + kp[24, 0]) / 2)
                
                track.kp.append(kp)
                track.motion.append((pan, track.last_hip_x))
                drawn[slot] = kp

            # PREDICT ONLY AFTER BOTH TRACKS HAVE THIS FRAME. Predicting inside the
            # per-slot loop would ask slot A about an opponent that is one frame
            # behind, and _predict's frame-alignment guard would then discard the
            # opponent on every A call -- silently reverting to the single-fencer
            # model for half the predictions.
            if idx % PREDICT_EVERY == 0:
                for slot in ("A", "B"):
                    track = tracks[slot]
                    if len(track.kp) >= MIN_REAL_FRAMES:
                        _predict(action_model, track, tracks["B" if slot == "A" else "A"])
                # both fencers now hold a distribution for THIS frame, which is what
                # the parry gate needs -- it asks whether the other one is attacking
                _apply_parry_gate(tracks)

            for slot, box in (("A", boxes["A"]), ("B", boxes["B"])):
                track = tracks[slot]
                kp = drawn[slot]
                # draw this fencer
                color = SLOT_COLORS[slot]
                if box is not None:
                    x1, y1, x2, y2 = box.astype(int)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                if np.any(kp):
                    pts = np.stack([kp[:, 0] * W, kp[:, 1] * H], axis=1)
                    draw_skeleton(frame, pts)
                org = (10, 40) if slot == "A" else (W - 360, 40)
                # TWO INDICATORS. Aaron: "what if we have two indicators, a footwork
                # and then a parry one, so that when a parry comes on, both can be
                # shown instead of one taking over the other." Footwork and blade are
                # near-orthogonal tracks -- a fencer parries WHILE retreating, and 86%
                # of labelled parries have the opponent attacking -- so a single label
                # has to throw one of them away. Only drawn when the gate agrees, which
                # is what makes the lamp worth believing: 29% precision ungated, 55%
                # with the veto, 58% now that the promoter also runs.
                #
                # The promoter is why this display earns its keep rather than merely
                # decorating. Its 45 promotions on bout 4 are overwhelmingly windows the
                # model called `retreat` and the labels call `parry` -- exactly the
                # both-at-once case a single label cannot show. Before the promoter the
                # lamp lit 61 times on bout 4; now it lights 106.
                if track.label == "parry" and track.footwork is not None:
                    quiet = (track.footwork in QUIET_CLASSES
                             or track.footwork_conf < ACTION_CONF_FLOOR)
                    if quiet:
                        draw_action_label(frame, f"{slot}: ready", None,
                                          org=org, color=(150, 150, 150))
                    else:
                        draw_action_label(frame, f"{slot}: {track.footwork}",
                                          track.footwork_conf, org=org, color=color)
                    draw_action_label(frame, "parry", track.conf,
                                      org=(org[0], org[1] + PARRY_LAMP_DY),
                                      color=PARRY_LAMP_COLOR)
                    continue
                is_action = (track.label is not None
                             and track.label not in QUIET_CLASSES
                             and track.conf >= ACTION_CONF_FLOOR)
                if is_action:
                    draw_action_label(frame, f"{slot}: {track.label}", track.conf,
                                      org=org, color=color)
                elif box is not None:
                    # tracked but not doing a scoring action -> a quiet "ready" tag
                    draw_action_label(frame, f"{slot}: ready", None, org=org, color=(150, 150, 150))

            if blade_model is not None:
                draw_blade_tip(frame, get_blade_tip(frame, blade_model))

            writer.write(frame)

    cap.release()
    writer.release()
    for slot, track in tracks.items():
        mix = ", ".join(f"{k} {v}" for k, v in sorted(track.counts.items(), key=lambda kv: -kv[1]))
        print(f"fencer {slot} label mix: {mix if mix else '(none)'}")
    print(f"\ndone -> {out_path}")


if __name__ == "__main__":
    main()
