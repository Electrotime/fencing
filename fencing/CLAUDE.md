# FenceVision — Claude Code Project Context

> **How to read this file.** The top third is current state and standing rules — read it every
> time. The rest is a dated findings log kept so that measurements, and especially *negative*
> results, are not re-run. Nothing in the log is deleted when it is superseded; it is marked
> **SUPERSEDED** with a pointer, because "we tried that and here is exactly how it failed" is
> the most expensive thing in the file to rebuild.

---

## CURRENT STATE (2026-08-13)

### Shipped demo configuration

All four constants live together at the top of `scripts/demo_video.py` and **must agree with
each other** — a checkpoint's pooling mode and feature width are part of its identity.

| constant | value | note |
|---|---|---|
| `MODEL_PATH` | `models/action_opp5.pth` | FIVE bouts incl. a second venue, continuous only |
| `POOL_MODE` | `"last"` | a wrong mode loads SILENTLY (identical parameter shapes) |
| `USE_OPPONENT` | `True` | 13 agg features; a wrong `n_agg` RAISES |
| `APPLY_CLASS_PRIOR` | `False` | retired 2026-08-09, no longer needed |
| `PARRY_NEEDS_ATTACKER` | `True` | no parry unless the opponent is attacking (86% of real parries) |
| `PARRY_OPP_LUNGE_MIN` | `0.20` | parry precision 29% → **55%** on held-out bout 4 |

### Where accuracy stands

Continuous broadcast footage, scored end-to-end through `evaluate_labels.py` on a bout the
checkpoint never trained on. Random is 16.7%.

| checkpoint | recipe | held-out bout 1 |
|---|---|---|
| `action_lstm` | clips only, mean pool, + class prior | 43.4% |
| `action_cont` | clips + continuous, last pool | 74.0% |
| `action_opp` | 4 bouts, last pool, **opponent** | 74.6% |
| **`action_opp5`** | **5 bouts — adds a second VENUE** | **76.4%** |

Held-out per class on `action_opp`: advance recall 53% → **80%**, retreat 96%, walking 95%,
neutral 39%, lunge 33%, **parry 12%**.

Leave-one-bout-out mean across four bouts: **70.3%** (was 42.9% on clips alone).

**ON A NEW VENUE IT WAS 58.1%** — `action_opp` scored on bout 5 while bout 5 was still unseen.
**That measurement is now SPENT**: bout 5 is in `action_opp5`'s training set, so it can never be
repeated, and no current checkpoint has an honest cross-venue number. 58.1% stands as the
historical figure for "four-bout model, unfamiliar venue"; **a third venue is what buys the
next one.** Until then, quote 76.4% as "held-out bout, familiar broadcast style" and be explicit
that cross-venue is unmeasured for the shipped model.

### What is open

1. **`parry` — precision is SOLVED (55%), recall is not (17%).** The parry gate fixed the false
   positives; what remains is that the model still misses ~5 of 6 real parries. Recall is now the
   whole problem, and the indicated route is a better blade FEATURE (blade motion energy, whose
   two known measurement flaws are untested — see its entry). More labels also buy recall, per the
   ablation below. Older framing, still relevant for the two-head option: See
   [two heads](#two-heads-aarons-two-indicator-framing-2026-08-09): parry precision scales with
   the number of two-track blade labels (189 labels → chance; 1329 → 2× base rate). Blocked by a
   catch-22: bout 4 is simultaneously the only adequate training set and the only adequate eval
   set for it. **Aaron's 10-minute label pass breaks that.**
2. **`neutral`** — 90% precision, 29% recall. Aaron: "neutral is sometimes called when someone
   does slower things or slows down." His definition is *continuous seconds of relative
   stillness*, so slow MOVEMENT reading as neutral is a genuine error, not a labelling
   disagreement.
3. **`lunge`** — 33% recall on held-out bout 1. Transient, same shape of problem as parry.
3b. **`advance` → `walking` at a new venue.** 282 of bout 5's 869 true advances, by far its
   largest error. These two classes are separated only by posture, via the crouch feature
   (knee angle ~140° fencing vs ~164° upright), and a different camera height shifts exactly
   that measurement. **The most concrete lead for what breaks off-venue, and it is one feature
   deep.** Bout 5 is now available to train on, which is the first thing to try.
4. **Phantom labels over broadcast filler.** 63% of the demo's predictions on bout 4 fall
   outside labelled fencing — the overlay confidently labels replays, crowd shots and graphics.
   Per-window scoring never sees this (unlabelled time is excluded) but it is most of what a
   viewer watches, and it is what holds the event timeline to 38% precision on a broadcast
   against 68% on a clean bout segment. The cheap geometric gate does **not** work; see
   [the gate](#is-this-fencing-gate-does-not-work-with-these-cues-not-shipped) and
   [the timeline](#event-timeline--bout-statistics-works-on-a-bout-not-on-a-broadcast-2026-08-12).

### Next steps

1. **Train on bout 5.** ✓ DONE 2026-08-13 — shipped as `action_opp5.pth`, +1.8/+3.7 on two
   held-out bouts. **A THIRD VENUE is now the highest-value labelling**: cross-venue accuracy
   is unmeasurable for the shipped model, and it is the number that most honestly describes
   the demo. Aim for parry-dense footage if there is a choice, since bout 5 diluted parry.
2. **Parry lamp** — measured 2026-08-13 and NOT shipped. The blade head now carries real signal
   (precision rises with threshold instead of sitting flat), but at a usable recall it is 15%
   precision on bout 4, so 85% of lit lamps would be wrong, and the good threshold differs per
   bout. Labels buy recall, opponent context buys precision; the next lever is a better blade
   FEATURE, not more of the same labels.
3. **Filler rejection**, which is now the blocker on the timeline being usable on full
   broadcasts rather than bout segments. Three attempts have failed: the geometric gate (36%
   precision), and duration + confidence gating, which cannot push filler below ~26% of emitted
   events. The one untested idea is the scoreboard/timer overlay region.
4. **Phase 5** (touch predictor) — the only unbuilt phase with a real design behind it.

### Standing workflow rule

**Stage changes and propose a commit message. Do not run `git commit`.** Aaron reviews the
staged diff and commits himself.

---

## READ THIS BEFORE TRUSTING ANY METRIC HERE

This is the most important section in the file. Nine "improvements" have been measured,
believed, and retracted. They come in exactly two shapes.

### Shape 1 — measuring how the CLIPS WERE CUT, not fencing

Training clips are hand-trimmed single actions; inference is a sliding window over continuous
video. Anything correlating with clip construction validates beautifully and then evaporates.

1. **Zero padding.** Clip length tracks class (lunge/parry 24f → 60% zeros, advance 46f,
   retreat 48f, sliced neutral/walking 0%) and the head mean-pooled across it. Re-padding by
   holding the last frame dropped in-sample accuracy 85% → 53% and flipped 40% of calls. Video
   never pads. **FIXED by masked pooling.**
2. **Bout label MIX is a distribution, not an accuracy.** With no ground truth, a mix drifting
   toward what a bout "should" look like proves nothing. A reported "advance 28 → 62" gain was
   false positives: advance firing on a fencer half off-screen (hip_x=0.00) and on
   stoppage-walking. Aaron caught it by eye; the metric could not.
3. **Clip START alignment** (`_first_mover`, 2026-07-31). Leg-order scored advance recall
   70% → 88% on held-out clips, shipped, then collapsed advance to 5 calls/fencer on video.
   Training clips begin at the action; sliding windows begin mid-stride. Measured: advance clips
   lean front-first 26%/17%, bout windows 26%/24% — a coin flip. **REVERTED.**

### Shape 2 — one bout talking

A single bout's headline number, a plausible mechanism written around it, and a contradicting
column sitting in the same table.

4. **Blade p99 energy.** 0.70 AUC on bout 2 (n=4 parries); a mechanism was written into this
   file for why p99 beats the mean. Bout 1 gave 0.49, pooled 0.55 vs the mean's 0.59.
   *Confirmed 2026-08-13 at n=70: p99 0.66, mean 0.79. The retraction was right.*
   **But the accompanying conclusion that blade energy DOESN'T WORK was itself wrong** — at n=6
   nothing was measurable; the mean scores 0.79 at n=70. A null from an underpowered sample is
   not a null, and this file said so at the time and then closed the idea anyway.
5. **Per-frame model routing.** Lunge recall 9% → 78% on bout 3. Checked on all three: recall
   replicated (81/74/78%) but precision was 3/29/26% and it fired lunge on 60% of bout 1's
   windows against a 2% true share. A saturated class, not a detector.
6. **"11 of 11 parries have retreat footwork"** (bout 3). Over bout 4's 46 parries it is 74%
   retreat, 7 neutral, 5 advance.
7. **Prior transfer "generalises."** Measured one direction only. The reverse direction
   (bout2-prior → bout 1) scores 24.4%.
8. **"Parry precision scales with blade supervision"** (2026-08-13). 9% on held-out bout 4 with
   189 labels, 28% on held-out bout 3 with 1329 — read as a scaling law and used to justify more
   parry labelling. But those are DIFFERENT BOUTS with different base rates (5.4% vs 13.8%), so
   the lifts are 1.7× and 2.0× — essentially the same. A same-bout ablation (`--blade-frac`) put
   precision at 14% with 189 labels and 15% with 1168. **Precision never scaled; recall did.**
   Comparing a rate across groups with different base rates is not a comparison.
9. **Timeline smoothing "is a clean null"** (2026-08-12). Measured on bout 1 across three
   metrics — precision, recall and count error — all flat or worse, and written up as closed. On
   bout 4 it lifts precision 38→47% and cuts count error 158→113. Same for the confidence gate:
   flat on bout 1, the strongest axis on bout 4. Three metrics agreeing on one bout is still one
   bout. Caught only because bout 4's run was already queued.

### Practical rules, earned the hard way

- **Precision next to recall, always.** Recall alone proves nothing; #5 above is the canonical
  case and it was the fifth wrong call of that day.
- **Check on every labelled bout before believing anything.** Behaviour varies enormously
  between bouts; any single-bout number is a sample of one.
- **When a change moves COVERAGE, compare absolute correct calls, not accuracy.** Scoring more
  windows means scoring harder windows, so accuracy can fall while the change is a clear win.
  This made the silhouette filter look like a regression twice.
- **Validation is LEAVE-ONE-BOUT-OUT, never a random split.** Windows are emitted every 5 frames
  from a 60-frame span, so neighbours share 92% of their frames and a random split reports a
  fantasy.
- **Every new feature needs a SHUFFLED-FEATURE CONTROL.** A new input also widens the head, so
  part of any gain is capacity rather than information. Permute the column within each bout and
  re-run: identical distribution, identical parameter count, no alignment. Blade energy read
  "+1.6 pts mean, positive on both held-out bouts" and the control reproduced the whole effect
  on one of them. One extra run; it is the difference between a feature and a null.
- **An AUC is only valid for the UNIT it was measured on.** Blade/torso scores 0.79 on hand-cut
  intervals and 0.53 on the 2 s windows the model actually consumes. Same data, same feature,
  different question. Check the unit before quoting a separability number.
- **Single-feature AUC shows a signal EXISTS, not that the model LACKS it.** `stance_ratio` had
  the best AUC of any feature ever tried here (0.91) and still made the model worse. Never add a
  feature on AUC alone.
- **Never reimplement the demo loop for a diagnostic** — reuse `demo_video`'s own functions. A
  reimplementation stubbed camera pan to zero once and inverted a conclusion.
- **A constant tuned on one video silently breaks another.** Check every threshold against all
  labelled bouts.
- Bout-mix numbers are noisy: two runs of the IDENTICAL config gave advance 8% vs 14%, lunge
  44% vs 32% on pure training noise. Do not chase small deltas.

---

## What this project is

FenceVision is a computer vision and machine learning system for analyzing fencing video. It is
a high school portfolio project targeting college admissions. It must be technically impressive,
well-structured, and produce a visually compelling real-time demo.

Built in Python inside VSCode using Claude Code. The developer fences and has some programming
experience but is not an ML/CV expert. Explain reasoning clearly. Prefer readable, well-commented
code over terse cleverness. When multiple approaches exist, explain the tradeoff and recommend
one.

When you are unsure whether something will work (library version, hardware capability, API
change), say so explicitly rather than guessing.

When the user says "let's start" or "let's do Phase X": state which phase, list the files you
will create or modify, write the code, then give the exact command to test it.

---

## System architecture (already decided — do not redesign)

```
Raw video
    ├── MediaPipe Pose Estimation → keypoint sequences (.npy)
    └── YOLO Blade Detector       → blade tip trajectory

Keypoint sequences
    └── Action Recognition LSTM  → action class probabilities

[Keypoints + Blade tip + Action probs] → Touch Predictor → who scores?

All streams → OpenCV + Streamlit overlay → real-time demo
```

- **MediaPipe** over OpenPose: real time on a laptop CPU, no GPU, Python-native
- **YOLO** for blade: pretrained backbone means ~300-500 labeled frames, not thousands
- **LSTM** over Transformer: simpler to implement and debug with a small dataset
- **Streamlit** for UI: fast to build, professional-looking, no frontend experience needed
- **Roboflow** for blade labeling: free tier, exports YOLO-ready dataset with YAML config

---

## Project file structure

Maintain this structure. Do not reorganize it.

```
fencing/
├── CLAUDE.md                       ← this file
├── requirements.txt
├── data/
│   ├── raw_video/                  1.mp4 … 4.mp4 — GITIGNORED (local/OneDrive)
│   ├── clips/<class>/              hand-cut action clips — video GITIGNORED
│   ├── keypoints/<class>/          <clip>.npy + <clip>.pan.npy — TRACKED in git
│   ├── labels/                     interval CSVs + cached window probs (*.npz)
│   ├── train_continuous/           extracted continuous windows — GITIGNORED
│   ├── blade_frames/               frames sampled for Roboflow labeling
│   ├── blade_dataset/              downloaded back from Roboflow
│   └── diagnostics/                annotated debug frames — GITIGNORED
├── models/
│   ├── action_opp5.pth (+ .m0–.m4) ← SHIPPED: FIVE bouts (two venues), last pool, opponent
│   ├── action_opp.pth  (+ .m0–.m4)    four bouts, one venue — the 58.1% cross-venue model
│   ├── action_cont.pth (+ .m0–.m4)    clips + continuous, last pool
│   ├── action_lstm.pth (+ .m0–.m4)    clips only, MEAN pool — historical, kept for comparison
│   ├── action_frame.pth               per-frame model — DEGENERATE, do not route to it
│   ├── verify_*.pth                   leave-one-out verification — GITIGNORED
│   ├── pose_landmarker_full.task
│   └── blade_yolo/fencing_blade_v2/weights/best.pt
├── notebooks/exploration.ipynb
├── src/
│   ├── action_model.py             Dataset + ActionLSTM + wide_agg + training loop
│   ├── pose_pipeline.py            MediaPipe extraction (frame / video / video+pan)
│   ├── person_detector.py          YOLOv8n person boxes, fencer A/B slots, silhouette filter
│   ├── blade_detector.py           blade YOLO inference + tip extraction
│   ├── labels.py                   interval-CSV parser + two-track collapse (ONE copy)
│   ├── train_action.py             thin wrapper that plots the loss curve
│   └── utils.py                    shared helpers (normalization, visualization)
└── scripts/                        see the grouped list below
```

**Not built yet:** `app.py` (Phase 6 Streamlit entry point) and `src/touch_predictor.py`
(Phase 5). An earlier version of this section listed both as if they existed.

`data/clips/extension/` and `data/keypoints/extension/` hold only a `.gitkeep` placeholder —
extension was dropped as a class. They are not empty, so deleting them drops a tracked file.
Harmless to leave.

### scripts/, grouped by what they are for

| group | scripts |
|---|---|
| **demo** | `demo_video.py` — the shipped inference loop; `--self-test` checks slot assignment |
| **data pipeline** | `process_clips.py`, `extract_continuous.py`, `extract_blade_frames.py` |
| **training** | `train_shipping.py` (the one that ships: `--ship`, `--holdout N`, `--pool`, `--opponent`), `train_continuous.py` (leave-one-bout-out), `learning_curve.py` |
| **evaluation** | `evaluate_labels.py` (`--model`, `--pool`, `--no-prior`, `--tag`, `--frame-model`), `estimate_class_prior.py` |
| **reporting** | `bout_timeline.py` — event timeline + per-fencer statistics from a probability cache; `--sweep`, `--csv`, `--self-test` |
| **label tooling** | `check_labels.py` (validates BOTH schemas, catches overlaps), `upgrade_labels.py`, `draft_labels.py`, `transcribe_bout4.py` |
| **closed experiments** | `exp_pooling.py`, `exp_two_head.py`, `exp_opponent.py`, `exp_window.py`, `exp_mirror.py`, `calibrate_gate.py`, `blade_energy.py`, `analyze_blade_energy.py`, `inspect_detections.py` |
| **misc** | `auto_clip.py` (experimental clip proposer — review output by hand), `smoke_test.py` |

Run either evaluation: `py -3 scripts/evaluate_labels.py <video.mp4> <labels.csv>`

---

## Coding conventions

- Python 3.10+ — `match/case`, `|` union types, `X | None` not `Optional[X]`
- Type hints on every function signature
- Docstrings on every class and public function (one-line summary + Args/Returns if non-obvious)
- No magic numbers — constants at the top of each file
- `pathlib.Path` for all file paths, not `os.path`
- Prefer `numpy` vectorized operations over Python loops for array math
- `tqdm` for loops over files or frames
- At least one `assert`-based smoke test at the bottom of each `src/` file under
  `if __name__ == "__main__":`

---

## Key technical facts (do not contradict these)

### MediaPipe landmarks (0-indexed)

- 0: nose, 11/12: shoulders, 13/14: elbows, 15/16: wrists
- 23/24: hips, 25/26: knees, 27/28: ankles
- **Landmarks 1-10 (face)** are physically hidden by the fencing mask, but MediaPipe places them
  with visibility ~1.0 anyway (verified on sample frames; it infers them from head shape). They
  are NOT zeroed by the carry-forward logic. Keep them in the tensor (INPUT_SIZE stays 132) —
  they track the head and are harmless — but never build handcrafted features from them.

### Fencing-specific challenges

- Don't rely on visibility to filter mask-hidden landmarks; see above.
- The sword arm (usually right for right-handed fencers) is the primary bladework signal.
- Footwork signal: relative distance and velocity between landmarks 27 and 28 (ankles).
- Fast actions (fleche, ballestra) cause motion blur. If visibility drops below 0.3 on >50% of
  landmarks, flag the frame as low quality.
- When two fencers overlap, MediaPipe may assign keypoints to the wrong person — hence the
  per-fencer crop before pose estimation.

### Action classes — SIX, and the list is LOCKED (Phase 5 input width depends on it)

- **advance** — step forward, FRONT foot then rear. Ankle gap widens then closes quickly.
- **retreat** — step backward, REAR foot then front. The mirror.
- **lunge** — explosive forward extension of sword arm and front leg; front knee drives to ~103°
  (vs ~137° in an advance), body lowers, ankle-gap/leg-length ~1.9 (vs ~1.0). Usually ends a
  series of advances, though it can stand alone.
- **parry** — defensive blade deflection, small fast wrist motion. The hardest class; see the
  parry thread in the findings log.
- **walking** — normal upright walking between phrases; legs extended, ankles close. Separate
  from neutral because it translates forward and would otherwise fire `advance`.
- **neutral** — continuous seconds of relative stillness (en-garde, bounces, pauses).

Not classes: **extension** lives on as the arm-reach FEATURE (every lunge contains one).
fleche / riposte / en-garde are out of scope — riposte and priority are the RULE layer in
Phase 5, not learned classes.

The footwork descriptions are mechanism, not decoration: they are the most productive source of
features found so far. Anything derived from them must be checked for **window-position
invariance** before shipping (see `_first_mover` in the metric warning above).

### THE CLASSES ARE NOT MUTUALLY EXCLUSIVE — the two-track schema

They are two near-orthogonal tracks:

- **FOOTWORK (legs):** advance / retreat / lunge / walking / neutral
- **BLADE (arm):** parry / extension / none

A fencer parries WHILE retreating (74% of parries across bout 4's 46). This is why parry has been
the worst class since the start and why no feature or architecture fix ever moved it. **Do not
try to suppress parry; label and predict the two tracks separately.** It also lines up with
Phase 5, which already needs blade action and footwork independently for priority.

**Two-track label format** — `fencer,start,end,footwork,blade`, one row per fencer per interval.
The exact vocabulary is enforced by `scripts/check_labels.py`; run it on any new label file
before using it.

| column | allowed values |
|---|---|
| `fencer` | `left`, `right` — **not** A/B |
| `footwork` | `advance`, `retreat`, `lunge`, `walking`, `neutral` (the six classes minus `parry`) |
| `blade` | `parry`, `extension`, `none`, `attack`, `beat` |
| either track | `TODO` where it genuinely cannot be inferred — write this rather than guessing |

`check_labels.py` validates BOTH schemas (`parry` stays legal as a single-track *label*, which is
the whole point of the upgrade) and catches **overlapping intervals**, which otherwise mis-score
silently because `truth_at()` returns the first match. `upgrade_labels.py` lifted bouts 1-2
mechanically, writing `TODO` on the 27 and 32 rows it could not infer.

**Collapsing two tracks to one for scoring defers to the blade track only when the blade label is
emittable** (`blade in CLASS_NAMES`). A naive blade-priority collapse would have deleted 10 of
bout 3's 14 lunges, relabelling them `extension` or UNSCORABLE.

### YOLO model sizes

- `yolov8n.pt` — fastest; use for real-time inference and fine-tuning on small datasets
- `yolov8s.pt` — slightly more accurate, still fast; use if nano is insufficient
- Do not use medium/large/xlarge — overkill for this dataset size

---

## Phases

### Phase 0 ✓ BUILT — environment

Deps in `requirements.txt`; verify with `scripts/smoke_test.py`. Runs on Python 3.14.

### Phase 1 ✓ BUILT — data collection

`scripts/extract_blade_frames.py` samples ~400 frames from `data/raw_video/` into
`data/blade_frames/` as `{video_stem}_f{frame_index:06d}.jpg`. Skips an interlaced original when
a `_deinterlaced` twin exists.

Worth keeping: sampling a fixed BUDGET beats every-Nth-frame. Every-5th on full matches gives
~9,300 frames — far past the 300-500 needed, and highly redundant since a blade barely moves
between frames at 6 fps. Target-count sampling hits the labeling budget and maximises pose
diversity.

### Phase 2 ✓ BUILT — pose estimation

`src/pose_pipeline.py` (extract from frame / from video, plus
`extract_keypoints_and_pan_from_video`), `src/person_detector.py`, `scripts/process_clips.py`
(writes `data/keypoints/<action>/<clip>.npy`).

MediaPipe Pose detects only ONE person per frame and fencing video has two, so
`person_detector.py` crops each fencer first using pretrained YOLOv8n (COCO class 0);
`YOLO("yolov8n.pt")` auto-downloads, no training needed.

**Train/serve note, checked and NOT a problem — do not "fix" it.** `process_clips.py` runs pose
on the FULL FRAME while the demo runs it on a tight person crop. Cropping training clips
actually RAISES jitter (0.0122 → 0.0158) rather than matching the demo's 0.008, and subject
sizes are comparable.

**MediaPipe config (Tasks API — `mp.solutions` was removed in 0.10.14+).**
`download_pose_model()` fetches `pose_landmarker_full.task` (~27 MB, equivalent to the old
`model_complexity=1`).

```python
options = mp.tasks.vision.PoseLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="models/pose_landmarker_full.task"),
    running_mode=mp.tasks.vision.RunningMode.IMAGE,   # or VIDEO for video streams
    num_poses=1,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
```

`RunningMode.IMAGE` for `extract_keypoints_from_frame` (no temporal context);
`RunningMode.VIDEO` for `extract_keypoints_from_video`, passing
`timestamp_ms = int(frame_index * 1000 / fps)` to each `detect_for_video()`.

**Post-processing rules (video path — all measured-and-revised 2026-07):**

1. Any keypoint with visibility < 0.5 → replace x,y,z with the previous frame's values
   (carry-forward). First frame uses zeros.
2. 3-frame median filter on x,y over the clip — the tracker occasionally teleports a joint for
   one frame (measured spikes up to ~100× body size); the median kills those without smearing
   real motion.
3. Normalize x,y relative to the hip midpoint (landmarks 23/24 average). After normalization the
   hip midpoint is (0, 0).
4. Scale by the per-clip **MEDIAN body height** (shoulder-mid to ankle-mid). History: spec said
   per-frame shoulder width → collapsed to ~0 for side-on fencers (~300× blowups); then
   per-frame torso length → foreshortens during lunges and crushed the leg-spread signal. A
   single per-clip height is stable through every action.
5. Do NOT normalize z or visibility — keep them raw.
6. The video extractor also returns a per-frame **motion track** of shape (n, 2) = [background
   pan (phase correlation on border strips), raw hip-x before centering] via
   `extract_keypoints_and_pan_from_video`. World travel = in-frame hip-x + camera pan; this is
   what makes advance vs retreat learnable (pan alone works only when the camera tracks tightly;
   the hip-x term catches looser broadcasts). `process_clips.py` saves it as `<clip>.pan.npy`.

Re-run `scripts/process_clips.py --force` after any normalization change.

**Verification:** draw the MediaPipe skeleton back onto a sample frame with
`mp.solutions.drawing_utils` to confirm keypoints look correct.

### Phase 3 ✓ BUILT — blade detection

Trained YOLO11n at `models/blade_yolo/fencing_blade_v2/weights/best.pt`; val P 0.79 / R 0.74 /
mAP50 0.74. Loaded by `src/blade_detector.py`, which exposes:

```python
def load_blade_model(weights_path: str) -> YOLO: ...

def get_blade_tip(frame: np.ndarray, model: YOLO) -> tuple[float, float] | None:
    """Centroid of the highest-confidence blade detection, or None.
    Tip = box centroid: ((x1+x2)/2, (y1+y2)/2). An approximation — the true tip is at one
    end, but without knowing which end faces the opponent the centroid is the least-wrong
    single point. A keypoint model trained to locate the tip is the future improvement."""

def get_blade_tip_trajectory(video_path: str, model: YOLO) -> list[tuple | None]: ...

def compute_tip_velocity(trajectory: list) -> list[tuple[float, float]]:
    """(dx, dy) per frame; None frames get (0.0, 0.0)."""
```

Training command (run from the terminal, no separate file):

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(data="data/blade_dataset/data.yaml", epochs=50, imgsz=640, batch=16,
            project="models/blade_yolo", name="fencing_blade_v1", patience=10)
```

Roboflow labeling recipe: Object Detection project → upload `data/blade_frames/` → box the
ENTIRE blade (not just the tip), class `blade` → auto-augment (flip, rotation, brightness) →
export YOLOv8 format → unzip into `data/blade_dataset/`.

**Coverage caveat that matters downstream:** the detector fires on only **1-2% of tracked
fencer-frames**, because fast blades blur or vanish. See the blade-energy entry in the findings
log for the attempt to work around that.

### Phase 4 ✓ BUILT — action recognition

**`src/action_model.py` is the source of truth — read it, don't re-spec it here.** An earlier
version of this section carried a full spec that drifted badly out of date (4 classes,
HIDDEN_SIZE 64, 4 features, "2-layer LSTM"; reality is 6 classes, HIDDEN_SIZE 128, 6 features,
ONE LSTM layer, and 2-layer measured WORSE). A stale spec under a "do not contradict" heading is
worse than no spec.

Only the non-obvious decisions live here:

- **`wide_agg()` in `action_model.py` is the ONLY definition of the 13-vector layout**
  (`[own(6) | opponent(6) | present(1)]`). Training, offline scoring and the demo all import it.
  It is exactly the kind of thing that gets hand-copied into three files and then drifts.
- The 6 engineered features (`_engineered_features`): net-forward, stance-width p90,
  wrist-speed p90, total-travel, arm-reach p90 (= extension, drives Phase 5 priority), crouch.
- Engineered features beat keypoints alone by +23 pts (51% → 74%) — the LSTM cannot rediscover
  them from 132 channels at this dataset size. Combining hip-x with camera pan (vs pan alone)
  lifted advance/retreat direction accuracy 84% → 94%.
- HIDDEN_SIZE 64 → 128 was worth +1.5 pts once the set reached 488 windows; at ~80 clips the
  smaller net won. Re-tune capacity when the dataset changes size substantially.
- Long clips are trimmed from the END (keep the first SEQ_LEN frames) because an action's
  initiation is its most distinctive part. **Caveat:** this is also why `_first_mover` failed on
  video. Clip-start alignment is a training-set property, not a fencing property.
- Long neutral/walking clips are auto-sliced into overlapping 60-frame windows
  (`SLICEABLE_CLASSES` / `SLICE_STRIDE`); splits are GROUP-aware so a clip's windows never
  straddle train/val (`group_stratified_split`).
- **Masked pooling is not optional.** Clip length is class-correlated, so pooling over padding
  lets the model read "how much of this is zeros" as a class cue.
- After training, print `sklearn.metrics.classification_report`; flag any class under 70% recall.

**Clip corpus:** 213 clips / 488 windows — advance 35, retreat 33, lunge 48, parry 47,
neutral 23, walking 27. Clip-cutting rules learned the hard way: crop tight (fencer ≥ half frame
height), keep the action inside the first 2 s, cut every class with the same ~0.5 s lead-in /
0.3 s lead-out, label by the clip's dominant action. Banners with printed fencers can fool the
YOLO person detector (not MediaPipe pose), so eyeball auto-crops before trusting them.

### Phase 5 — touch predictor (NOT BUILT)

**`src/touch_predictor.py`**

**Feature vector.** For a window of recent frames:

| component | count |
|---|---|
| action class probabilities, 6 per fencer × 2 | 12 |
| blade tip velocity (dx, dy) per fencer | 4 |
| three joint angles (elbow, knee, hip) per fencer | 6 |
| **total** | **22** |

*(An earlier version of this spec said 18, computed from the obsolete 4-class list. The width
follows the LOCKED 6-class list — this is why that list is locked.)*

```python
def build_feature_vector(
    kp_a: np.ndarray,            # (SEQ_LEN, 33, 4)
    kp_b: np.ndarray,
    action_probs_a: np.ndarray,  # (6,)
    action_probs_b: np.ndarray,  # (6,)
    blade_velocity_a: tuple,     # (dx, dy)
    blade_velocity_b: tuple,
) -> np.ndarray:                 # (22,)
    """Assembles the input vector for the touch predictor."""

def compute_joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at joint B formed by A-B-C, in degrees. Each point is an (x, y) array."""
```

Landmark triples for the angles: elbow 11-13-15 (left) / 12-14-16 (right); knee 23-25-27 /
24-26-28; hip 11-23-25 / 12-24-26.

```python
class TouchPredictor(nn.Module):
    """Simple 3-layer MLP. Input (batch, 22) → output (batch, 3) logits for
    [fencer A scores, fencer B scores, no touch]; softmax.
    The "no touch" class is REQUIRED — in real-time inference this runs on every frame,
    not just touch windows, and without a neutral class it hallucinates a scorer constantly."""
```

**Label OUTCOMES, not actions.** Action labeling is done and feeds this stage. New effort is a
small scorer CSV per bout — `timestamp_start,timestamp_end,scorer` with scorer in {A,B,N}. At
train time the action model runs on each window and its class probs become INPUT features; the
label is who scored. Aim for roughly equal numbers of A, B and N windows.

```
timestamp_start,timestamp_end,scorer
00:00:05,00:00:07,A
00:00:15,00:00:17,B
00:00:09,00:00:11,N
```

**Priority is a RULE layer over the action streams, not a learned model** — extension
establishes attack; parry transfers priority to the defender → riposte; simultaneous → no touch.
No extra labels, and more explainable. Awarded touch = blade contact + priority. Point-in-line =
sustained high arm-reach + no travel.

### Phase 6 — Streamlit overlay (NOT BUILT)

**`app.py`** — the main demo application.

- Left column: live video with skeleton overlay + blade tip dot + action label
- Right column: current action label (large), confidence bar chart (`st.bar_chart`), touch
  probability gauge (A% vs B%), session stats (touches detected, most common action)
- Video source: webcam (`cv2.VideoCapture(0)`) or a file path via CLI arg / text input

Overlay drawing lives in `src/utils.py`: `draw_skeleton`, `draw_blade_tip`, `draw_action_label`
(all take a frame, return the annotated frame).

Target 15 fps minimum. If pose estimation is too slow, run it every 2nd frame and reuse the
previous result.

**Overlay conventions already implemented in `scripts/demo_video.py` (2026-07-23):**
`QUIET_CLASSES = {neutral, walking}` render as a grey "ready" tag rather than an action label,
and any call under `ACTION_CONF_FLOOR` (0.50) is suppressed the same way — so labels stay silent
through idle/repositioning and light up only on real actions. Verified on clips (walking →
"ready", parry → amber "parry 83%") and on a match segment.

The demo also runs **multi-scale windows**: a 25-frame short window for fast actions and a
60-frame long window for sustained ones, where a confident (≥ `FAST_CONF` = 0.65) fast-class hit
on the short window overrides the long-window label. **⚠ That path is effectively dead** — see
the FAST_CLASSES entry in the findings log; parry probability maxes at 0.106, roughly 6× below
the threshold.

### What "done" looks like

A 60-90 second video showing real footage in the Streamlit app, skeleton overlay on both
fencers, blade tip tracked, action label updating in real time, and a touch probability bar
developing through a phrase. That video is the portfolio artifact; everything else supports it.

---

# FINDINGS LOG

Chronological. Negative results are kept deliberately.

## The clip era (2026-07)

Measured on hand-cut clips only, before any ground truth on continuous video existed. The
headline of the day was **10-seed 85.1% ± 6% on held-out clips** (lunge 98 / walking 90 /
retreat 85 / neutral 80 / advance 65 / parry 65). **That number is not a generalisation
estimate** — the same checkpoint scored 19.0% on continuous footage. Read every clip-based
number in this file in that light.

What survives from that era, all still baked into the pipeline:

- The broadcast camera PANS to follow the fencer, so keypoints alone cannot tell advance from
  retreat — hip drift in-frame is near zero and points both ways. True travel is recovered from
  background pan (phase correlation) × facing direction (nose vs hips).
- Clips contain fencers facing both directions; without the pan × facing feature those classes
  are mirror-ambiguous.
- Per-frame scale references wobble with pose (shoulders collapse side-on, torso foreshortens in
  lunges); per-clip median body height is the stable choice.
- Tracker teleport spikes forged fake wide stances in advance clips until the 3-frame median
  filter; that plus a stance-width p90 feature is what makes lunge separable.
- Walking and advance are both "moving forward" — only posture splits them. The crouch feature
  (median knee angle: fencing ~140° vs upright ~164°, 84% separable) plus inverse-frequency class
  weighting rescued advance after the 6-class flip.
- Confusion (10-seed): advance leaks to parry/lunge/walking; parry leaks to retreat; the biggest
  raw error mass is neutral↔walking swaps, which are downstream-harmless (both mean "no priority
  action").

### Fencer-B retreat bug — RULED OUT, don't re-run (2026-07-24)

B read `retreat` on 134/237 windows through a stoppage (clock frozen at 2:34). NOT the
engineered features (ablating each to its neutral median leaves 134→148; pathway swap localises
it to the LSTM path). NOT `nose_dir` instability (4 flips/bout; B's net-forward correctly
balanced 82+/92−). NOT pose quality (A vs B identical on jitter, frozen joints, visibility). NOT
facing coverage (every class ≥11 clips on its thinner side). NOT fixable by mirroring (both
variants lose: 81.8% and 80.6% vs ~85-87%). There IS a modest out-of-sample facing gap — 91%
facing right vs 84% left — far too small to explain a 134:1 skew.

**Resolved 2026-08-04:** it was never the fencer. See the A/B asymmetry entry there.

### There is NO domain gap — do not re-derive this

An earlier claim that training clips were "textbook-wide" vs "compact" match footwork was wrong
twice over: Aaron's clips are themselves cut from broadcast matches (same 1920×1080 / 29.97 fps),
and the comparison averaged ALL bout windows when most are idle/stoppage while training clips are
pure action. Restricted to ACTIVE bout windows (top-quartile travel) they line up: stance
0.58 vs 0.54, crouch 0.55 vs 0.57, arm-reach 0.21 vs 0.21. Windows inside the advance band on all
six features go 19% → 40%.

**And the root cause was the decision boundary, not coverage.** Of 89 bout windows sitting inside
the training advance band on all six features, the model said lunge 55 (62%), retreat 30 (34%),
advance 4 (4%). Median probability gap to the winner is 0.341 — far too wide for threshold
tuning. Most likely the task was ill-posed at this window size: a 2 s window during an exchange
holds step-step-lunge-recover while the classes are defined on single-action clips, so mixed
windows fall to whichever class has the loosest boundary. That reading motivated both the
per-frame model (failed) and continuous training (worked).

### Shipped 2026-07-29 → 07-31

- **Masked pooling** (`ActionLSTM.forward(..., lengths)`) — pools real frames only, killing the
  padding artifact. val@hold 82.3% → 85.0%, padding-style gap 4.0 → 0.8 pts, costing 0.6 pts on
  the zero-padded score that was partly measuring the artifact. `lengths=None` still pools
  everything, so old callers work. The demo's 25-frame SHORT window (58% zeros) fed the artifact
  hardest, so `parry` shifted most.
- **`ActionEnsemble` + `load_action_model()`** — 5 members at `models/action_lstm.m*.pth`. For
  CONSISTENCY, not accuracy: single checkpoints at equal val accuracy land anywhere from
  advance=6%/lunge=50% to advance=25%/lunge=7% on the same bout. Measured 2.4× less
  out-of-domain variance (sd 4.1% → 1.7%). Averages **PROBABILITIES not logits** (independently
  trained members are not calibrated against each other). Falls back to the single checkpoint
  when members are absent.
- **`ActionFrameLSTM` / `--frame-model`** — one label per FRAME, so a window can hold an advance
  and a lunge instead of being forced to pick. No new annotation needed (each clip is one action,
  so every frame carries its label); 488 windows → ~20k supervised frames. 12 seeds: bout advance
  9.7% → 14.3%, lunge 42.3% → 31.0%, for ~3 pts of held-out accuracy. **Do NOT ensemble it** — 5
  members gave the best lunge figure anywhere (23%) but advance 16% → 8% and parry 5% → **0%**, a
  class gone. Averaging dilutes BRIEF actions' probability peaks, so persistent classes take
  every frame. **⚠ Superseded entirely 2026-08-08 — this model is degenerate, do not route to
  it.**
- **`MAX_FROZEN_FRAC` gate** (demo) — refuses to classify a window whose skeleton is >25%
  carried-forward. When a fencer leaves frame the visibility fallback holds joints in place and
  the model confidently labels missing data; this removed ALL 16 of fencer B's `advance` calls,
  every one a frozen skeleton.

### ⚠ SCOPE CORRECTION (2026-08-02) — everything above is BOUT-2-SPECIFIC

Everything measured before this date came from ONE 40 s segment of
`Bout #2 without break.mp4`. A second segment (`Bout #1 without break (1).mp4`, ~5:00 in)
behaves completely differently on the same checkpoints:

| | bout 2 | bout 1 |
|---|---|---|
| lunge share | **43%** | **13%** |
| fencer B advance | 9 | 29 (more than A) |
| fencer B neutral | 1 | 86 (its top label) |

So `lunge` over-prediction and the A/B asymmetry do NOT reproduce. Measured differences: bout 2
is a TIGHTER shot (fencer height 0.463 vs 0.391 of frame) with far more reliable two-fencer
detection (1.83 vs 1.19 detections/frame). **Do not conclude bout 1 is "correct"** — there is no
ground truth for it either, and a plausible histogram is exactly what produced the false advance
result before.

Bout-2-specific issues, recorded and then explained by later work:

- **`lunge` over-predicted (~42%)** — not a bad draw (5 members spanning 7-50% still average 42%)
  and not the features (active windows sit at wrist-speed 0.06 / reach 0.21 vs train lunge
  0.17 / 0.28), so within that footage it is the LSTM path. *Explained 2026-08-04: prior
  mismatch.*
- **`advance` not detected on bout 2** — after the frozen gate fencer B produces 0 and A's fire
  mostly on stoppage-walking; `walking` and `advance` are effectively swapped there.
- **A/B asymmetry** — A advance=26%/lunge=37% vs B advance=7%/lunge=62%, and no fix touched it.
  Bout 1 reverses which fencer looks better, so NOT positional and NOT slot assignment.
  Handedness ruled out: training is 61% right / 39% left, and bout 2's broken fencer B is
  RIGHT-handed, the majority class. *Solved 2026-08-04.*
- **No selection rule for the per-frame model.** Best-val picked a lunge-heavy checkpoint twice
  (window seed 8 → 52%, per-frame seed 7 → 49%, vs sweep averages 42%/31%). **Val accuracy looks
  mildly ANTI-correlated with video behaviour.** The ensemble sidesteps this for the window
  model. This is why `train_shipping.py` deliberately does no best-epoch selection.

### Demo framing bug, FIXED (2026-08-02)

`MIN_BOX_H_FRAC` was 0.35, tuned on bout 2 where fencers fill 0.30-0.60 of frame height. Bout 1
frames them at 0.20-0.50 (median box 0.360), so the filter discarded **44% of real detections**
and fencers went unlabelled for stretches (one slot down to 27% coverage). Now **0.25**,
calibrated not guessed: bout 2's banner-graphic detections top out at 0.20 and its real fencers
start at 0.30, leaving an empty gap. Coverage 90% → 99%.

### `parry` over-prediction is the MODEL, not the demo's fast-path (2026-08-02)

`parry` is the only FAST_CLASS, so a short 25-frame window scoring ≥0.65 can override the
60-frame call, and that path can only ADD parries. Measured by disabling it: it contributes 16 of
fencer A's 68 parry calls and 5 of B's 22 — about a quarter. The rest come from the long window.
A confidence guard (short may not overrule a more-confident long) is in place but removes only
~4 calls; short windows are systematically more confident than long ones, so it rarely triggers.
**Do not tune the demo for this** — the fix is upstream.

*Method note:* a first diagnostic claimed 87% of parries came via the override. It stubbed camera
pan to ZERO, which shifted long-window confidences and inverted the conclusion.

### `advance` vs `neutral` on small advances (open)

Aaron: fencer B's *small* advances read `neutral`. Consistent with the features — a small advance
keeps a narrow stance and shallow crouch, which is the neutral profile, so only world-motion
separates "small step" from "standing". Windows called neutral vs advance for B: stance
0.25 vs 0.56, crouch 0.44 vs 0.61. Needs labelled small-advance examples or a sharper
world-motion signal.

---

## FIRST REAL EVALUATION (2026-08-04) — ground truth exists

Aaron labelled `data/raw_video/1.mp4` (FIE Worlds Hong Kong, DOSA vs CHOUPENITCH, 104 s) as
intervals; `data/labels/bout1_intervals.csv`. 793 windows fall inside labelled time. Scored by
reusing `demo_video`'s own functions — never reimplement that loop.

**Overall raw accuracy 19.0%. Random is 16.7%.** Barely above chance on continuous footage,
against ~86% on held-out clips.

| class | n_true | precision | recall |
|---|---|---|---|
| advance | 151 | 15% | **5%** |
| lunge | 21 | **3%** | 57% |
| neutral | 191 | 28% | 5% |
| parry | 8 | 2% | 12% |
| retreat | 78 | 33% | **73%** |
| walking | 344 | 53% | 19% |

- **`lunge` is a default, not a class.** 21 true windows; predicted **371 times**. It swallows
  advance (122 of 151), walking (155), neutral (70).
- **`retreat` works** (73% recall) — direction itself is learnable; the failure is specific.
- **A/B ASYMMETRY SOLVED — it was never the fencer.** A 10.4% vs B 27.2%, and the labels show A
  advances while B retreats in nearly every exchange. A scores badly because `advance` is broken,
  B well because `retreat` works. Mirroring, pose quality, facing coverage and handedness all came
  back negative because they were the wrong question.

*Scoring note:* the "viewer sees" variant originally counted `ready` as wrong when truth is
neutral/walking, but `ready` is the INTENDED render for QUIET_CLASSES, so it reported 7.3% — a
scoring artifact, not a result. The scorer now maps truth into the overlay's vocabulary instead.
(The first attempted fix would have made `ready` precision read 0%; mapping truth into the
overlay vocabulary is the correct direction.)

### Class-prior correction: 19.0% → ~42% (2026-08-04). ⚠ RETIRED 2026-08-09

Kept because the diagnosis is still correct and explains the shape of the whole clip era.

Root cause of `lunge` eating everything: **prior mismatch, not a broken feature.** Clips were cut
OF actions, so the corpus over-represents them (lunge 10% of training windows vs 2.6% of real
footage, parry 10% vs 1.0%), and inverse-frequency class weighting then drove the model's
effective prior to UNIFORM. It expected lunge ~6× too often, so lunge stopped being a class and
became a default.

Fix was standard label shift in `demo_video._classify_window`:
`p(c|x) * CLASS_PRIOR(c) / train_prior(c)`; since the train prior is uniform it drops out of the
renormalisation.

| class | before | after (held-out) |
|---|---|---|
| advance | 15% / **5%** | 53% / **67%** |
| walking | 54% / 19% | 54% / **49%** |
| retreat | 32% / 73% | 28% / 60% |
| neutral | 28% / 5% | 33% / 8% |
| lunge | 3% / 57% | 0% / 0% |
| **overall** | **19.0%** | **41.9%** |

**Quote 41.9%, not 50.2%.** The 50.2% figure fits the prior on the same footage it scores. 41.9%
is honest: prior fitted on one half of the labelled bout, scored on the other, both directions
agreeing (15.7%→42.6%, 22.6%→41.2%).

- `lunge` going to 0% is not a regression. Its old "57% recall" came with 3% precision off 378
  predictions for 21 true windows.
- **EM prior estimation does NOT work here** (Saerens et al., needs no labels). It estimated
  lunge at 0.635 against a true 0.027 and made things worse, 19.2% → 14.6%: the classifier is
  biased enough that EM just confirms its own error. The prior must come from labelled footage.

### PRIOR TRANSFER IS ASYMMETRIC — correcting an earlier over-claim (2026-08-05)

"The prior generalises" was based on one direction only. Measured all four ways:

| prior | bout 1 | bout 2 |
|---|---|---|
| none (uniform) | 19.2% | **46.1%** |
| bout 1 | 50.2%\* | 43.3% |
| bout 2 | **24.4%** | 47.2%\* |
| pooled (was shipped) | 45.9% | 44.1% |

(\* circular — that prior saw that bout.) Bout 1 is idle-heavy (walking 0.457), bout 2
action-dense (walking 0.232). A prior lifted from busy footage under-weights walking and wrecks
quiet footage (24.4%). Bout 2 scores 46.1% with NO correction because its true distribution is
already near uniform — **the correction's value depends on the footage**, which is the argument
that eventually retired it. Re-derive with `scripts/estimate_class_prior.py`.

### Second labelled bout (2026-08-05) — `data/raw_video/2.mp4`

`data/labels/bout2_intervals.csv` (ONO vs ITKIN, 110.9 s, 254 scored windows), prior fitted on
bout 1 and applied unchanged.

| | bout 1 (held-out prior) | bout 2 (bout-1 prior) |
|---|---|---|
| overall | 41.9% | **43.3%** |
| advance | 53% / 67% | **69% / 70%** |
| lunge | 0% / 0% | **50% / 19%** |
| walking | 54% / 49% | 40% / 55% |
| retreat | 28% / 60% | 38% / 35% |
| neutral | 33% / 8% | 38% / 12% |
| parry | 0% / 0% | 3% / 7% |

- **`advance` is now usable** at 69/70 — the class that was 15%/5% before the correction.
- **`lunge` is NOT dead.** Its 0% on bout 1 was 21 windows of thin data; on 31 windows here it
  reaches 50% precision.
- **`parry` took over lunge's old role.** Predicted ~37 times for 1 correct, and it eats
  `retreat`: 22 of 54 true retreat windows are called parry, dragging retreat recall to 35%.
- **A/B asymmetry, final form.** Bout 2 REVERSES the footwork roles (A retreats, B advances) yet
  A is still worse (32.4% vs 56.5%) — because in bout 2 fencer A performs all four parry
  sequences. **Per-fencer accuracy tracks WHICH ACTIONS that fencer performs**, not the slot, not
  the fencer, not handedness.
- `extension` appears in bout 2's labels; it is not one of the six classes, so
  `evaluate_labels.py` counts and EXCLUDES those windows (22 here) rather than penalising the
  model for a label it cannot emit.

### Parry vs retreat: a schema problem, with corrected numbers (2026-08-05)

Aaron: "when parrying, retreating is super duper common."

**The numbers first cited for this were wrong twice over.** The original table reported "median
999 s" for correctly-called retreats — 999 was a SENTINEL meaning "that fencer has no parry
anywhere to measure a distance to", so taking its median produced a duration that means nothing.
And the counts predated the silhouette filter. Recomputed on the current pipeline, separating "no
parry exists" from a real distance:

| bout | model said (truth = retreat) | n | fencer has NO parry | median dist. | within 3 s |
|---|---|---|---|---|---|
| 1 | parry | **0** | — | — | — |
| 1 | retreat | 62 | 0 | 15.15 s | 19% |
| 2 | parry | 25 | 0 | **2.65 s** | **60%** |
| 2 | retreat | 18 | **17** | — | — |

Bout 1's 22 retreat→parry confusions are GONE after the detector fix. Bout 2's 25 do sit near
real parries, but the control group is contaminated: 17 of the 18 correctly-called retreats
belong to a fencer with no parries at all. So the contrast is largely BETWEEN FENCERS rather than
between moments — much weaker support than originally claimed.

The schema argument does not depend on those numbers and still holds: a fencer genuinely parries
WHILE retreating, and six mutually-exclusive classes cannot express both.

### BLADE MOTION ENERGY — ⚠ REOPENED 2026-08-13, IT WORKS (0.79). The v1 null was underpowered

Aaron: "the problem with fast blades is that in the video sometimes it just disappears or it's
only a blur." Right, and that kills the detector on its own terms — there is nothing sharp in
those pixels, so more training frames cannot help. But blur is high frame-DIFFERENCE energy, so
the property that breaks a detector is the one a motion measure wants.

**Everything dated 2026-08-05 below was concluded from SIX parry intervals.** Re-scored over
**70 parries / 616 intervals** (bouts 3-5), pooled in one shot by `pool_blade_energy.py`:

| parry vs non-blade | pooled | bout 3 (n=11) | bout 4 (n=46) | bout 5 (n=13) |
|---|---|---|---|---|
| **v1 blade/torso (the MEAN)** | **0.79** | 0.62 | **0.79** | **0.75** |
| v1 p99 | 0.66 | 0.75 | 0.68 | 0.81 |
| v2 strip/ctrl (oriented, body-aligned) | 0.66 | 0.63 | 0.58 | 0.58 |
| v2 p99 | 0.55 | 0.58 | 0.50 | 0.52 |

**Blade motion energy WORKS — it was closed on a sample too small to see it.** 0.79 is the best
single-feature parry AUC this project has produced, and it survives the HARD comparison:
blade-action vs footwork 0.77, which is where a naive "motion" feature gets exposed, since during
an advance the whole fencer moves and the torso control divides that out.

The file predicted this at the time: "6 parry intervals cannot settle anything either way, which
is the concrete answer to *should I get more parry intervals*: **yes, that is exactly what they
resolve**." They did.

**AND THE TWO 2026-08-13 "FIXES" MADE IT WORSE — v2 scores 0.66 against v1's 0.79.** Both flaws
named at the bottom of this section were real, both were fixed, and the crude version still wins:

- *oriented strip* replacing the axis-aligned box — verified 30x more sensitive to a moving
  diagonal on synthetic frames
- *fencer's-own-torso alignment* replacing the global pan shift — recovers a known translation to
  0.01 px and flattens a pure translation to 27% residual

`blade_energy.py --self-test` asserts both, so this is not a geometry bug. Best untested guess:
`STRIP_HALFWIDTH` was halved to 0.45 forearm lengths because a blade is thin — but the strip is
anchored on the WRIST KEYPOINT, and a few pixels of pose error slide a narrow strip off the blade
while a fat box still contains it. **Tolerance of pose noise may be what the "wasteful" box was
actually buying.** Testing that costs a 2 h re-run at a wider strip; not done.

*A real bug found while building v2, worth remembering:* `prev_gray = gray` executes BEFORE the
per-fencer loop, so anything reading `prev_gray` in that loop differences a frame against itself
and reads zero motion everywhere — a clean, convincing null result for entirely the wrong reason.

*Also settled: the p99 retraction was CORRECT.* Pooled p99 is 0.66 against the mean's 0.79, so
the mean was always better and bout 2's p99 = 0.70 was noise, exactly as recorded.

#### AS A 7th FEATURE: NULL. A shuffled control caught what two positive holdouts hid

`scripts/exp_blade_feature.py`. Joins the two caches on `(slot, time)` — the same trick
`exp_opponent.py` used before opponent context was built — so no re-extraction was needed to
test it. Layout `[own(7) | opponent(7) | present(1)] = 15` against the shipped 13.

**First, the AUC does not survive the change of unit.** 0.79 was measured on hand-cut
INTERVALS; the model consumes fixed 2 s WINDOWS:

| blade/torso p90, parry vs non-blade | bout 3 | bout 4 | bout 5 | pooled |
|---|---|---|---|---|
| span 2.00 s (the model's window) | 0.34 | 0.49 | 0.43 | **0.53** |
| span 0.35 s | 0.51 | 0.65 | 0.57 | **0.66** |

Chance at the window level. The cause is DILUTION — a parry is ~0.6 s, so p90 over 2 s mostly
reports whatever else was in the window — and shortening the span recovers it monotonically, the
same reasoning that made `last` pooling beat `mean`. `BLADE_SPAN = 0.35 s`.
*(Note the pooled 0.53 exceeds all three bouts individually: Simpson's paradox from pooling
distributions with different per-bout scales. The interval-level 0.79 is pooled the same way.)*

**Then the training result, 4 seeds, and the control that kills it:**

| held out | baseline | + blade | **+ SHUFFLED blade** | real − shuffled |
|---|---|---|---|---|
| bout 4 | 67.9% | 69.1% (+1.20) | 67.7% (−0.16) | **+1.36** |
| bout 1 | 73.3% | 75.3% (+2.04) | **75.5% (+2.24)** | **−0.20** |

`--shuffle` permutes the blade column within each bout: identical marginal distribution,
identical parameter count, zero alignment to the window it describes. On bout 1 it performs **as
well as the real feature**, and its parry-recall delta is IDENTICAL (+3.1 both). So bout 1's
entire gain was the head's first Linear getting two inputs wider — capacity, not information.

**Verdict: NULL, do not ship.** Mean real-minus-control is +0.58 pts against seed sd of
0.7-2.1, positive on one holdout and negative on the other.

**And parry recall — the thing this was for — went the WRONG way on the bout that has the
data:** 18% → 16% on bout 4 (207 parry windows), against 16% → 19% on bout 1 (8 windows). The
apparent gains landed on `neutral` (+10.2) and `advance` (+11.2), i.e. the classes it was not
built for, which is usually the tell that a feature is not doing what its story says.

**METHOD, worth adopting generally: add a SHUFFLED-FEATURE CONTROL to every new feature test.**
Without it this reads "+1.6 pts mean, positive on both held-out bouts" and ships. Any new input
also widens the head, so some of every gain is capacity. The control costs one extra run and is
the only thing separating the two.

*Also fixed while running this: the delta line printed "+0.0 pts" for a real +1.6, because a
fraction was formatted with `:+.1f` instead of scaled to points. Exactly the kind of slip that
lets a null through as a win.*

**Where that leaves parry:** precision is solved (55%, via the gate), recall is not (17%) and
neither more labels, a separately-supervised blade head, nor blade motion energy has moved it.
The remaining ideas all need a better blade OBSERVABLE rather than a better use of this one.

---

*Original 2026-08-05 entry follows, superseded above but kept for the method note.*

`scripts/blade_energy.py` measures mean/p99 frame-difference inside a box projected along the
forearm (camera pan removed first; an equal-area TORSO box as control).
`scripts/analyze_blade_energy.py` scores it. Parry vs non-blade AUC, per interval:

| statistic | bout 2 (n=4 parries) | bout 1 (n=2) | **POOLED (n=6)** |
|---|---|---|---|
| box **mean**, blade/torso | 0.62 | 0.65 | **0.59** |
| box **p99**, blade/torso | **0.70** | 0.49 | **0.55** |

**This is the retraction described as shape 2 above** — on bout 2 alone p99 beat the mean and a
confident mechanism was written into this file for why (a blade is a thin streak; averaging over
a mostly-background box destroys it; p99 asks the right question). The mechanism is plausible and
the number did not replicate. Neither statistic is distinguishable from chance.

What survives:

- **Coverage, which is real and large.** The blade box is measurable on **100% of tracked
  fencer-frames** versus 1-2% for the detector. Whatever gets measured there gets measured
  everywhere, which the detector can never do on blurred frames.
- Before retrying, the two things most likely wrong with the measurement (untested): the box is
  ~3× forearm long and mostly background/opponent, and pan compensation is a single global
  x-shift estimated at 320×180, which will not cancel parallax on a moving camera.

This does NOT settle whether blade information helps — only that this measurement of it does not.

### THE REFEREE AND FOREGROUND SPECTATORS — diagnosed, filter SHIPPED (2026-08-05)

Aaron: "in bout 1 the person detector was clipping onto the referee silhouette in the middle a
lot." Confirmed. `get_fencer_boxes` resolved >2 people by keeping the **two highest-confidence**
boxes, and on bout 1:

- 68% of frames contain more than two tall people, so the tiebreak runs constantly
- on **51% of those** it picks a different pair than horizontal separation would
- the box it keeps and separation rejects sits at median **x = 0.49** (dead centre) with median
  confidence **0.85**, displacing a real fencer at **0.71**; 64% are in the middle third

The mechanism: confidence measures *resemblance to a standing person*. A referee is still,
upright and unoccluded; a fencer mid-lunge is blurred, horizontal and self-occluded. So
confidence systematically prefers the referee — the opposite of what is wanted.

**Three fixes were tried and ALL made it worse.** Bout 1, RAW model call:

| rule | overall | advance recall | retreat recall |
|---|---|---|---|
| top-2 by confidence (then current) | **45.9%** | 50% | 64% |
| all candidates → continuity via `_assign_boxes` | 35.7% | 11% | 13% |
| top-4 by confidence → continuity | 35.8% | 11% | 13% |
| widest-pair | 34.1% | — | — |

The second attempt tested the obvious explanation for the first — that removing the confidence
gate let junk 0.4-confidence blobs win on distance — and changed **nothing** (35.7 → 35.8). So
the regression was in the assignment logic itself, not the candidate pool. *That turned out to
be the absorbing-error mode later found and fixed in `_assign_boxes` (see bout 4).*

**THEN I LOOKED AT THE FRAMES, which should have come first.** `scripts/inspect_detections.py`
draws every tall detection on frames where the rules disagree (output in
`data/diagnostics/<stem>_detections/`). A single frame settled what three experiments could not:

| # | conf | x | height | what it actually is |
|---|---|---|---|---|
| 0 | 0.87 | 0.26 | 0.27 | left fencer |
| 1 | **0.85** | 0.11 | 0.32 | **foreground spectator silhouette** |
| 2 | 0.80 | 0.44 | **0.41** | the referee |
| 3 | 0.65 | 0.71 | 0.32 | right fencer |

The intruders are not only the referee, and they are **TALLER than the fencers** (0.32-0.41 vs
0.26-0.27) because they stand between the camera and the piste — so `MIN_BOX_H_FRAC` cannot reach
them, and raising it would delete the fencers first. Here the confidence rule keeps #0 and #1,
**drops the right fencer entirely**, and sorts the remaining pair by x so the LEFT fencer lands
in slot B. Slot B faces left, so net-forward inverts: the advance→retreat sign flip, visible
directly.

**The silhouette filter.** Two cues separate silhouettes from fencers and **neither works
alone** — over 1262 tall detections, boxes running to the frame bottom have median brightness 51
in bout 1 (vs 101 for the rest) but 106 in bout 2, whose tighter framing puts real fencers' feet
near the bottom edge. Together (dark AND bottom-anchored) they flag 16.2% of bout 1 and 4.1% of
bout 2 — the expected split, bout 1 being the wide shot. Fencers wear WHITE, which is what keeps
them clear. Brightness is judged **RELATIVE to the brightest box in frame**, since an absolute
cutoff also deletes a fencer in a dark patch and does not survive a change of venue.

Constants in `src/person_detector.py`: `FOREGROUND_MAX_REL_MEAN = 0.55`,
`FOREGROUND_MIN_BOTTOM = 0.90`. `FENCING_NO_SILHOUETTE_FILTER=1` toggles it off for A/B testing;
`_self_test_foreground()` covers it.

**It looked like it cost accuracy, and that was the coverage confound.** See the re-measurement
under bout 4 — absolute correct calls go UP, and the filter is KEPT.

---

## PARRY: the full thread

Kept together because the conclusion reversed twice and the intermediate steps are what make the
current position defensible.

### Bout 3 settles that the model had not learned parry AT ALL (2026-08-08)

`data/labels/bout3_intervals_2track.csv` — 44 intervals, action-hunted, **11 parries** against 6
across bouts 1+2, and parries from BOTH fencers (left 6, right 5), removing the per-fencer
confound that made bout 2's evidence weak.

Scored on 399 labelled windows:

| | parry predictions | parry recall | overall |
|---|---|---|---|
| shipped CLASS_PRIOR | **0** | 0% | 42.1% |
| no prior (uniform) | **0** | 0% | 35.3% |

Zero either way, so the prior is not the cause. The raw parry output is near-noise:

| | mean | median | max |
|---|---|---|---|
| true parry windows (55) | 0.037 | 0.038 | 0.106 |
| everything else (344) | 0.031 | 0.027 | 0.121 |

Parry never ranks #1 or #2 on ANY true-parry window (median rank #4 of 6); AUC 0.61. **No amount
of labels, priors or thresholds can move an output this uninformative.** It needs either blade
information the pose keypoints do not carry, or a window short enough not to average a 0.92 s
action across 2.0 s.

**Dead code path found by the same measurement:** `FAST_CLASSES = {"parry"}` lets the short window
override the long one, but requires `short_conf > FAST_CONF = 0.65`. Parry probability maxes at
0.106. The one mechanism built specifically to rescue parry is **unreachable by roughly 6× and
has likely never fired.** Verify before removing or re-tuning it.

Also on bout 3: **`advance` had become the default class** — 152 predictions for 60 true windows,
swallowing 51 of 64 lunges (lunge recall 9%), exactly the pathology `lunge` had before the prior
correction, transplanted. Bout 3 is action-hunted so its true shares (advance 0.150, lunge 0.160,
parry 0.138) are nothing like real footage — do NOT pool it into CLASS_PRIOR.

Bout 3 overall 42.1% sits with bout 1's 42.9% and bout 2's 42.3%: the pipeline was consistent
across three matches.

### "PARRY IS CLOSED" — ⚠ this conclusion has since been PARTLY REVERSED

The position taken on 2026-08-09 was: 0% in both architectures, with 47 parry clips (more than
advance's 35 or retreat's 33) — not a data problem, not a schema problem, not a prior problem,
not a pooling problem, so stop labelling for it.

**Two later results contradict the "stop labelling" part:**

1. **Continuous training moved parry for the first time** — 40% recall on held-out bout 2, 14% on
   bout 4 (shipped model at the time: 1%). Still 0% on bouts 1 and 3, so not solved, but no
   longer inert. What moved it was training data with real transitions, not any feature,
   architecture or threshold change.
2. **A separately-supervised blade head scales with blade labels** — 189 two-track labels put it
   at chance; 1329 put it at 2× base precision. See the two-head entry.

**Current position: parry is closed for the single-label six-way argmax; it is NOT closed for a
separately-supervised blade head, and more two-track parry labels are the thing that resolves
it.** Parry AUC after the slot fix is 0.62 on bout 4's 207 windows (bout 2 0.56, bout 3 0.60;
bout 1's 0.85 is 8 windows and means nothing) — some signal exists, far too little to win a
six-way argmax against a 0.017 prior.

### ⚠ SUPERSEDED — "interval labels are NOT training data"

An earlier note said: *"they are NOT training data — the model trains only from `data/clips/`. 20
minutes of intervals would not reach the model at all as things stand."* **That is now false.**
`extract_continuous.py` + `train_continuous.py` closed exactly that gap on 2026-08-09 and it was
worth +17.5 pts. The note's own prediction of where interval labels WOULD pay was right: "(2)
training on continuous windows instead of hand-cut clips, which would end the clip-cutting
artifact class that has produced three false results here."

### THE PER-FRAME MODEL IS A DEGENERATE LUNGE PREDICTOR (2026-08-08)

Recorded because the opposite conclusion was reached first, on bout 3 alone, and was wrong.

The hypothesis: `out.mean(dim=1)` over 60 frames dilutes a 0.8 s lunge, since a window is scored
at its NEWEST frame, so `ActionFrameLSTM` + `frame_logits_to_window(mode="last")` should fix it.
On bout 3 lunge recall went 9% → 78% and that looked decisive.

Checked on all three bouts, recall DOES replicate — 81/74/78% — but precision does not:

| `--frame-model`, raw | bout 1 | bout 2 | bout 3 |
|---|---|---|---|
| overall | **14.8%** | 38.1% | 34.3% |
| lunge recall | 81% | 74% | 78% |
| lunge **precision** | **3%** | 29% | 26% |
| predicts lunge on | **60%** of windows | 33% | 49% |
| true lunge share | **2%** | 12% | 16% |

High recall at 3% precision is a saturated class, not a detector, and the same pattern explains
its parry "recall" of 25% and 20% on bouts 1-2 (precision 2% and 6%). Fitting the prior to each
bout's OWN labels — which is cheating — still leaves bout 1 at 16.6% with lunge precision 3%, so
this is not a calibration problem either.

**Note the pooling hypothesis was nonetheless CORRECT** — it just needed a single-output model to
show it. See the pooling experiment below, where `last` beat `mean` by +4-5 pts on every bout.

---

## BOUT 4 (2026-08-09) — the continuous corpus becomes viable

`data/labels/bout4_intervals_2track.csv`, from a 26.1 min source: **304 intervals, 708 s of
labelled fencer-time**, 2.4× bouts 1-3 combined. Sparse by design (23% coverage) — Aaron: "if
there is a gap it probably means there's no arm/blade thing."

| | bouts 1-3 | bout 4 | total |
|---|---|---|---|
| labelled fencer-time | 294 s | **708 s** | 1002 s |
| parries | 17 | **46** | 63 |
| lunges | 33 | 61 | 94 |
| independent ~2 s windows | ~147 | **~354** | **~500** |

**This crossed the threshold that made continuous training not worth doing.** The clip corpus is
488 windows; the continuous corpus is now ~500, with realistic transitions and no clip-cutting
artifacts.

**Still do NOT pool bout 4 into CLASS_PRIOR.** 23% coverage selected for action: its duration
shares describe where Aaron looked, not the sport.

Transcription notes (source table had a few slips, recorded rather than silently fixed): two
malformed timestamps (`20.53.520`, `22:118.967`) read as `20:53.520` and `22:18.967`; one
right-fencer blade action at 1360.045 s dropped because its footwork cell was blank; five
overlaps of 0.01-0.14 s clamped by truncating the earlier interval (the shortest real interval in
the bout is 0.30 s, so a clamp that size cannot move a label onto the wrong action).

### FIXED: A/B SLOT SWAPPING WAS THE LARGEST ERROR IN THE SYSTEM. Bout 4 43.6% → 55.1%

Found only because bout 4 is big enough (3822 scored windows) to separate two hypotheses that
look identical at bout-1-to-3 scale. The dominant error was 556 advance↔retreat confusions,
near-symmetric. Two possible causes: the model cannot judge direction, or it can and the slots
are swapped. The test: fencers move in OPPOSITE directions essentially always, so ask what the
model says about the PAIR.

| model committed to a direction for both fencers | n | |
|---|---|---|
| INVERTED pair — right that they oppose, wrong which way | 155 | **47%** |
| correct pair | 110 | 34% |
| both same (physically impossible) | 63 | 19% |

81% opposite when it commits to both, against 50% for a coin flip — so directional signal was
always there. And the 155 inverted windows form **~19 CONTIGUOUS RUNS** (median gap 0.17 s =
consecutive predictions), not scattered flicker: a sustained mis-assignment, not noise.

Cause, in `_assign_boxes`: with two detections it swapped A/B whenever remembered hip positions
preferred it. Assignment writes `last_hip_x`, which drives the next assignment, so **ONE bad swap
persists indefinitely** — the same absorbing-error mode that wrecked the three referee fixes, in
code that had been read twice without noticing.

**Fix: fencers never cross a piste, so relative x-order IS identity.** Leftmost → A, rightmost →
B, always, no history. Memoryless, so a bad frame cannot propagate. Covered by
`_self_test_assign()` (`demo_video.py --self-test`).

| bout | before | after | scored windows |
|---|---|---|---|
| 1 | 42.9% | 43.4% | 874 |
| 2 | 42.3% | **48.3%** | 266 |
| 3 | 42.1% | 42.1% | 399 |
| **4** | 43.6% | **55.1%** | **3822** |
| 4, viewer view | 51.7% | **61.5%** | |

No bout regressed. On bout 4: advance 29%/30% → **52%/54%**, retreat 29%/39% → **50%/67%**, and
the 556 direction confusions fall to 174 (−69%) — the mechanism check the fix was predicted to
pass. Parry is untouched (1 correct of 211 before, 3 of 207 after), as expected.

**Why this was missed for so long:** at 266-874 windows the inverted runs are rare enough to look
like ordinary confusion. It took 26 minutes of labelled footage to make a tracking bug
distinguishable from a learning problem. **That is the strongest argument yet for continuous
labelled data — not as training material, but as instrumentation.**

### Two follow-ups after the fix, both cheap and both negative

- **CLASS_PRIOR was already optimal.** Suspected of having been tuned around the swap bug.
  Re-fitted on the corrected pipeline it gives 43.8 / 48.3 / 42.1 / 55.0 against the shipped
  43.4 / 48.3 / 42.1 / 55.1. No difference.
- **Silhouette filter re-measured — KEEP IT.** Its original "−3 pt cost" was taken under the swap
  bug. Re-run against the corrected pipeline:

  | bout | filter ON | windows | correct | filter OFF | windows | correct |
  |---|---|---|---|---|---|---|
  | 1 | 43.4% | 869 | **377** | 46.0% | 789 | 363 |
  | 2 | 48.3% | 265 | **128** | 50.4% | 252 | 127 |
  | 3 | 42.1% | 399 | 168 | 42.1% | 399 | 168 |

  Filter OFF looks 2 pts better and is not: it scores 80 FEWER windows on bout 1 and gets 14
  FEWER calls right. Accuracy per window falls with the filter on because the windows it adds are
  hard ones. **Absolute correct calls is the honest metric when a change moves coverage.** Bout 3
  has no foreground silhouettes so the filter is a no-op there.

  (Earlier figures under the swap bug, kept for the record: dark+bottom with absolute brightness
  42.4% / 42.3%, with relative brightness 42.9%, against top-2-by-confidence at 45.9% / 44.1%;
  coverage rose 787→874 windows because a silhouette in a slot produced a frozen skeleton that
  `MAX_FROZEN_FRAC` suppressed. Under the swap bug, advance recall also dropped 50%→40% on
  essentially the same window set (151→156), which coverage does not explain and which
  re-fitting the class prior did not recover — held-out 46.2% / 39.4% against the shipped
  prior's 42.4%. The slot fix is what explained it.)

---

## THE 2026-08-09 SWEEP — three ships, three null results, one learning curve

### CONTINUOUS TRAINING WORKS: +17.5 pts leave-one-bout-out

The model had only ever trained on 488 hand-cut clip windows; interval labels were used for
evaluation and the prior, never for learning. `scripts/extract_continuous.py` closes that (5352
windows across four bouts), `scripts/train_continuous.py` tests it. `window_tensors()` in the
extractor mirrors `_classify_window` exactly.

| held-out | clips only | **clips + continuous** |
|---|---|---|
| bout 1 | 41.0% | **70.0%** |
| bout 2 | 45.7% | **54.3%** |
| bout 3 | 35.3% | **60.4%** |
| bout 4 | 49.5% | **57.0%** |
| **mean** | **42.9%** | **60.4%** |

Same recipe on both sides, 2 seeds. Every bout improves.

**THE POST-HOC CLASS_PRIOR CAN BE RETIRED.** Training unweighted on continuous windows, which
arrive at their NATURAL frequencies, scores **60.4%** — identical to inverse-frequency weighting
plus the prior multiplied back in. The prior was always a patch for clips over-representing
exciting classes while inverse-frequency weighting drove the effective prior to uniform. Real
frequencies remove the need for both: same accuracy, one fewer hand-tuned constant, no per-venue
re-estimation.

**End-to-end verification on HELD-OUT bout 1** (`train_shipping.py --holdout 1`):

| bout 1 | shipped (clips only) | continuous, held-out |
|---|---|---|
| raw overall | 43.4% | **69.0%** |
| offline tensor estimate | — | 70.0% |
| advance | 47%/37% | **81%/53%** |
| retreat | 34%/67% | **41%/98%** |
| walking | 66%/63% | **84%/94%** |
| neutral | 30%/6% | **90%/29%** |

**Online 69.0% vs offline 70.0% — a 1 pt gap, so extraction and inference agree**, and the
leave-one-bout-out numbers can be trusted. Per-fencer is balanced (A 70.0%, B 68.2%).

Caveats:

- The shipped checkpoint trains on all four bouts, so it **cannot** be honestly scored on them.
  The honest estimates are the leave-one-bout-out 60.4% and this held-out 69.0%.
- The `clips only` baseline here is a single 40-epoch model, weaker than the then-shipped 5-model
  ensemble with best-epoch selection (which averages 47.2% across these bouts). **Compare 60.4%
  against 47.2% for the honest gain**, not +17.5.
- 2 seeds. Variance low (±0.1-1.4%) except the clips-only baseline (±13.3% on bout 1), which is
  itself informative: 488 windows is not enough to train stably.
- **No best-epoch selection and no class weighting, both deliberate.** Best-epoch selection is a
  known trap here — it lands on lunge-heavy checkpoints and validation accuracy is
  anti-correlated with demo behaviour.
- Training uses `--stride 3` on continuous windows: at 92% overlap, every third window still
  leaves 75% overlap, so almost no information is lost and training cost drops 3×. **Evaluation
  always uses every window.**

### POOLING: `last` beats `mean` by +4-5 pts on every bout. SHIPPED

`scripts/exp_pooling.py`. Five masked reductions, identical trunk and recipe, leave-one-bout-out.

| pooling | bout 1 | bout 3 | bout 4 |
|---|---|---|---|
| mean (was shipped) | 69.0% | 64.9% | 57.9% |
| max | 71.6% | 65.2% | 60.5% |
| meanmax | 69.3% | 64.0% | 59.5% |
| attn | 70.0% | 64.0% | 58.4% |
| **last** | **73.0%** | **69.8%** | **62.6%** |

Consistent +4 to +4.9 on all three, and the gain is concentrated where predicted — bout 4: lunge
37→56%, advance 30→43%, parry 13→20%. **A window is scored at its NEWEST frame, so the last real
timestep is the one the label actually describes**; averaging a 0.7 s lunge across 2 s was
destroying it.

**End-to-end verified** on held-out bout 1: **74.0%** (mean-pooled 69.0%, original shipped 43.4%),
viewer view 79.5%. advance 67%/65%, lunge 44%/33%, neutral 85%/39%, retreat 54%/96%,
walking 88%/95%.

**`ActionLSTM(pool=...)` defaults to `mean` deliberately: all modes have IDENTICAL parameter
shapes, so a mismatched mode loads silently and behaves wrong.** A checkpoint's mode is part of
its identity — `demo_video.POOL_MODE` sits next to `MODEL_PATH`, and `train_shipping.py --pool`
must agree with it. Pre-2026-08-09 checkpoints are `mean`.

An earlier version of this conclusion was WRONG (the per-frame model, above). What made the
difference was a single-output model, three held-out bouts, and reading precision next to recall.

### OPPONENT CONTEXT WORKS: +2.9 pts, replicated on three bouts. SHIPPED

**Each fencer was being classified completely independently** — slot A's window never saw slot B.
Fencing is interactive and the labels prove it: advance/retreat arrive as opposing pairs almost
always, and 34 of 46 parries happen during a retreat because the opponent is attacking. The model
was being asked to recognise a RESPONSE without seeing the STIMULUS.

`scripts/exp_opponent.py` pairs each window with the opponent's window at the same timestamp
(extraction cached `time` and `slot`, so no re-extraction) and appends their 6 engineered
features: 6 own + 6 opponent + 1 presence flag = 13. Opponent found on 88-96% of windows.

| cont only (clean arm) | own | + opponent |
|---|---|---|
| bout 1 | 71.5% | 72.0% |
| bout 3 | 70.3% | **72.4%** |
| bout 4 | 60.3% | **66.5%** |
| **mean** | 67.4% | **70.3%** |

Positive on all three. **And 70.3% beats the clips+cont recipe at 68.5%** — dropping the clip
corpus entirely and adding the opponent is better than keeping clips without it.

**THE CLIP CORPUS CANNOT BE NAIVELY MIXED IN.** Clips are single-fencer keypoint files with no
opponent, so their opponent block is all zeros — perfectly correlated with "this came from a
clip" and usable as a source shortcut. The presence flag makes the zeros explicit but does not
fix it: the clips+cont arm scored −3.4 / +2.4 / +2.4, actively harmful on bout 1. **The
`cont only` arm exists precisely because it cannot cheat, and it is the one to trust.**

Implementation notes worth keeping:

- `wide_agg()` is the ONLY definition of the 13-vector layout; everything imports it.
- `_window_inputs()` was split out of `_classify_window` so the OPPONENT's features come from
  exactly the same gates and normalisation as the fencer's own.
- **Predict only after BOTH tracks have the current frame.** `demo_video` used to predict inside
  the per-slot loop, which would have handed slot A an opponent one frame stale; the
  frame-alignment guard would then have silently discarded the opponent on every A call and
  quietly reverted half the predictions to the single-fencer model.
- A wrong `n_agg` **RAISES** (the head's Linear changes shape) — unlike `pool`, which loads
  silently. That difference caught a missed loader during this very change.

**KNOWN train/serve gap, measured and benign.** In training, "opponent present" means present AND
LABELLED at that timestamp (extraction only emits labelled windows), so coverage is 88-96%. At
inference it means present, ~99%. The live model gets slightly MORE opponent information than it
trained with, which is why end-to-end (74.6%) beat offline (72.0%) by 2.6 pts rather than falling
short. **If that ever inverts, re-extract with opponent features computed regardless of label
coverage.**

### TWO HEADS: Aaron's two-indicator framing (2026-08-09)

Aaron: "what if we have two indicators, a footwork and then a parry one, so that when a parry
comes on, both can be shown instead of one taking over the other."

This exposed a scoring error. `exp_two_head.py` originally COLLAPSED the two heads back to one
label, so every false parry overwrote a correct footwork call and the metric charged it the full
cost of a destroyed prediction. **Displayed as two independent indicators that cost does not
exist.** I scored a two-indicator model with a one-indicator metric.

Re-run with `--pool last` and reported per track:

| held out | blade labels in training | footwork 5-way | parry lamp, best threshold | collapsed |
|---|---|---|---|---|
| bout 4 | 189 | 64.0% | 9% prec @ 36% lit — CHANCE (base 5.4%) | 51.8% |
| bout 3 | 1329 | **77.9%** | 28% prec @ 3.4% lit (base 13.8%) — 2× chance | **71.3%** |

- **The two-head structure does pay with `last` pooling**: collapsed 71.3% on bout 3 beats the
  single head's 69.8%. With `mean` it did not (65.5%).
- ~~**Parry precision scales with blade supervision.** Starved (189 labels) the head is at
  chance; fed (1329) it is 2× base rate.~~ **⚠ WRONG — RETRACTED 2026-08-13, see below.** That
  compared precision across two DIFFERENT held-out bouts with different base rates: 9% against
  bout 4's 5.4% base is 1.7×, 28% against bout 3's 13.8% base is 2.0×. Almost the same lift. The
  "scaling" was an artifact of comparing bouts, not of label count.

**A threshold sweep is the right diagnostic for a lamp.** On bout 4 precision stays pinned at 9%
from 0.25 to 0.90 while the lit fraction falls 36%→20% — exactly what no discriminative signal
looks like. On bout 3 precision moves 0%→33% across the sweep, which is what real (if weak)
signal looks like.

**These numbers predate opponent context. Re-run with `--pool last` + opponent before acting.**

### SHIPPED: THE PARRY GATE — a parry needs an attacker. Precision 29% → 55% (2026-08-13)

Aaron: "lunge and parry are usually together (not always, but many times they are)."

Measured across bouts 3-5, taking each labelled interval and asking what the OPPONENT is
doing at the same moment (67 parries, 89 lunges):

| during a PARRY, the opponent is | | during a LUNGE, the opponent is | |
|---|---|---|---|
| lunge + extension | **76%** | retreat + parry | 48% |
| advance + extension | 10% | lunge + extension | 15% |
| **attacking in some form** | **86%** | **parrying in some form** | 58% |

**The relationship is ASYMMETRIC and that is what makes it usable.** A parry is a RESPONSE —
it essentially does not happen unless someone is coming at you (86%). A lunge is only a
coin-flip to draw one (58%). So the rule runs one way only: opponent-attacking gates parry,
never the reverse.

`demo_video._apply_parry_gate` drops a `parry` call unless the opponent's lunge probability
clears `PARRY_OPP_LUNGE_MIN = 0.20`, demoting it to that fencer's own runner-up class. It runs
after BOTH tracks have predicted, so each fencer sees the other's distribution for the same
frame, and it is wired into all three prediction loops (demo, `evaluate_labels`, `draft_labels`)
so scoring measures what the demo shows.

Offline on cached held-out probabilities, two bouts at two venues:

| | overall | parry precision |
|---|---|---|
| bout 4 (verify_h4) | 67.4% → **68.2%** | 18% → **38%** |
| bout 5 (action_opp) | 57.0% → **58.7%** | 12% → **27%** |

**End-to-end on held-out bout 4 with the current checkpoint, and it beat the offline estimate:**

| held-out bout 4 | gate off | **gate on** |
|---|---|---|
| overall | 71.3% | **72.2%** |
| parry precision / recall | 29% / 19% | **55% / 17%** |
| viewer view, parry precision | 42% | **61%** |

**Precision nearly doubles and recall barely moves** (19% → 17%), because a better base model
leaves fewer true parries for the gate to remove. 55% is by a wide margin the best parry
precision this project has recorded.

**Overall accuracy goes UP as well**, which is the tell that this is not a precision/recall
trade being spun: the suppressed parries were mostly wrong, so the runner-up class is more often
right. Threshold 0.2 is best-or-tied on both bouts — what a venue-INDEPENDENT rule looks like,
in contrast to the fencing gate whose best cue inverted between venues.

**Why this worked where more labels did not.** The model already receives the opponent's six
engineered features; what it never receives is the opponent's predicted CLASS, because the two
fencers are classified independently. This couples them at decision time — the cheapest possible
joint decoding.

**⚠ THE SAME TRICK DOES NOT WORK FOR DIRECTION — checked before building it.** The obvious next
step looked like "advance and retreat must be opposite", and Aaron stopped it: "many times both
fencers will still advance to each other." The labels agree (315 overlapping interval pairs,
bouts 3-5):

| left | right | n | share |
|---|---|---|---|
| advance | retreat | 41 | 13% |
| retreat | advance | 40 | 13% |
| **advance** | **advance** | **37** | **12%** |
| lunge | retreat | 29 | 9% |
| lunge | lunge | 7 | 2% |

**Both fencers moving forward is 18% of pairs against 43% opposed.** A hard opposite-direction
rule would be wrong about one time in five, and wrong specifically during the approach that opens
every phrase. P(opponent retreating | I advance) is 47% — nothing like the parry conditional's
86%.

*Method note:* the tempting evidence for the direction rule was the slot-bug finding that the
model is "81% opposite when it commits to both directions". That is conditioned on the model
having already predicted one advance and one retreat — a statement about model outputs, not
about how often the truth is opposed. Generalising a conditional into a prior is the same error
shape as the retracted "parry precision scales with blade supervision", which compared rates
across different base rates.

### PARRY LAMP, corrected (2026-08-13): labels buy RECALL, opponent buys PRECISION

`exp_two_head.py` gained three things it needed before it could answer this: bout 5 in
`CSV_FOR` (it would have raised KeyError), an `--opponent` flag threading `N_AGG_WIDE` through
both heads, and `--opponent` implying no clips (a clip's all-zero opponent block is a perfect
"came from a clip" shortcut, measured harmful for the single head).

With bout 5 AND opponent, `--pool last`, 2 seeds:

| held out | blade labels | parry lamp | footwork 5-way | collapsed |
|---|---|---|---|---|
| bout 4 (base 5.4%) | 189 → **1168** | 9% flat → **15% @ 0.25, 46% @ 0.90** | 64.0 → **75.2%** | 51.8 → **67.7%** |
| bout 3 (base 13.8%) | 1329 → **2308** | 28% → **65% @ 0.25** | 77.9 → **78.2%** | 71.3 → **73.1%** |

**The shape of the curve is the evidence, not the headline.** On bout 4 precision used to be
PINNED at 9% from threshold 0.25 to 0.90 while the lit fraction fell 36%→20% — flatness is what
no discriminative signal looks like, and it is how the original failure was diagnosed. It now
rises monotonically: 15, 15, 16, 16, 17, **46%**. A threshold that buys precision means the head
is genuinely ranking real parries above false ones.

**ABLATION — two variables changed at once, so they were separated.** `--blade-frac 0.162` cuts
blade labels back to the old 189 on the SAME held-out bout, everything else fixed:

| blade labels | precision @0.25 | recall @0.25 | lamp lit |
|---|---|---|---|
| 189 (ablated, only 6 parry labels) | **14%** | 11% | 4.4% |
| 1168 (full) | **15%** | **36%** | 13.8% |

**Precision is identical; recall more than triples.** So:

- **More two-track parry labels buy RECALL.**
- **Opponent context buys PRECISION** (9% → 15% on bout 4, ~2.8× base).

This is the useful correction, and it retargets the labelling advice: labelling more parries
makes the lamp light MORE OFTEN on real parries, but does NOT make a lit lamp more trustworthy.
Only better features/context have done that so far.

**STILL NOT SHIPPED, and the reason is the operating point, not the signal.** At 15% precision
85% of lit lamps are wrong — unusable for a display where a wrong lamp is the visible failure.
The usable end differs by bout: bout 3 reaches 65% at threshold 0.25, bout 4 needs 0.90 to reach
46%, where it lights on 1.2% of windows. **A single shipped threshold will behave differently
per broadcast** — the same venue-dependence that sank the fencing gate.

Also note the collapsed metric: 67.7% against the single head's 71.3% on the same held-out bout.
**Two-head is an ADDITIONAL indicator, not a replacement** — which is exactly Aaron's original
framing ("both can be shown instead of one taking over the other"). Do not swap the six-way
model out for it.

### LEARNING CURVE: more labelling still pays, but only a few points

`scripts/learning_curve.py --holdout 1`. Fractions are **CONTIGUOUS TIME SLICES**, not random
window samples — labelling less footage means covering less of the match, and since neighbouring
windows share 92% of their frames a random 25% sample still covers the whole bout and would
flatter the curve badly. (`--random-subsample` exists to demonstrate that, not to be used.)

| fraction | continuous windows | held-out bout 1 |
|---|---|---|
| 0.00 (clips only) | 0 | 25.8% |
| 0.10 | 149 | 46.3% |
| 0.25 | 373 | 58.4% |
| 0.50 | 746 | 64.2% |
| 0.75 | 1122 | 66.8% |
| 1.00 | 1495 | **69.0%** |

Marginal value of the last 25%: **+2.2 pts** (previous step +2.6). Shallow and steady, not a
cliff. **Practical translation: another bout the size of bout 4 buys roughly +3-5 points**, not
another +20.

**It will NOT fix parry: 0% recall at FULL data.** Weak classes at full data are parry 0%, lunge
24%, neutral 30% — the transient and quiet ones. retreat (98%) and walking (94%) are done. So the
remaining headroom is in classes that more of the same footage demonstrably does not reach, which
points at model changes (a separate blade head, pooling) rather than at more labelling of the
same kind.

### WINDOW LENGTH: null result, 2 s is about right

`WINDOW_LONG = 60` had been fixed since the start and never swept. `last` pooling winning
suggested recent frames carry the signal, and the actions are short (lunge 0.7 s, parry 0.6 s),
so a shorter window was a live hypothesis. Tested free by slicing the newest k frames out of
cached windows (`scripts/exp_window.py`; note `tail()` takes a **SUFFIX** — `X[:, :k]` would grab
the OLDEST frames and silently test the opposite thing).

| frames | seconds | bout 1 | bout 4 |
|---|---|---|---|
| 15 | 0.50 | 65.3% | 58.1% |
| 25 | 0.83 | 69.3% | 60.4% |
| 35 | 1.17 | 72.9% | 60.3% |
| 45 | 1.50 | **73.5%** | **63.4%** |
| 60 (current) | 2.00 | 73.0% | 62.6% |

45 frames edges 60 by +0.5/+0.8, inside the ±0.6-2.4 seed noise. Shorter windows are clearly
worse — 15 frames costs 8 points. **The hypothesis is wrong and the current window is fine.**

*Limitation, stated because it would otherwise be invisible:* `agg` still covers the full 2 s in
every row (the motion tracks were never cached and two features are raw sums), so this isolates
SEQUENCE length rather than the whole pipeline's window. A win would have justified the expensive
re-extraction; a null result closes it cheaply.

*This does supersede the older speculation that "footwork is a 1-3 s phenomenon and blade action
a 0.5 s one; they probably should not share a window length." At the sequence level they can.*

### MIRRORING: correct, invariance proven, NOT WORTH SHIPPING

Aaron asked whether the video could be cropped/edited for variation, the way Roboflow augmentation
helped the blade detector. **No** — and the reason is worth keeping: `_normalize_sequence`
hip-centres and divides by body height, so translation and scale are gone before the LSTM sees
anything. Crop or zoom the video and the tensors are ~identical. **The blade model was a PIXEL
model; this one is not.**

Mirroring is the one augmentation normalisation does not erase (which leg leads, which arm
extends, which way the fencer faces). `scripts/exp_mirror.py`.

**The trap, and the proof.** `forward = world_vel * nose_dir`. Mirror the keypoints and `nose_dir`
flips; miss the motion track and `world_vel` does not, so **every advance silently becomes a
retreat** — the same direction inversion that cost 11 points earlier the same day, and invisible
because the data would just be quietly wrong. Under a TRUE mirror both terms flip, so `forward`
is unchanged, and by symmetry so is everything else. `test_invariance()` checks that against the
real `_engineered_features` over 200 random windows: **worst delta 2.98e-07.** So the cached `agg`
can be reused and only X needs mirroring — which matters, because the motion tracks were never
cached.

| held out | baseline | + mirrored |
|---|---|---|
| bout 1 | 73.0% | 73.2% |
| bout 3 | 69.8% | 70.8% |
| bout 4 | 62.6% | 62.8% |

Positive on all three but **+0.5 mean against ±1.5 seed noise**, so not claimable, and on bout 1
it trades advance 61→76 for lunge 43→24 and parry 38→25. Doubles training time for an effect
indistinguishable from zero. NOT SHIPPED; kept for when there is more data.

(An earlier run used `mean` pooling — no longer the deployed config — and came out +1.5 / −1.3,
i.e. pure noise. Always pass `--pool last`.)

**What augmentation CANNOT touch:** the framing-sensitive parts of this pipeline are not learned.
`MIN_BOX_H_FRAC`, the silhouette brightness/bottom thresholds and the pan strips are hand-
calibrated constants, all calibrated on the same broadcast style. **Only real footage from a
different venue tests those.**

### "IS THIS FENCING?" GATE: does not work with these cues. NOT SHIPPED

63% of the demo's predictions on bout 4 (6469 of 10288) land outside labelled fencing, i.e. the
overlay labels replays, crowd shots and graphics. `scripts/calibrate_gate.py` tests whether cheap
detection-only cues can tell live fencing from filler, calibrated on bout 4 — Aaron confirmed its
label gaps ARE broadcast filler, so inside-labelled = fencing, outside = filler. 4699 sampled
frames, 23% fencing.

| cue | fencing med | filler med | AUC |
|---|---|---|---|
| n_tall | 2.00 | 2.00 | 0.71 |
| sep (box-centre gap) | 0.45 | 0.21 | 0.71 |
| h_ratio | 0.93 | 0.77 | 0.69 |
| foot_dy | 0.03 | 0.08 | 0.35 (0.65 inverted) |
| box_h | 0.43 | 0.54 | 0.37 (0.63 inverted) |
| **motion** | 22.28 | 21.47 | **0.50** |

Best conjunction over a full threshold grid: **36% precision at 86% recall** against a 23% base
rate — and the grid is FLAT at 34-36% everywhere, so the thresholds contribute nothing and all
the discrimination is "exactly 2 tall people".

**Root cause: a replay of a touch is geometrically identical to the live touch.** Two fencers,
similar size, separated, feet aligned — the same picture. ~47% of filler frames pass the geometry
gate. Geometry catches crowd shots, graphics and close-ups, which are the minority.

Two hypotheses died here, both worth not re-testing:

- **Slow motion: AUC 0.50, literally zero signal.** Replays were expected to have much lower
  inter-frame motion. They do not — presumably re-encoded to the same frame rate. The most
  intuitive cue, carrying no information at all.
- **Shot tightness: mild and insufficient.** box_h 0.63 inverted, i.e. filler DOES sit closer to
  camera, consistent with replays being tighter shots — nowhere near enough alone.

Untested idea for next time: the **scoreboard/timer overlay**, present during live play and often
removed or restyled during replays. A fixed-region cue would be venue-specific, which is its own
problem, but it targets the actual distinction rather than the fencers.
**→ CHECKED ON BOUT 5 AND DEAD.** That venue's score bar is a permanent graphic, up during live
fencing and during close-up filler alike. The arena's physical LED board shows only in the wide
shot, but "is the wide shot up" is just shot tightness, which is already measured.

Shipping a 36%-precision gate would suppress **14% of real fencing labels** to remove about half
the phantom ones. Not worth a missing label on a real action, which is the failure a viewer
notices.

#### BOUT 5 REVERSES THE CUES — the gate is VENUE-DEPENDENT, which is why it stays unshipped

Re-run unchanged on the new venue (3460 frames, 54% fencing), the script that failed on bout 4
now looks like a success:

| cue | bout 4 AUC | **bout 5 AUC** | bout 5 medians (fencing / filler) |
|---|---|---|---|
| n_tall | 0.71 | 0.76 | 2.00 / 1.00 |
| h_ratio | 0.69 | **0.79** | 0.91 / 0.00 |
| sep | 0.71 | 0.78 | 0.35 / 0.00 |
| foot_dy | 0.65 inv | 0.72 inv | 0.01 / 1.00 |
| box_h | 0.63 inv | 0.66 inv | 0.36 / 0.74 |
| **motion** | **0.50 — no signal** | **0.83 INVERTED** | **7.05 / 17.03** |

Best conjunction: **76% precision at 65% recall against a 54% base rate**, against bout 4's 36%
at a 23% base.

**The motion cue does not merely get stronger, it points the OPPOSITE WAY, and one mechanism
explains both bouts.** On bout 4 the camera pans to follow the fencers, so live play carries as
much frame-difference as a replay (22.28 vs 21.47, AUC 0.50). On bout 5 the camera is locked off
for fencing and cuts to close-ups for filler, so filler carries **2.4× the motion of live play**.
The cue was never measuring slow motion at all — **it measures CAMERA WORK**, a property of the
broadcast director rather than of fencing.

**So bout 5 strengthens the case against shipping a gate rather than weakening it.** A rule
calibrated here reads 76% precision and would read 36% on bout 4; a `motion` threshold learned
here would be inverted there and would actively suppress live fencing. Same failure mode as
`MIN_BOX_H_FRAC` at 0.35 (tuned on bout 2, amputated bout 1) — except the correct setting
differs in SIGN, not magnitude, so no single value exists.

And even where it "works", bout 5's gate tops out at **65-69% recall**: it discards roughly a
third of real fencing, because "exactly 2 tall people" fails whenever the detector drops a fencer
(12% of frames) or picks up a referee. The precision reads well because the base rate is 54%, not
because the rule is good.

### BOUT 5 (2026-08-12) — the first DIFFERENT VENUE

`data/labels/bout5_intervals_2track.csv`, from `data/raw_video/5.mp4`: ANANE (FRA) vs
BIANCHI (ITA), a Genoa FIE event. **144 intervals, 619 s of labelled fencer-time, 13
parries, 51 arm extensions**, transcribed by `scripts/transcribe_bout5.py`.

**Why this bout is different in kind, not just more data.** Every constant that governs
framing — `MIN_BOX_H_FRAC`, the silhouette brightness/bottom thresholds, the pan strips
— was hand-calibrated on bouts 1-4, which are all 1920x1080 at 29.97 fps from similar
broadcasts. Bout 5 is **1906x1080 at exactly 30.000 fps**, dark arena with spotlit
piste. CLAUDE.md had this listed as untestable ("What augmentation CANNOT touch: only
real footage from a different venue tests those"). It is now testable.

| bout | intervals | fencer-time | parries | extensions |
|---|---|---|---|---|
| 1 | 26 | 164 s | 2 | 0 |
| 2 | 22 | 58 s | 4 | 3 |
| 3 | 22 | 72 s | 11 | 11 |
| 4 | 152 | 708 s | 46 | 73 |
| **5** | **144** | **619 s** | **13** | **51** |
| **total** | **366** | **1621 s** | **76** | **138** |

Labelled fencer-time is up 62% (1002 s → 1621 s), and bout 5 is densely labelled —
**54% coverage per fencer** against bout 4's 23%, with the between-phrase `walk` marked
rather than left as a gap.

**`MIN_BOX_H_FRAC = 0.25` TRANSFERS — first real test, and it passes.** Sampled every
45th frame:

| | ≥2 boxes over 0.25 | two tallest, p5/p50/p95 |
|---|---|---|
| inside labelled fencing | **88%** | 0.24 / 0.37 / 0.43 |
| outside (filler) | 36% | 0.23 / **0.78** / 0.98 |

Fencers sit at 0.24-0.43 of frame height, the same band as bout 1 (0.20-0.50) and
bout 2 (0.30-0.60). The constant that broke bout 1 when it was 0.35 survives a genuinely
new broadcast at 0.25.

**Label alignment verified against frames, not assumed.** At 86.75 s both fencers are
mid-lunge exactly as labelled; at 248 s a fencer is on the piste being helped up, which
is the `walk` reset the labels claim. No time offset.

**THE SCOREBOARD GATE CUE IS DEAD.** The one untested idea left over from the failed
"is this fencing?" gate was the score/timer overlay, on the theory that broadcasts drop
it during replays. In bout 5 the bottom score bar is present during live fencing AND
during the close-up filler — it is a permanent graphic, not a live-play indicator. The
arena's physical LED board is visible only in the wide shot, but "is the wide shot up"
is just shot tightness, which is already measured (box_h AUC 0.63 on bout 4).

**CROSS-VENUE GENERALISATION: 58.1%. The first honest number of its kind.** Bout 5 was
never in training, so the SHIPPED `action_opp.pth` is legitimately held out on it — no
verify checkpoint needed, no circularity.

| held-out bout | per-window accuracy | viewer view |
|---|---|---|
| bout 1 (same broadcast family) | 74.6% | 80.8% |
| bout 4 (same broadcast family) | 67.6% | 70.4% |
| **bout 5 (NEW VENUE)** | **58.1%** | **63.3%** |

**A new venue costs 10-17 points.** That is the number to quote for "how well does this
work on footage we have never seen", and it is the first time the project has been able
to measure it. Every earlier held-out figure held the venue constant.

Per class on bout 5 (3078 scored windows):

| class | n_true | precision | recall |
|---|---|---|---|
| advance | 869 | 60% | **40%** |
| retreat | 529 | 69% | 48% |
| neutral | 683 | 68% | 51% |
| walking | 857 | **59%** | 92% |
| lunge | 70 | 32% | 34% |
| parry | 70 | **12%** | **39%** |

**The dominant new-venue failure is `advance` → `walking`: 282 of 869 true advances.**
Nothing else comes close (the next is retreat→advance at 111 and retreat→parry at 104).
These are the two classes separated ONLY by posture — CLAUDE.md, clip era: "walking and
advance are both moving forward, only posture splits them", rescued originally by the
crouch feature (median knee angle, fencing ~140° vs upright ~164°). A different camera
height and piste angle shifts exactly that measurement. This is the most concrete lead
the project has for what breaks at a new venue, and it is one feature deep.

**Parry recall 39% is the highest ever recorded** (bout 4 held out: 21%; the shipped
model historically ~1-12%), at 12% precision on 70 windows. Consistent with the standing
position: parry is not inert, it is imprecise.

**Timeline on bout 5: 55% event precision and only ONE filler event.** Dense labelling
(54% coverage, `walk` marked rather than left as a gap) all but removes the filler
problem that held bout 4 to 38%. But the counts now run the OTHER way — they
UNDER-report, where bout 4 over-reported:

| | advance | lunge | parry | retreat |
|---|---|---|---|---|
| A counted / actual | **28 / 46** | 2 / 13 | **8 / 8** | 9 / 18 |
| B counted / actual | **18 / 59** | 7 / 5 | 4 / 5 | 9 / 15 |

Bout 4 read advance 72/34; bout 5 reads 18/59. **Counts are unreliable in BOTH
directions**, driven by whichever error dominates that footage — false positives on
bout 4, missed advances on bout 5. Do not report a count without the true count beside
it.

#### SHIPPED `action_opp5.pth` — a second venue helps, replicated on two held-out bouts

`extract_continuous.py` on 5.mp4 gives **3078 windows**, taking the continuous corpus from
5352 to **8430** (+58%).

A five-bout model cannot be honestly scored on any of the five, so "does bout 5 help?" was
answered as a **matched A/B**: identical recipe, identical held-out bout, the only difference
being whether bout 5 is in the training set.

| held out | without bout 5 | **with bout 5** | Δ | training windows |
|---|---|---|---|---|
| bout 1 | 74.6% | **76.4%** | +1.8 | 512 → 2521 |
| bout 4 | 67.6% | **71.3%** | **+3.7** | 512 → 1538 |

Trust the bout 4 row: 3822 scored windows, and +3.7 is outside the ±0.6-2.4 seed noise this
project normally sees. Bout 1's +1.8 is marginal alone but points the same way. Contrast the
mirroring result, rejected at +0.5 mean against ±1.5 noise.

Per class on held-out bout 4, precision / recall:

| class | without bout 5 | with bout 5 |
|---|---|---|
| advance | 62% / 65% | **70% / 73%** |
| retreat | 70% / 54% | 70% / **74%** |
| lunge | 60% / 46% | **65% / 56%** |
| walking | 86% / 85% | **92%** / 78% |
| parry | 22% / 21% | **29%** / 19% |

**Bout 5 improved the OTHER venue too**, which was not the expected outcome — the risk was that
a differently-framed broadcast would drag the majority style down. advance +8/+8 and retreat
recall +20 on bout 4 say the opposite: more variety helped generalisation rather than diluting
it.

**Parry is the exception and it went the wrong way on recall** (21% → 19%, precision 22% → 29%).
Bout 5 carries only **70 parry windows against bout 4's 207**, so it dilutes the class. Confirms
that the parry lamp needs parry-DENSE labelling, not more footage in general — more of the same
kind of bout makes parry relatively rarer.

`models/action_opp.pth` is kept for comparison, not deleted, like every checkpoint before it —
and it is the ONLY checkpoint with an honest cross-venue number attached to it (58.1% on bout 5
before bout 5 entered training), so do not delete it.

**Shipped with `--members 5`**, matching `action_opp` and `action_cont`. The default is 4, and
the first `--ship` run produced a 4-member ensemble before this was noticed. The A/B above is
unaffected — all four verify checkpoints are 4-member, so it was like for like — and the
ensemble exists for CONSISTENCY rather than accuracy (measured 2.4x variance reduction), so
shipping one member short of the predecessor would have been a silent regression in the one
property it is there to provide.

**Degeneracy check on the shipped checkpoint** (offline, over all 8430 cached windows; accuracy
is circular because it trained on them, but the class DISTRIBUTION is still diagnostic):

| | advance | lunge | parry | retreat | neutral | walking |
|---|---|---|---|---|---|---|
| predicted | 24% | 5% | 3% | 19% | 14% | 36% |
| true | 23% | 6% | 4% | 18% | 15% | 36% |

No class swallows the others — the failure mode that `lunge` showed before the prior correction
and `advance` showed on bout 3. Training at natural frequencies keeps the output calibrated,
which is exactly why `APPLY_CLASS_PRIOR` could be retired.

**Three source anomalies, resolved and recorded rather than silently fixed:**

1. **The 03:21-03:30 phrase is transcribed TWICE** with slightly different boundaries
   (03:27.169/03:27.869 vs 03:27.083/03:27.950). Same phrase, not a repeat — the
   exchange cannot recur inside 9 s. Keeping both would double-count it and create
   overlapping intervals, which mis-score silently because `truth_at()` returns the
   first match. The second copy is kept; `--keep-first-duplicate` flips it.
2. **`04:04.067 - 04:10.183` runs backwards** — the preceding row ends at 04:06.067.
   Read as `04:06.067` (a 4↔6 slip), the only reading that leaves no overlap. Confirmed
   on the video: 04:08 is a between-phrase reset.
3. **Two 0.233 s intervals** (06:22.784, 07:55.800), shorter than anything in bouts 1-4
   (minimum 0.30 s). Transcribed as written; a 2 s window cannot recover them.

### EVENT TIMELINE + BOUT STATISTICS: works on a bout, not on a broadcast (2026-08-12)

`scripts/bout_timeline.py` collapses the per-window prediction stream (~6 calls/s per
fencer) into discrete events and summarises them per fencer. `src/labels.py` is new:
one parser and one two-track collapse rule, shared by `evaluate_labels.py` and the
timeline, verified identical to the old inline parser on all seven label files.

It reads the probability cache from `evaluate_labels.py`, not the video, so gate
tuning costs milliseconds instead of a 90-minute re-run.

**USE A HELD-OUT CHECKPOINT.** `action_opp.pth` trained on all four bouts and scores
**91.6%** on bout 1 against the honest held-out 74.6%. The script now refuses to stay
quiet unless the cache filename says `held` or `verify`. New checkpoint for this work:
`verify_h4.pth` (bouts 1-3 only, opponent, last pool) → bout 4 per-window **67.6%**,
matching the 66.5% offline estimate.

| held-out | event precision | strict recall | filler events | per-window |
|---|---|---|---|---|
| bout 1 (104 s clean segment, 80% labelled) | **68%** | 52% | **0** | 74.6% |
| bout 4 (26 min broadcast, 23% labelled) | **38%** | 36% | **87 of 246** | 67.6% |

**The gap between those two rows is the whole finding.** On a clean bout segment the
timeline is usable. On a full broadcast roughly one event in three is captioning a
replay, and no operating point fixes it: across the entire gate grid bout 4's filler
share never falls below ~26%, even at 73% precision where only 24 events survive from
26 minutes. This is the same unsolved filler problem that killed the geometric gate,
now showing up where a viewer can actually see it.

**COUNTS ARE NOT TRUSTWORTHY, and it is FALSE POSITIVES rather than over-segmentation.**
Counts run 2-3x high:

| bout 4, held out | advance | lunge | parry | retreat |
|---|---|---|---|---|
| fencer A counted / actual | **72 / 34** | 17 / 29 | 17 / 22 | **6 / 19** |
| fencer B counted / actual | **47 / 19** | 25 / 32 | **0 / 24** | **62 / 31** |

Note parry 17 for A and **0** for B against a true 22/24 — the class is not merely
weak, it is one-sided. Report event counts with this caveat or not at all.

**THE SEGMENTER IS NOT THE PROBLEM — the counting ceiling is ~90-100%.** Worth knowing
before anyone tries to fix counting with better merging logic. Feed the segmenter
PERFECT predictions (the truth resampled at the demo's 167 ms cadence) and it recovers
the counts almost exactly:

| bout 4, perfect input | advance | lunge | parry | retreat |
|---|---|---|---|---|
| fencer A counted / actual | 29 / 34 | **29 / 29** | 21 / 22 | 17 / 19 |
| fencer B counted / actual | 16 / 19 | **32 / 32** | **24 / 24** | 28 / 31 |

The only losses are consecutive same-class actions labelled back-to-back with no gap,
which merge into one run — about 10% of advances and retreats, and genuinely
unrecoverable without boundary supervision. Everything else is exact.

**So the model has no concept of "one advance", and it turns out it barely needs one.**
`extract_continuous.py` labels each window by `truth_at(slot, now)` — whatever interval
contains its NEWEST frame — so training carries no boundary signal at all. The model is
a per-window state classifier answering "is this advance-ish right now?" six times a
second. Because Aaron leaves small gaps between consecutive actions, that state stream
is enough to count from.

The bad counts are therefore the model's error rate, not a missing notion of an event:

- **80% of raw events (723 of 906) sit on no true interval of their own class.** These
  are the inflators — filler and wrong-class calls.
- Fragmentation is the minor term: 52% of true intervals get exactly one correct run,
  16% get two or more, 32% get none.

**Practical consequence: fixing counts means suppressing false positives and steadying
the output inside an action, NOT teaching the model where actions begin and end.** That
is consistent with smoothing cutting bout 4's count error 158 → 113 while barely moving
recall.

**Three gates measured, and the two bouts DISAGREE — do not generalise from one:**

| gate | bout 1 (clean) | bout 4 (broadcast) |
|---|---|---|
| min duration | works, 43%→80% precision | works, 21%→73% precision |
| min confidence | **flat** (70/70/68/71%) | **strong** (32/34/38/42/59%) |
| probability smoothing | **loses** (recall 52→44→37%) | **wins** (prec 38→47%, count error 158→113) |

Duration is the only axis that behaves the same on both. Confidence and smoothing
help on the long noisy broadcast and cost on the clean segment, so both default to
conservative settings (`--min-conf 0.55`, `--smooth 0`) and should be raised for
broadcast footage. *An earlier version of this entry called smoothing a clean null
on bout 1 evidence alone — the eighth instance of one bout talking, caught only by
re-running on bout 4.*

**Scoring honestly required splitting "phantom" in two.** Aaron described bout 4's
gaps two ways — "if there is a gap it probably means there's no arm/blade thing" and
"the gaps in intervals aren't gaps in fencing, it's just that the broadcast has other
stuff". Measured, both are true of different gaps: **median hole 5.3 s, but 32 holes
over 15 s carrying 906 of the 1190 gap-seconds**. So events in unlabelled time split
into `filler` (hole ≥15 s — a replay, a real error) and `in-pause` (short hole inside
live fencing — ambiguous, excluded from precision rather than guessed at). Treating
every unlabelled call as a hallucination overstates the error.

Two further metric traps handled in the script:

- **Recall is reported twice.** "Any overlap" overcredits long events: on bout 1 four
  events each straddled two true intervals, so 8 of 23 credits came from events that
  had localised nothing — one 9 s "advance" spanning a whole approach sequence
  collects a credit per step inside it. "≥50% covered" is the honest column.
- **`4_probs.npz` and friends are STALE, not circular.** They predate `action_cont`,
  so they are the old clips-only checkpoint. A timeline built on them reads 15%
  precision on bout 4 where the current model gets 38%. Check cache mtimes against
  checkpoint mtimes before quoting anything.

### GAPS IN THE LABELS ARE FINE — corrected advice

Aaron: "the gaps in intervals aren't gaps in fencing, it's just that the broadcast has other
stuff." That **reverses** the "label contiguously, no gaps" advice given earlier the same day.
That advice assumed gaps meant SKIPPED FENCING, which would bias the model's implicit prior now
that CLASS_PRIOR is retired and training runs on natural frequencies. Gaps that are replays,
crowd shots and graphics contain no fencing to label, so the labelled fraction is a faithful
sample of fencing time and the natural-frequency argument is satisfied. **Label the fencing, skip
the filler.**

### Video quality: `3.mp4` is FINE — a retracted claim

A diagnostic reported "46.9% duplicate frames" and concluded the source was laggy. The threshold
(0.35 on a 320×180 grayscale difference) was measuring **low motion, not repeats**. There are
**zero exact duplicate frames**; it is 34% near-static idle content, which is what fencing
between phrases looks like. The video is fine.

---

## DEAD ENDS — all measured, do NOT re-run

### Architecture / training

- Per-frame world-motion channels into the LSTM (advance recall 78→60)
- Shorter SEQ_LEN with all classes sliced (24f/30f → advance recall 51%/73%)
- Pre-padding so the action ends at the window end (83→63), and filling that pad with real
  neutral context (→57)
- x-axis stance augmentation (val 86.3→82.9, advance 83→67)
- Rate-normalised length-invariant sum features (val +0.8, advance −13)
- Within-window RELATIVE stance/crouch/reach (val 84.6→85.2, bout advance 14→10)
- Mirror augmentation (see the 2026-08-09 re-test: +0.5 against ±1.5 noise)
- 2-layer LSTM; reseeding / best-of-N
- Ensembling the per-frame model (parry → 0%)
- EM prior estimation at runtime (Saerens et al.) — 19.2% → 14.6%
- `_first_mover` leg-order feature — clip-start artifact, reverted

### Demo-side

- Shorter windows, or adding advance to `FAST_CLASSES` — advance is flat at 3-6% across
  60f/40f/25f/18f, so it is not being diluted; shortening just converts retreat into lunge and
  parry
- Continuity-based box selection and its variants (35.7 / 35.8 / 34.1% vs 45.9%) — the
  absorbing-error mode, later fixed properly by memoryless x-order assignment
- The geometric "is this fencing?" gate (36% precision vs 23% base)

### Features from Aaron's biomechanics (2026-07-31), tested by AUC on labelled clips

- **`stance_ratio`** (ankles / leg length) had the BEST single-feature AUC found for
  advance-vs-lunge (0.91 vs raw stance 0.87, crouch 0.77) and perfect lunge-vs-walking (1.00) —
  **and still made the model worse** (advance 88→80). This is the canonical "AUC is not
  sufficiency" case.
- **`front_knee`** (lunge 103° vs advance 137°, AUC 0.80) — real but redundant with `crouch`,
  which uses min(left, right) and so already picks the front knee during a lunge.
- **Front-foot ACCELERATION** (the lunge "kick") — AUC 0.54, nothing, probably because a lunge
  extension is ~5 frames at 30 fps.
- **Back-leg extension** — AUC 0.58; the back knee is 177-178° in EVERY class including neutral
  and walking, so "more extended than before" is not recoverable from 2D pose.

### Blade

- Blade DETECTION as an action-model input — 1-2% frame coverage, killed by motion blur
- **Blade motion energy AS A MODEL FEATURE — dead, and properly killed this time.** The signal
  is real at INTERVAL level (0.79 over 70 parries; the original n=6 null was underpowered) but
  chance at WINDOW level (0.53), and as a 7th feature a shuffled control reproduces the entire
  gain on one of two holdouts. Parry recall went the wrong way on the bout with the data.
- The ORIENTED-STRIP rewrite of blade energy (v2): 0.66 vs the crude axis-aligned box's 0.79.
  Both named flaws were genuinely fixed and it still lost; the fat box's tolerance of
  wrist-keypoint error is the untested explanation.

---

## Secondary lever, still open

More transient-class clips. Not for any domain gap, but because 35 advance + 33 retreat + 48 lunge
clips cannot determine a boundary in a 128-dim representation — the 0.4%-34% seed spread IS that
underdetermination. `advance` is the smallest class (35 clips, not sliceable; walking gets 232
windows from 27 clips), so it is cheapest to improve. Lower priority than continuous labels now
that continuous training works.
