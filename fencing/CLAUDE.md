# FenceVision — Claude Code Project Context

## What this project is

FenceVision is a computer vision and machine learning system for analyzing fencing (the sport) video. It is being built as a high school portfolio project targeting college admissions. The system must be technically impressive, well-structured, and produce a visually compelling real-time demo.

The project is being built in Python inside VSCode using Claude Code. The developer is a high school student who fences and has some programming experience but is not yet an expert in ML or CV. Explain your reasoning clearly. Prefer readable, well-commented code over terse cleverness. When multiple approaches exist, explain the tradeoff and recommend one.

---

## System architecture (already decided — do not redesign)

The system has three parallel input streams that merge into a touch predictor:

```
Raw video
    ├── MediaPipe Pose Estimation → keypoint sequences (.npy)
    └── YOLOv8 Blade Detector    → blade tip trajectory

Keypoint sequences
    └── Action Recognition LSTM  → action class probabilities

[Keypoints + Blade tip + Action probs] → Touch Predictor → who scores?

All streams → OpenCV + Streamlit overlay → real-time demo
```

### Why these choices were made
- **MediaPipe** over OpenPose: runs in real time on a laptop CPU, no GPU needed, Python-native
- **YOLOv8** for blade: pretrained backbone means we only need ~300-500 labeled frames, not thousands
- **LSTM** over Transformer for action recognition: simpler to implement and debug with a small dataset
- **Streamlit** for UI: fast to build, looks professional, no frontend experience needed
- **Roboflow** for blade labeling: free tier, exports YOLOv8-ready dataset with YAML config automatically

---

## Project file structure

Maintain this structure exactly. Do not reorganize it.

```
fencing-ml/
├── CLAUDE.md                   ← this file
├── requirements.txt
├── app.py                      ← Streamlit entry point
├── data/
│   ├── raw_video/              ← original footage (.mp4)
│   ├── clips/                  ← action clips organized by class
│   │   ├── lunge/
│   │   ├── parry/
│   │   ├── advance/
│   │   └── retreat/
│   ├── keypoints/              ← .npy files output by pose pipeline
│   │   ├── lunge/
│   │   ├── parry/
│   │   ├── advance/
│   │   └── retreat/
│   ├── blade_frames/           ← extracted frames for Roboflow labeling
│   └── blade_dataset/          ← downloaded from Roboflow after labeling
├── models/
│   ├── action_lstm.pth         ← saved action recognition weights
│   └── blade_yolo/             ← YOLOv8 training output directory
│       └── fencing_blade_v1/
│           └── weights/
│               └── best.pt
├── src/
│   ├── __init__.py
│   ├── pose_pipeline.py        ← MediaPipe extraction (single frame + batch)
│   ├── blade_detector.py       ← YOLOv8 inference + tip extraction
│   ├── person_detector.py      ← YOLOv8n (COCO) person bounding boxes, assigns fencer A/B
│   ├── action_model.py         ← Dataset class + LSTM model + training loop
│   ├── touch_predictor.py      ← feature fusion + touch prediction model
│   ├── utils.py                ← shared helpers (normalization, visualization)
│   └── train_action.py         ← standalone training script
├── notebooks/
│   └── exploration.ipynb       ← for experimentation and visualization
└── scripts/
    ├── extract_blade_frames.py ← samples frames from raw video for labeling
    └── process_clips.py        ← batch-runs pose pipeline on all clips
```

---

## Phase-by-phase implementation plan

Build in this order. Do not skip ahead. Each phase produces something testable before moving to the next.

### Phase 0 — Environment setup ✓ BUILT

Deps in `requirements.txt`; verify with `scripts/smoke_test.py`. Runs on Python 3.14.

---

### Phase 1 — Data collection scripts ✓ BUILT

`scripts/extract_blade_frames.py` samples ~400 frames total from `data/raw_video/` into
`data/blade_frames/`, named `{video_stem}_f{frame_index:06d}.jpg`. Skips an interlaced
original when a `_deinterlaced` twin exists.

Worth keeping: sampling a fixed BUDGET beats every-Nth-frame. Every-5th on full matches
gives ~9,300 frames — far past the 300-500 needed, and highly redundant since a blade
barely moves between frames at 6 fps. Target-count sampling hits the labeling budget and
maximises pose diversity.

---

### Phase 2 — Pose estimation pipeline

✓ BUILT: `src/pose_pipeline.py` (extract from frame / from video, plus
`extract_keypoints_and_pan_from_video`), `src/person_detector.py`,
`scripts/process_clips.py` (writes `data/keypoints/<action>/<clip>.npy`).

MediaPipe Pose detects only ONE person per frame and fencing video has two, so
`person_detector.py` crops each fencer first using pretrained YOLOv8n (COCO class 0);
`YOLO("yolov8n.pt")` auto-downloads, no training needed.

**Train/serve note:** `process_clips.py` runs pose on the FULL FRAME, while the demo runs it
on a tight person crop. Checked and it is not a problem — cropping training clips actually
RAISES jitter (0.0122 → 0.0158) rather than matching the demo's 0.008, and subject sizes are
comparable. Do not "fix" it.

**MediaPipe config (Tasks API — `mp.solutions` was removed in 0.10.14+).**
`download_pose_model()` fetches `pose_landmarker_full.task` (~27 MB, equivalent to the old
model_complexity=1). Options:
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

Use `RunningMode.IMAGE` for `extract_keypoints_from_frame` (no temporal context).
Use `RunningMode.VIDEO` for `extract_keypoints_from_video` (temporal tracking); pass `timestamp_ms = int(frame_index * 1000 / fps)` to each `detect_for_video()` call.

**Post-processing rules (video path — all measured-and-revised 2026-07):**
1. Any keypoint with visibility < 0.5 → replace its x,y,z with the previous frame's values (carry-forward interpolation). For the first frame, use zeros.
2. 3-frame median filter on x,y over the clip — the tracker occasionally teleports a joint for one frame (measured spikes up to ~100x body size); the median kills those without smearing real motion.
3. Normalize x,y relative to the hip midpoint (landmarks 23/24 average). After normalization the hip midpoint is (0, 0).
4. Scale by the per-clip MEDIAN body height (shoulder-mid to ankle-mid). History: spec said per-frame shoulder width → collapsed to ~0 for side-on fencers (~300x blowups); then per-frame torso length → foreshortens during lunges and crushed the leg-spread signal. A single per-clip height is stable through every action.
5. Do NOT normalize z or visibility — keep them raw.
6. The video extractor also returns a per-frame **motion track** of shape (n, 2) = [background pan (phase correlation on border strips), raw hip-x before centering] via `extract_keypoints_and_pan_from_video`. World travel = in-frame hip-x + camera pan; this is what makes advance vs retreat learnable (pan alone works only when the camera tracks tightly; the hip-x term catches looser broadcasts). `process_clips.py` saves it as `<clip>.pan.npy` (still that name) beside each keypoint file.

**Verification step:** After running the pipeline, write a quick visualization that draws the MediaPipe skeleton back onto a sample frame using `mp.solutions.drawing_utils` to confirm keypoints look correct.

---

### Phase 3 — Blade detection (YOLOv8 fine-tuning)

**`src/blade_detector.py`** must expose:

```python
def load_blade_model(weights_path: str) -> YOLO:
    """Loads and returns a YOLOv8 model from the given weights file."""

def get_blade_tip(frame: np.ndarray, model: YOLO) -> tuple[float, float] | None:
    """
    Runs inference on a single BGR frame.
    Returns (tip_x, tip_y) as the centroid of the highest-confidence blade detection.
    Returns None if no blade is detected.
    Tip = bounding box centroid: tip_x = (x1+x2)/2, tip_y = (y1+y2)/2.
    Note: this is an approximation — the true tip is at one end of the blade, but
    without knowing which end faces the opponent, the centroid is the least-wrong
    single point. A future improvement is a keypoint model trained to locate the tip.
    """

def get_blade_tip_trajectory(video_path: str, model: YOLO) -> list[tuple | None]:
    """
    Runs blade detection on every frame of a video.
    Returns a list of (tip_x, tip_y) or None per frame.
    """
```

**Training script (inline, not a separate file):**
The user will run training from the terminal. Provide the exact command and a short Python snippet using `ultralytics` API:

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.train(
    data="data/blade_dataset/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    project="models/blade_yolo",
    name="fencing_blade_v1",
    patience=10   # early stopping
)
```

**Roboflow labeling instructions to give the user:**
1. Go to roboflow.com, create a free account
2. New Project → Object Detection → name it "fencing-blade"
3. Upload all images from `data/blade_frames/`
4. Draw bounding boxes around the entire blade (not just the tip) — class name: `blade`
5. Apply auto-augmentation (flip, rotation, brightness) in Roboflow to expand the dataset
6. Export → YOLOv8 format → download ZIP → unzip into `data/blade_dataset/`

**Blade tip velocity computation (for touch predictor input):**
```python
def compute_tip_velocity(trajectory: list) -> list[tuple[float, float]]:
    """
    Given a list of (x, y) or None tip positions, returns (dx, dy) per frame.
    None frames get (0.0, 0.0) velocity.
    """
```

---

### Phase 4 — Action recognition model ✓ BUILT

**`src/action_model.py` is the source of truth — read it, don't re-spec it here.** An earlier
version of this section carried a full spec that drifted badly out of date (it still said 4
classes, HIDDEN_SIZE 64, 4 features, "2-layer LSTM"). Reality: 6 classes, HIDDEN_SIZE 128,
6 features, ONE LSTM layer — and 2-layer was measured WORSE. A stale spec under a
"do not contradict" heading is worse than no spec, so it is gone.

Only the non-obvious decisions live here:
- Face landmarks 1-10 stay in the tensor. MediaPipe reports them at visibility ~1.0 even
  under a fencing mask (it infers them from head shape), so the carry-forward step never
  zeroes them; they just track the head.
- Engineered features beat keypoints alone by +23 pts (51%→74%) — the LSTM cannot rediscover
  them from 132 channels at this dataset size. Combining hip-x with camera pan (vs pan alone)
  lifted advance/retreat direction accuracy 84%→94%.
- HIDDEN_SIZE 64→128 was worth +1.5 pts once the set reached 488 windows; at ~80 clips the
  smaller net won. Re-tune capacity when the dataset changes size substantially.
- Long clips are trimmed from the END (keep the first SEQ_LEN frames) because an action's
  initiation is its most distinctive part. **Caveat:** this is also why `_first_mover` failed
  on video — see the metric warning below. Clip-start alignment is a training-set property.
- `src/train_action.py` is a thin wrapper that plots the loss curve.
- After training, print `sklearn.metrics.classification_report`; flag any class under 70%
  recall as needing more clips.

---

### Phase 5 — Touch predictor

**`src/touch_predictor.py`**

#### Feature vector construction
For a window of recent frames, compute:
- Action class probabilities from the action model: 4 values per fencer = 8 total (2 fencers)
- Blade tip velocity (dx, dy) for each fencer: 4 values
- Three key joint angles per fencer (elbow, knee, hip): 6 values
- Total input: 18 features

```python
def build_feature_vector(
    kp_a: np.ndarray,        # keypoints for fencer A, shape (SEQ_LEN, 33, 4)
    kp_b: np.ndarray,        # keypoints for fencer B
    action_probs_a: np.ndarray,  # shape (4,)
    action_probs_b: np.ndarray,  # shape (4,)
    blade_velocity_a: tuple,     # (dx, dy)
    blade_velocity_b: tuple,     # (dx, dy)
) -> np.ndarray:             # shape (18,)
    """Assembles the 18-feature input vector for the touch predictor."""
```

#### Joint angle computation
```python
def compute_joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Computes angle at joint B formed by points A-B-C.
    Each point is a (x, y) array.
    Returns angle in degrees.
    """
```

Use these MediaPipe landmark indices for angles:
- Elbow angle: landmarks 11-13-15 (left) or 12-14-16 (right)
- Knee angle: landmarks 23-25-27 (left) or 24-26-28 (right)
- Hip angle: landmarks 11-23-25 (left) or 12-24-26 (right)

#### Model
```python
class TouchPredictor(nn.Module):
    """
    Simple 3-layer MLP.
    Input: (batch, 18)
    Output: (batch, 3) — logits for [fencer A scores, fencer B scores, no touch]
    Final activation: softmax
    Must include a "no touch" class because in real-time inference (Phase 6) this
    model runs on every frame, not just touch windows — without a neutral class it
    would hallucinate a scorer constantly.
    """
```

#### Labeling instructions to give the user
Touch labels are simple: watch a bout, write down timestamps and which fencer (A or B) scored, plus non-touch windows labeled "N". Store in a CSV:
```
timestamp_start,timestamp_end,scorer
00:00:05,00:00:07,A
00:00:15,00:00:17,B
00:00:09,00:00:11,N
```
The "N" (no touch) class is required — the model must learn what neutral fencing looks like, not just touches. Aim for roughly equal numbers of A, B, and N windows. Extract features for each labeled window and train on those.

---

### Phase 6 — Real-time Streamlit overlay

**`app.py`** — the main demo application.

**Layout:**
- Left column: live video feed with skeleton overlay + blade tip dot + action label
- Right column: 
  - Current action label (large text)
  - Action confidence bar chart (using `st.bar_chart`)
  - Touch probability gauge (fencer A% vs fencer B%)
  - Session stats (touches detected, most common action)

**Video source:** Support both webcam (`cv2.VideoCapture(0)`) and a video file path passed as a command-line argument or Streamlit text input.

**Overlay drawing (in `src/utils.py`):**
```python
def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """Draws MediaPipe skeleton connections onto frame. Returns annotated frame."""

def draw_blade_tip(frame: np.ndarray, tip: tuple | None) -> np.ndarray:
    """Draws a green circle at the blade tip. Returns annotated frame."""

def draw_action_label(frame: np.ndarray, action: str, confidence: float) -> np.ndarray:
    """Draws action label and confidence in top-left corner. Returns annotated frame."""
```

**Performance note:** The Streamlit loop should target 15fps minimum. If pose estimation is too slow, run it every 2nd frame and use the previous result on skipped frames.

---

## Key technical facts (do not contradict these)

### MediaPipe landmarks (0-indexed)
- 0: nose, 11: left shoulder, 12: right shoulder
- 13: left elbow, 14: right elbow, 15: left wrist, 16: right wrist
- 23: left hip, 24: right hip, 25: left knee, 26: right knee
- 27: left ankle, 28: right ankle
- Landmarks 1-10: face — physically hidden by the fencing mask, but MediaPipe still places them with visibility ~1.0 (verified on sample frames; it infers them from head shape). They will NOT be zeroed by the carry-forward logic. Keep them in the feature vector (INPUT_SIZE stays 132) — they track the head and are harmless, just don't build handcrafted features from them.

### Fencing-specific challenges
- Fencing masks cover the face, but MediaPipe confidently hallucinates landmarks 1-10 anyway (visibility ~1.0). Don't rely on visibility to filter them; just never use them as action features.
- The sword arm (typically right arm for right-handed fencers) is the primary action signal for bladework.
- Footwork signal: relative distance and velocity between landmarks 27 and 28 (ankles).
- Fast actions (fleche, ballestra) cause motion blur. If keypoint visibility drops below 0.3 on >50% of landmarks in a frame, flag it as a low-quality frame and consider skipping it for training.
- When two fencers overlap in frame, MediaPipe may assign keypoints to the wrong person. For now, crop each fencer into separate frame regions using a person bounding box before running pose estimation.

### YOLOv8 model sizes (use nano for speed)
- yolov8n.pt: fastest, use for real-time inference and fine-tuning with small datasets
- yolov8s.pt: slightly more accurate, still fast — use if nano accuracy is insufficient
- Do not use medium/large/xlarge — overkill for this dataset size

### Action class definitions — SIX, and the list is LOCKED (Phase 5 input width depends on it)
- **advance**: step forward, FRONT foot then rear foot. Ankle gap widens then closes quickly.
- **retreat**: step backward, REAR foot then front foot — the mirror.
- **lunge**: explosive forward extension of sword arm and front leg; front knee drives to
  ~103° (vs ~137° in an advance), body lowers, ankle-gap/leg-length ~1.9 (vs ~1.0). Usually
  ends a series of advances, though it can stand alone or be followed by more advances.
- **parry**: defensive blade deflection, small fast wrist motion. Hardest class — its true
  signature is a lateral blade sweep, which is UNRECOVERABLE from side-on 2D (image-plane
  direction and MediaPipe z both measure identical to lunge). Would need 3D pose, a second
  camera, or the blade detector wired into the action model (it is not, currently).
- **walking**: normal upright walking between phrases — legs extended, ankles close together.
  Separate from neutral because it translates forward and would otherwise fire `advance`.
- **neutral**: continuous seconds of relative stillness (en-garde, bounces, pauses).

Not classes: **extension** lives on as the arm-reach FEATURE (every lunge contains one).
fleche / riposte / en-garde are out of scope — riposte and priority are the RULE layer in
Phase 5, not learned classes.

The footwork descriptions above are mechanism, not decoration: they are the most productive
source of features found so far (see the biomechanics entries under "Dead ends" and the
`_first_mover` warning). Anything derived from them must be checked for window-position
invariance before shipping.

---

## Coding conventions

- Python 3.10+ — use `match/case`, `|` union types, and `X | None` instead of `Optional[X]`
- Type hints on every function signature
- Docstrings on every class and public function (one-line summary + Args/Returns if non-obvious)
- No magic numbers — define constants at the top of each file
- Use `pathlib.Path` for all file paths, not `os.path`
- Prefer `numpy` vectorized operations over Python loops for array math
- When printing progress, use `tqdm` for loops over files or frames
- Tests: write at least one `assert`-based smoke test at the bottom of each `src/` file under `if __name__ == "__main__":`

---

## What "done" looks like for the portfolio demo

The final demo is a short video (60–90 seconds) showing:
1. Real fencing footage playing in the Streamlit app
2. Skeleton overlay rendered on both fencers
3. Blade tip tracked with a colored dot
4. Action label updating in real time ("Lunge!", "Advance", etc.)
5. Touch probability bar updating as a phrase develops

This video is the thing that goes in the portfolio and GitHub README. Everything else (code quality, architecture, writeup) supports it.

---

## Current status

**Updated 2026-07-23 — Phase 4: SIX classes shipped (advance, lunge, parry, retreat,
neutral, walking), h128 hybrid LSTM + 6 engineered features, 213 clips / 488 windows,
10-seed 85.1% ± 6% (lun 98 / wal 90 / ret 85 / neu 80 / adv 65 / par 65). Extension is a
FEATURE (arm-reach), not a class.**

Key findings baked into the pipeline (don't re-learn these the hard way):
- The broadcast camera PANS to follow the fencer, so keypoints alone cannot tell advance
  from retreat — hip drift in-frame is near zero and points both ways. True travel is
  recovered from background pan (phase correlation) x facing direction (nose vs hips).
- Clips contain fencers facing both directions; without the pan-x-facing feature those
  classes are mirror-ambiguous.
- Per-frame scale references wobble with pose (shoulders collapse side-on, torso
  foreshortens in lunges); per-clip median body height is the stable choice.
- Tracker teleport spikes forged fake wide stances in advance clips until the 3-frame
  median filter; that plus a stance-width p90 feature is what makes lunge separable.
- Walking and advance are both "moving forward" — only posture splits them. The crouch
  feature (median knee angle: fencing ~140° vs upright ~164°, 84% separable) plus
  inverse-frequency class weighting (slicing makes walking/neutral many more windows)
  rescued advance after the 6-class flip.
- The 6 engineered features (`_engineered_features`): net-forward, stance-width p90,
  wrist-speed p90, total-travel, arm-reach p90 (=extension, drives Phase 5 priority),
  crouch.
- Improvement dead ends (measured, don't re-run): mirror augmentation trades parry for
  advance (no net win); velocity channels never help; 2-layer LSTM worse. h64→h128 at
  488 windows was worth +1.5 pts. Parry's true signature (lateral blade sweep) is
  UNRECOVERABLE from side-on 2D — image-plane direction and MediaPipe z both measured
  identical to lunge. Needs 3D pose or a second camera; document as a known limitation.
- Confusion (10-seed): advance leaks to parry/lunge/walking; parry leaks to retreat;
  the biggest raw error mass is neutral↔walking swaps, which are downstream-harmless
  (both mean "no priority action").

---

## READ THIS BEFORE TRUSTING ANY METRIC HERE

Three times this project has "improved" on a number and got worse on video. All three were
the same mistake: **measuring a property of how the CLIPS WERE CUT, not a property of
fencing.** Training clips are hand-trimmed single actions; inference is a sliding window
over continuous video. Anything that correlates with clip construction will validate
beautifully and then evaporate.

1. **Zero padding.** Clip length tracks class (lunge/parry 24f → 60% zeros, advance 46f,
   retreat 48f, sliced neutral/walking 0%) and the head mean-pooled across it. Re-padding
   by holding the last frame dropped in-sample accuracy 85% → 53% and flipped 40% of calls.
   Video never pads. FIXED by masked pooling.
2. **Bout label MIX is a distribution, not an accuracy.** There are no ground-truth labels
   for the bout, so a mix drifting toward what a bout "should" look like proves nothing
   about whether labels land on the right events. A reported "advance 28 → 62" gain turned
   out to be false positives: advance was firing on a fencer half off-screen (hip_x=0.00)
   and on stoppage-walking. Aaron caught it by eye; the metric could not.
3. **Clip START alignment** (`_first_mover`, 2026-07-31). Leg-order scored advance recall
   70% → 88% on held-out clips, shipped, and collapsed advance to 5 calls/fencer on video.
   Training clips begin at the action; sliding windows begin mid-stride. Measured:
   advance clips lean front-first 26%/17%, bout windows are 26%/24% — a coin flip. REVERTED.

**Practical rules:** score candidates on `val@hold` (clips re-padded by holding the last
frame) as a continuous-video proxy, never on val alone. Bout-mix numbers are noisy — two
runs of the IDENTICAL config gave advance 8% vs 14%, lunge 44% vs 32% on pure training
noise, and per-seed bout advance spans 0.4%–34% at 82–92% val. Do not chase small deltas.
**Nothing more can be honestly validated without ground-truth labels on continuous
footage — that is the blocking task** (see "Next step" below).

---

**Fencer-B retreat bug (2026-07-24) — ruled out, don't re-run.** B read `retreat` 134/237
windows through a stoppage (clock frozen at 2:34). NOT the engineered features (ablating each
to its neutral median leaves 134→148; pathway swap localises it to the LSTM path). NOT
`nose_dir` instability (4 flips/bout; B's net-forward correctly balanced 82+/92−). NOT pose
quality (A vs B identical on jitter, frozen joints, visibility). NOT facing coverage (every
class ≥11 clips on its thinner side). NOT fixable by mirroring (both variants lose: 81.8% and
80.6% vs ~85–87%; it just moves damage between facings, and handedness is real signal). There
IS a modest out-of-sample facing gap, 91% facing right vs 84% left — too small to explain a
134:1 skew.

**There is NO domain gap — do not re-derive this.** An earlier claim that training clips were
"textbook-wide" vs "compact" match footwork was wrong twice over: Aaron's clips are themselves
cut from broadcast matches (same 1920x1080 / 29.97 fps), and the comparison averaged ALL bout
windows when most are idle/stoppage while training clips are pure action. Restricted to ACTIVE
bout windows (top-quartile travel) they line up: stance 0.58 vs 0.54, crouch 0.55 vs 0.57,
arm-reach 0.21 vs 0.21. Windows inside the advance band on all six features go 19% → 40%.
Also NOT pose quality: re-extracting training clips through the demo's person-crop RAISES
jitter (0.0122 → 0.0158). No re-extraction needed.

**ROOT CAUSE — advance loses INSIDE its own feature region.** Of 89 bout windows sitting
inside the training advance band on all six features, the model says lunge 55 (62%), retreat
30 (34%), advance 4 (4%). Median probability gap to the winner is 0.341 — far too wide for
threshold tuning. It is the decision boundary, not coverage, and most likely the task is
ill-posed at this window size: a 2 s window during an exchange holds step-step-lunge-recover
while the classes are defined on single-action clips, so mixed windows fall to whichever class
has the loosest boundary.
**SHIPPED (2026-07-29).** Two changes, and neither one fixes advance — be clear about that.
- *Masked pooling* (`ActionLSTM.forward(..., lengths)`). Pools over real frames only, so the
  padding artifact above is gone. Isolated 6-seed test: val@hold 82.3% → 85.0%, padding-style
  gap 4.0 → 0.8 pts, costing 0.6 pts on the zero-padded score that was partly measuring the
  artifact. `lengths=None` still means "pool everything", so old callers keep working. Note
  the demo's SHORT window (25 real frames padded to 60, 58% zeros) fed that artifact hardest,
  so `parry` behaviour shifts most after this change.
- *`ActionEnsemble` + `load_action_model()`*, 5 members at `models/action_lstm.m*.pth`. For
  CONSISTENCY, not accuracy: single checkpoints at equal val accuracy land anywhere from
  advance=6%/lunge=50% to advance=25%/lunge=7% on the same bout. Measured 2.4x less
  out-of-domain variance (sd 4.1% → 1.7%). Averages PROBABILITIES not logits (independently
  trained members are not calibrated against each other). Falls back to the single checkpoint
  when the members are absent.

**STILL OPEN — `lunge` is systematically over-predicted (~42% of bout windows).** A real bout
should be nearer 10-15%. This is NOT a bad checkpoint: 5 ensemble members spanning 7%-50%
still average 42%. It is also not the features — bout active windows sit at wrist-speed 0.06
and reach 0.21 against train lunge's 0.17 and 0.28, so the engineered features do NOT say
lunge; the LSTM path does. The cause is most likely that the task is ill-posed at this window
size: a 2 s window during active fencing holds step-step-lunge-recover, the classes are
defined on single-action clips, so mixed windows fall to whichever class has the loosest
boundary. That reading is what motivated the per-frame model below.

### Shipped (2026-07-29 → 07-31) — none of it fixes advance on video

- **Masked pooling** (`ActionLSTM.forward(..., lengths)`) — pools real frames only, killing
  the padding artifact. val@hold 82.3% → 85.0%, padding-style gap 4.0 → 0.8 pts.
  `lengths=None` still pools everything, so old callers work. The demo's 25-frame SHORT
  window (58% zeros) fed the artifact hardest, so `parry` shifted most.
- **`ActionEnsemble` + `load_action_model()`** — 5 members at `models/action_lstm.m*.pth`.
  For CONSISTENCY, not accuracy: single checkpoints at equal val land anywhere from
  advance=6%/lunge=50% to advance=25%/lunge=7%. Measured 2.4× less out-of-domain variance
  (sd 4.1% → 1.7%). Averages PROBABILITIES not logits. Falls back to the single checkpoint.
- **`ActionFrameLSTM` / `--frame-model`** — one label per FRAME, so a window can hold an
  advance and a lunge instead of being forced to pick. No new annotation needed (each clip
  is one action, so every frame carries its label); 488 windows → ~20k supervised frames.
  12 seeds: bout advance 9.7% → 14.3%, lunge 42.3% → 31.0%, for ~3 pts of held-out accuracy.
  **Do NOT ensemble it** — 5 members gave the best lunge figure anywhere (23%) but advance
  16% → 8% and parry 5% → **0%**, a class gone. Averaging dilutes BRIEF actions' probability
  peaks, so persistent classes take every frame. Helps the window model, harms this one.
- **`MAX_FROZEN_FRAC` gate** (demo) — refuses to classify a window whose skeleton is >25%
  carried-forward. When a fencer leaves frame the visibility fallback holds joints in place
  and the model confidently labels missing data; this removed ALL 16 of fencer B's `advance`
  calls, every one a frozen skeleton.

### Open bugs

**⚠ SCOPE CORRECTION (2026-08-02) — the two bugs below are BOUT-2-SPECIFIC, not systematic.**
Everything in this file up to this date was measured on ONE 40 s segment of
`Bout #2 without break.mp4`. A second segment (`Bout #1 without break (1).mp4`, ~5:00 in)
behaves completely differently on the same checkpoints:

| | bout 2 | bout 1 |
|---|---|---|
| lunge share | **43%** | **13%** |
| fencer B advance | 9 | 29 (more than A) |
| fencer B neutral | 1 | 86 (its top label) |

So `lunge` over-prediction and the A/B asymmetry do NOT reproduce. Bout 1's mix even looks
like a plausible bout (walking+neutral 49%, retreat 16%, lunge 13%, parry 11%, advance 10%).
Measured differences: bout 2 is a TIGHTER shot (fencer height 0.463 vs 0.391 of frame) with
far more reliable two-fencer detection (1.83 vs 1.19 detections/frame).
**Do not conclude bout 1 is "correct"** — there is no ground truth for it either, and a
plausible histogram is exactly what produced the false advance result before. What IS
established is that behaviour varies enormously between bouts, so any single-bout number
here is a sample of one. Re-check on both before believing anything.

- **`lunge` over-predicted on bout 2 (~42%; bout 1 gives 13%).** Within bout 2 it is not a
  bad draw — 5 members spanning 7–50% still average 42% — and not the features either
  (active windows sit at wrist-speed 0.06 / reach 0.21 vs train lunge 0.17 / 0.28), so
  within that footage it is the LSTM path.
- **`advance` not detected on bout 2.** After the frozen gate fencer B produces 0 and A's
  fire mostly on stoppage-walking; `walking` and `advance` are effectively swapped there.
  Bout 1 does not show this. Adjacent classes separated in training only by crouch/stance.
- **A/B asymmetry: bout 2 only.** There A advance=26%/lunge=37% vs B advance=7%/lunge=62%
  and no fix touched it. Bout 1 reverses which fencer looks better, so it is NOT positional
  and NOT the slot-assignment logic. Handedness is also ruled out: training is 61% right /
  39% left (well balanced), and bout 2's broken fencer B is RIGHT-handed, i.e. the majority
  class. The bout-2 lefty (A) is the one the model handled better.
- **No selection rule for the per-frame model.** Best-val picked a lunge-heavy checkpoint
  twice (window seed 8 → 52%, per-frame seed 7 → 49%, vs sweep averages 42%/31%). Val
  accuracy looks mildly ANTI-correlated with video behaviour. The ensemble sidesteps this
  for the window model; the per-frame model has no equivalent.

### Dead ends — all measured, do NOT re-run

*Architecture / training:* per-frame world-motion channels into the LSTM (advance recall
78→60); shorter SEQ_LEN with all classes sliced (24f/30f → advance recall 51%/73%);
pre-padding so the action ends at the window end (83→63) and filling that pad with real
neutral context (→57); x-axis stance augmentation (val 86.3→82.9, advance 83→67);
rate-normalised length-invariant sum features (val +0.8, advance −13); within-window
RELATIVE stance/crouch/reach (val 84.6→85.2, bout advance 14→10); mirror augmentation;
2-layer LSTM; reseeding/best-of-N.

*Demo-side:* shorter windows or adding advance to FAST_CLASSES — advance is flat at 3–6%
across 60f/40f/25f/18f, so it is not being diluted; shortening just converts retreat into
lunge and parry.

*Features from Aaron's biomechanics (2026-07-31), tested by AUC on labelled clips:*
`stance_ratio` (ankles / leg length) had the BEST single-feature AUC found for
advance-vs-lunge (0.91 vs raw stance 0.87, crouch 0.77) and perfect lunge-vs-walking (1.00)
— and still made the model worse (advance 88→80). **Single-feature AUC shows a signal
EXISTS, not that the model LACKS it; never add a feature on AUC alone.** `front_knee`
(lunge 103° vs advance 137°, AUC 0.80) is real but redundant with `crouch`, which uses
min(left,right) and so already picks the front knee during a lunge. Front-foot ACCELERATION
(the lunge "kick") AUC 0.54 — nothing, probably because a lunge extension is ~5 frames at
30fps. Back-leg extension AUC 0.58 — the back knee is 177–178° in EVERY class including
neutral and walking, so "more extended than before" is not recoverable from 2D pose.

**Demo framing bug, FIXED (2026-08-02).** `MIN_BOX_H_FRAC` was 0.35, tuned on bout 2 where
fencers fill 0.30-0.60 of frame height. Bout 1 frames them at 0.20-0.50 (median box 0.360),
so the filter discarded **44% of real detections** and fencers went unlabelled for stretches
(one slot down to 27% coverage). Now 0.25, calibrated not guessed: bout 2's banner-graphic
detections top out at 0.20 and its real fencers start at 0.30, leaving an empty gap. Coverage
90% → 99%. Lesson: constants tuned on one video silently break another — check any threshold
against BOTH bouts.

**`parry` over-prediction is the MODEL, not the demo's fast-path (2026-08-02).** `parry` is
the only FAST_CLASS, so a short 25-frame window scoring >=0.65 can override the 60-frame call,
and that path can only ADD parries. Measured properly by disabling it: it contributes 16 of
fencer A's 68 parry calls and 5 of B's 22 — about a quarter. The rest come from the long
window. A confidence guard (short may not overrule a more-confident long) is in place but
removes only ~4 calls; short windows are systematically more confident than long ones, so it
rarely triggers. **Do not tune the demo for this** — parry has exactly one feature and the
fix is upstream.
*Method note:* a first diagnostic claimed 87% of parries came via the override. It stubbed
camera pan to ZERO, which shifted long-window confidences and inverted the conclusion. Any
replay of the demo loop must compute pan the same way the demo does, or its numbers are junk.

**`advance` vs `neutral` on small advances (open).** Aaron: fencer B's *small* advances read
`neutral`. Consistent with the features — a small advance keeps a narrow stance and shallow
crouch, which is the neutral profile, so only world-motion separates "small step" from
"standing". Windows called neutral vs advance for B: stance 0.25 vs 0.56, crouch 0.44 vs 0.61.
Not obviously fixable in code; needs labelled small-advance examples or a sharper world-motion
signal.

## FIRST REAL EVALUATION (2026-08-04) — ground truth exists now

Aaron labelled `data/raw_video/1.mp4` (FIE Worlds Hong Kong, DOSA vs CHOUPENITCH, 104 s) as
intervals; see `data/labels/bout1_intervals.csv`. 793 windows fall inside labelled time.
Scored with `scratchpad/evaluate.py`, which reuses demo_video's own functions (a reimplemented
loop got pan wrong once and inverted a conclusion — never reimplement it).

**Overall raw accuracy 19.0%. Random is 16.7%.** The model is barely above chance on
continuous footage, against ~86% on held-out clips. Every clip-based number in this file
should be read in that light.

| class | n_true | precision | recall |
|---|---|---|---|
| advance | 151 | 15% | **5%** |
| lunge | 21 | **3%** | 57% |
| neutral | 191 | 28% | 5% |
| parry | 8 | 2% | 12% |
| retreat | 78 | 33% | **73%** |
| walking | 344 | 53% | 19% |

- **`lunge` is a default, not a class.** 21 true windows; predicted **371 times**. It swallows
  advance (122 of 151), walking (155), neutral (70). Fixing lunge over-prediction is THE task.
- **`advance` → `lunge` on 81% of true advances.** Confirms what Aaron saw by eye.
- **`retreat` works** (73% recall) — so direction itself is learnable; the failure is specific.
- **A/B ASYMMETRY SOLVED — it was never the fencer.** A 10.4% vs B 27.2%, and the labels show
  A advances while B retreats in nearly every exchange. A scores badly because `advance` is
  broken, B well because `retreat` works. Mirroring, pose quality, facing coverage and
  handedness all came back negative because they were the wrong question.

*Scoring note:* the "viewer sees" variant counts `ready` as wrong when truth is
neutral/walking, but `ready` is the INTENDED render for QUIET_CLASSES. That number (7.3%) is
a scoring artifact; fix the scorer before quoting it.

### Next step — more labels, and fix lunge

The blocking task is no longer "get labels" — it is **fix `lunge` over-prediction**, now
measurable. 104 s gives only 21 true lunge and 8 parry windows, so those two rows are thin;
another 2-3 minutes of labelled footage would firm them up and is the cheapest next step.

### Older note — ground-truth labels (was blocking, now partly done)

Everything above is measured on hand-trimmed clips, and three separate metrics have now
lied because of that. The unblocking task is **INTERVAL labels on continuous footage**:
`fencer,start,end,label` per fencer (left/right, not A/B), covering every frame including
stoppages, classes from the six plus `unclear`. Intervals not fixed-size clips — a 3 s clip
re-creates the single-label ill-posedness. ~60–80 boundaries covers a 40 s bout; aim for
2–3 minutes for stable per-class precision/recall. That also becomes the first training
data from continuous video rather than pre-segmented clips.

Secondary lever: more transient-class clips. Not for any domain gap, but because 35 advance
+ 33 retreat + 48 lunge clips cannot determine a boundary in a 128-dim representation — the
0.4%–34% seed spread IS that underdetermination. `advance` is the smallest class (35 clips,
not sliceable; walking gets 232 windows from 27 clips), so it is cheapest to improve.

- Phase 0 ✓ — environment verified on Python 3.14 (`scripts/smoke_test.py`)
- Phase 1 ✓ — `scripts/extract_blade_frames.py`; 400 frames extracted and labeled in Roboflow
- Phase 2 ✓ — `src/pose_pipeline.py` (Tasks API), `src/person_detector.py`, `scripts/process_clips.py`
- Phase 3 ✓ — blade detector trained (YOLO11n): `models/blade_yolo/fencing_blade_v2/weights/best.pt`,
  val metrics P 0.79 / R 0.74 / mAP50 0.74; loaded by `src/blade_detector.py`
- Extra: `scripts/auto_clip.py` — experimental heuristic that proposes action-clip windows from a
  raw match video (pass the video path as an argument); review its output manually and moved
  keepers into `data/clips/<action>/`
- Raw match footage is gitignored and lives locally/OneDrive, not in the repo

- Phase 4 ✓ (first pass) — `src/action_model.py` + `src/train_action.py`; trained on 53 clips
  (advance 21 / lunge 12 / retreat 11 / parry 9): 70% val accuracy, `models/action_lstm.pth`.
  Normalization was changed from shoulder-width to torso-length scaling (see Phase 2 rules) and
  all keypoints regenerated — re-run `scripts/process_clips.py --force` after any future
  normalization change.

**The 6-class expansion is DONE (2026-07-23), built for the Phase 5 right-of-way engine:**
- **neutral** (23 clips) — idle baseline: en-garde, bounces, prep steps, pauses. Keeps demo
  labels blank until a real action happens.
- **walking** (27 clips) — upright between-phrase walking; separate from neutral because it
  translates forward and would otherwise fire "advance". Priority logic treats walking ≡
  neutral (no priority action).
- **extension** — DROPPED as a class (super short, and every lunge contains one). It lives
  on as the arm-reach FEATURE, which the priority engine reads for "who extended first";
  point-in-line = sustained high arm-reach + no travel. `data/clips/extension/` and
  `data/keypoints/extension/` still exist but hold only a `.gitkeep` placeholder — they
  are not empty, so deleting them also drops the tracked placeholder. Harmless to leave.
- Long neutral/walking clips are auto-sliced into overlapping 60-frame windows
  (SLICEABLE_CLASSES / SLICE_STRIDE in action_model.py); splits are GROUP-aware (a clip's
  windows never straddle train/val — see group_stratified_split) and the loss is
  class-weighted to counter the slicing imbalance.
- The 6-class list is LOCKED for Phase 5 (touch predictor input width depends on it).
- Demo overlay polish DONE (2026-07-23): `QUIET_CLASSES = {neutral, walking}` render as a
  grey "ready" tag instead of an action label, and any call under `ACTION_CONF_FLOOR`
  (0.50) is suppressed the same way. So labels stay silent through idle/repositioning and
  only light up on real actions — verified on clips (walking -> "ready", parry -> amber
  "parry 83%") and the match segment.

The demo already runs **multi-scale windows** (implemented 2026-07-23 in
`scripts/demo_video.py`): a 25-frame short window catches fast actions and a 60-frame long
window reads sustained ones; a confident (>=0.65) fast-class hit on the short window
overrides the long-window label. Measured motivation: parries are ~12 frames, and a single
2 s window buried them (peak parry prob 0.57 -> never wins for one fencer; 3x more parry
detections at short windows). When the 7-class model lands, add `extension` to
FAST_CLASSES there.

Clip-cutting rules learned the hard way: crop tight (fencer ≥ half frame height), keep the
action inside the first 2 s, cut every class with the same ~0.5 s lead-in / 0.3 s lead-out,
label by the clip's dominant action. Clip videos are gitignored (local/OneDrive only); the
extracted `data/keypoints/*.npy` are tracked in git. Banners with printed fencers can fool the
YOLO person detector (not MediaPipe pose), so eyeball auto-crops before trusting them.

**Phase 5 (touch predictor) data — label OUTCOMES, not actions.** Action labeling is done and
feeds this stage. New effort is a small scorer CSV per bout: `timestamp_start,timestamp_end,
scorer` with scorer in {A,B,N}. At train time the action model runs on each window and its
class probs become INPUT features (with blade-tip velocity + joint angles); the label is who
scored. Priority = a RULE layer over the action streams (extension establishes attack; parry
transfers priority to defender -> riposte; simultaneous -> no touch) — no extra labels, and
more explainable than a learned model. Awarded touch = blade contact + priority.

When the user says something like "let's start" or "let's do Phase X", begin by:
1. Stating which phase you're working on
2. Listing the files you're about to create or modify
3. Writing the code
4. Telling the user exactly what command to run to test it

When you are unsure whether something will work (library version, hardware capability, API change), say so explicitly rather than guessing.
