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

### FIXED — class-prior correction (2026-08-04): 19.0% → ~42%

Root cause of `lunge` eating everything: **prior mismatch, not a broken feature.** Clips were
cut OF actions, so the corpus over-represents them (lunge 10% of training windows vs 2.6% of
real footage, parry 10% vs 1.0%), and inverse-frequency class weighting then drives the
model's effective prior to UNIFORM. It expected lunge ~6x too often, so lunge stopped being a
class and became a default.

Fix is standard label shift, applied in `demo_video._classify_window`:
`p(c|x) * CLASS_PRIOR(c) / train_prior(c)`, and since the train prior is uniform it drops out
of the renormalisation. `APPLY_CLASS_PRIOR = False` disables it for measurement.

| class | before | after (held-out) |
|---|---|---|
| advance | 15% / **5%** | 53% / **67%** |
| walking | 54% / 19% | 54% / **49%** |
| retreat | 32% / 73% | 28% / 60% |
| neutral | 28% / 5% | 33% / 8% |
| lunge | 3% / 57% | 0% / 0% |
| overall | **19.0%** | **41.9%** |

**Quote 41.9%, not 50.2%.** The 50.2% you get by running `evaluate_labels.py` as shipped uses
a prior fitted on the same footage it scores. 41.9% is honest: prior fitted on one half of the
labelled bout, scored on the other, both directions agreeing (15.7%→42.6%, 22.6%→41.2%).

Notes:
- `lunge` going to 0% is not a regression. Its old "57% recall" came with 3% precision off 378
  predictions for 21 true windows — it was never working, the correction just stops it being
  the default. 21 windows is too thin to tell whether lunge is recoverable.
- `neutral` is still weak (8%), leaking mostly to `walking` — the benign confusion (both mean
  "no priority action").
- **EM prior estimation does NOT work here** (Saerens et al., needs no labels). It estimated
  lunge at 0.635 against a true 0.027 and made things worse, 19.2% → 14.6%: the classifier is
  biased enough that EM just confirms its own error. So the prior must come from labelled
  footage, not be inferred at runtime.
- The prior is from ONE 104 s bout. Re-estimate as more is labelled, and re-check it transfers
  — a bout with more stoppage time would shift walking/neutral.

### TRANSFER CONFIRMED (2026-08-05) — second labelled bout, `data/raw_video/2.mp4`

`data/labels/bout2_intervals.csv` (ONO vs ITKIN, 110.9 s, 254 scored windows). Prior fitted on
bout 1, applied unchanged to a different match and different fencers — a true held-out test.

| | bout 1 (held-out prior) | bout 2 (bout-1 prior) |
|---|---|---|
| overall | 41.9% | **43.3%** |
| advance | 53% / 67% | **69% / 70%** |
| lunge | 0% / 0% | **50% / 19%** |
| walking | 54% / 49% | 40% / 55% |
| retreat | 28% / 60% | 38% / 35% |
| neutral | 33% / 8% | 38% / 12% |
| parry | 0% / 0% | 3% / 7% |

- **The prior generalises.** 43.3% on an unseen match vs 41.9% held-out within bout 1. It is
  not fitted to one match's stoppage pattern.
- **`advance` is now usable** at 69/70 — the class that was 15%/5% before the correction.
- **`lunge` is NOT dead.** Its 0% on bout 1 was 21 windows of thin data; on 31 windows here it
  reaches 50% precision.
- **`parry` is now the dominant error and it took over lunge's old role.** Predicted ~37 times
  for 1 correct, and it eats `retreat`: 22 of 54 true retreat windows are called parry, which
  is what drags retreat recall to 35%. Fixing parry is the next lever.
- **A/B asymmetry, final form.** Bout 2 REVERSES the footwork roles (A retreats, B advances)
  yet A is still worse (32.4% vs 56.5%) — because in bout 2 fencer A performs all four parry
  sequences, and parry is the worst class. Per-fencer accuracy tracks WHICH ACTIONS that
  fencer performs, not the slot, not the fencer, not handedness. A simply drew the hard
  actions in both bouts.
- `extension` appears in bout 2's labels; it is not one of the six classes (it is the arm-reach
  FEATURE), so `evaluate_labels.py` counts and EXCLUDES those windows (22 here) rather than
  penalising the model for a label it cannot emit.

Run either bout: `py -3 scripts/evaluate_labels.py <video.mp4> <labels.csv>`

**PRIOR TRANSFER IS ASYMMETRIC — correcting an earlier over-claim.** "The prior generalises"
was based on one direction only. Measured both ways (uniform / bout1-prior / bout2-prior /
pooled, scored on each bout):

| prior | bout 1 | bout 2 |
|---|---|---|
| none (uniform) | 19.2% | **46.1%** |
| bout 1 | 50.2%* | 43.3% |
| bout 2 | **24.4%** | 47.2%* |
| pooled (shipped) | 45.9% | 44.1% |

(*circular — that prior saw that bout.) Bout 1 is idle-heavy (walking 0.457), bout 2
action-dense (walking 0.232). A prior lifted from busy footage under-weights walking and
wrecks quiet footage (24.4%). Note bout 2 scores 46.1% with NO correction, because its true
distribution is already near uniform — the correction's value depends on the footage. Pooled
is shipped as the most balanced. Re-derive with `scripts/estimate_class_prior.py`.

**PARRY IS A SCHEMA PROBLEM, NOT (mainly) A MODEL FAILURE (2026-08-05).** Aaron: "when
parrying, retreating is super duper common." Confirmed — the `retreat`→`parry` confusions
cluster on real parries, the correct `retreat` calls do not:

**The numbers first cited for this were wrong twice over and are corrected here.** The
original table reported "median 999 s" for correctly-called retreats — 999 was a SENTINEL
written when that fencer had no parry anywhere to measure a distance to, so taking its median
produced a duration that means nothing. And the counts came from the pre-silhouette-filter
detector. Recomputed on the current pipeline, separating "no parry exists" from a real
distance:

| bout | model said (truth = retreat) | n | fencer has NO parry | median dist. | within 3 s |
|---|---|---|---|---|---|
| 1 | parry | **0** | — | — | — |
| 1 | retreat | 62 | 0 | 15.15 s | 19% |
| 2 | parry | 25 | 0 | **2.65 s** | **60%** |
| 2 | retreat | 18 | **17** | — | — |

Bout 1's 22 retreat→parry confusions are GONE after the detector fix. Bout 2's 25 do sit near
real parries, but the control group is contaminated: 17 of the 18 correctly-called retreats
belong to a fencer with no parries at all. So the contrast is largely BETWEEN FENCERS (the one
who parries gets called parry) rather than between moments, which is much weaker support than
was originally claimed.

The schema argument does not depend on those numbers and still holds: a fencer genuinely
parries WHILE retreating, and six mutually-exclusive classes cannot express both, so some
windows are scored wrong whatever the model says.

**BOUT 3 SETTLES IT: THE MODEL HAS NOT LEARNED PARRY AT ALL (2026-08-08).**
`data/labels/bout3_intervals_2track.csv` — 44 intervals, action-hunted, **11 parries** against
6 across bouts 1+2, and parries from BOTH fencers (left 6, right 5), which removes the
per-fencer confound that made bout 2's evidence weak.

First: **11 of 11 parries have `retreat` as their footwork.** (CORRECTED by bout 4 below --
across 46 parries it is 74% retreat, not 100%. One bout was talking.)

But the schema is not what is blocking parry. Scored on 399 labelled windows:

| | parry predictions | parry recall | overall |
|---|---|---|---|
| shipped CLASS_PRIOR | **0** | 0% | 42.1% |
| no prior (uniform) | **0** | 0% | 35.3% |

Zero either way, so the prior is not the cause. The raw parry output is near-noise:

| | mean | median | max |
|---|---|---|---|
| true parry windows (55) | 0.037 | 0.038 | 0.106 |
| everything else (344) | 0.031 | 0.027 | 0.121 |

Parry never ranks #1 or #2 on ANY true-parry window (median rank #4 of 6); AUC 0.61.
**Do not spend more effort on parry labels, priors, or thresholds** — none of them can move an
output this uninformative. It needs either blade information the pose keypoints do not carry,
or a window short enough not to average a 0.92 s action across 2.0 s.

**Dead code path found by the same measurement:** `FAST_CLASSES = {"parry"}` lets the short
window override the long one, but requires `short_conf > FAST_CONF = 0.65`. Parry probability
maxes at 0.106. The one mechanism built specifically to rescue parry is unreachable by roughly
6x and has likely never fired. Verify before removing or re-tuning it.

**`advance` is now the default class.** 152 predictions for 60 true windows, swallowing 51 of
64 lunges (lunge recall 9%). This is exactly the pathology `lunge` had before the prior
correction, transplanted. Bout 3 is action-hunted so its true shares (advance 0.150, lunge
0.160, parry 0.138) are nothing like real footage — do NOT pool it into CLASS_PRIOR — but the
lunge→advance collapse is worth investigating on its own.

Bout 3 overall 42.1% sits with bout 1's 42.9% and bout 2's 42.3%: the pipeline is consistent
across three matches now.

**THE PER-FRAME MODEL IS A DEGENERATE LUNGE PREDICTOR. Do not route to it (2026-08-08).**
Recorded because the opposite conclusion was reached first, on bout 3 alone, and was wrong.

The hypothesis was that `out.mean(dim=1)` over 60 frames dilutes a 0.8 s lunge, since a window
is scored at its NEWEST frame, and that `ActionFrameLSTM` + `frame_logits_to_window(
mode="last")` would fix it. On bout 3 lunge recall went 9% -> 78% and that looked decisive.

Checked on all three bouts, recall DOES replicate — 81% / 74% / 78% — but precision does not:

| `--frame-model`, raw | bout 1 | bout 2 | bout 3 |
|---|---|---|---|
| overall | **14.8%** | 38.1% | 34.3% |
| lunge recall | 81% | 74% | 78% |
| lunge **precision** | **3%** | 29% | 26% |
| predicts lunge on | **60%** of windows | 33% | 49% |
| true lunge share | **2%** | 12% | 16% |

It fires lunge on 60% of bout 1's windows against a 2% true share. High recall at 3% precision
is a saturated class, not a detector, and the same pattern explains its parry "recall" of 25%
and 20% on bouts 1-2 (precision 2% and 6%). Fitting the prior to each bout's OWN labels — which
is cheating — still leaves bout 1 at 16.6% with lunge precision 3%, so this is not a
calibration problem either.

**The pooling hypothesis therefore has no support.** There is no evidence that mean-pooling is
what hurts transient actions. The window model remains the best available at 42.9 / 42.3 /
42.1% across the three bouts. `FAST_CLASSES` + `FAST_CONF = 0.65` is still an unreachable dead
path (parry's probability maxes at 0.106) but replacing it with a per-frame router is NOT the
answer.

**Method note — this was the day's fifth wrong call and its shape was familiar:** a single
bout's headline number, a plausible mechanism written around it, and a precision column sitting
in the same table that contradicted it. Recall without precision proves nothing; on this
project, neither does one bout.

### BOUT 4 (2026-08-09) — the continuous corpus is now viable

`data/labels/bout4_intervals_2track.csv`, from a 26.1 min source: **304 intervals, 708 s of
labelled fencer-time**, which is 2.4x bouts 1-3 combined. Sparse by design (23% coverage) —
Aaron: "if there is a gap it probably means there's no arm/blade thing."

| | bouts 1-3 | bout 4 | total |
|---|---|---|---|
| labelled fencer-time | 294 s | **708 s** | 1002 s |
| parries | 17 | **46** | 63 |
| lunges | 33 | 61 | 94 |
| independent ~2 s windows | ~147 | **~354** | **~500** |

**This crosses the threshold that made continuous training not worth doing.** The clip corpus
is 488 windows; the continuous corpus is now ~500, with realistic transitions and no
clip-cutting artifacts. Train on BOTH — the clips carry match diversity that four bouts do not.

**CORRECTION to bout 3's headline finding.** "11 of 11 parries have retreat footwork" does not
generalise. Over 46 parries:

| footwork under a parry | n |
|---|---|
| retreat | 34 (74%) |
| neutral | 7 |
| advance | 5 |

Parries during an ADVANCE exist. This strengthens the two-track case rather than weakening it —
a single label could not express any of the three — but the unanimity was one bout talking, the
same error shape as the p99 blade result and the per-frame routing result. Parries are also
balanced across fencers (left 22, right 24), so bout 4 carries no per-fencer confound.

**Still do NOT pool bout 4 into CLASS_PRIOR.** 23% coverage selected for action: its duration
shares describe where Aaron looked, not the sport. A contiguous stretch is still needed for that.

**FIXED: A/B SLOT SWAPPING WAS THE LARGEST ERROR IN THE SYSTEM. Bout 4 43.6% -> 55.1%.**

Found only because bout 4 is big enough (3822 scored windows) to separate two hypotheses that
look identical at bout-1-to-3 scale. The dominant error was 556 advance<->retreat confusions,
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
preferred it. Assignment writes `last_hip_x`, which drives the next assignment, so ONE bad swap
persists indefinitely — the same absorbing-error mode diagnosed on the referee experiments
earlier the same day, in code that had been read twice without noticing.

**Fix: fencers never cross a piste, so relative x-order IS identity.** Leftmost -> A, rightmost
-> B, always, no history. Memoryless, so a bad frame cannot propagate.

| bout | before | after | scored windows |
|---|---|---|---|
| 1 | 42.9% | 43.4% | 874 |
| 2 | 42.3% | **48.3%** | 266 |
| 3 | 42.1% | 42.1% | 399 |
| **4** | 43.6% | **55.1%** | **3822** |
| 4, viewer view | 51.7% | **61.5%** | |

No bout regressed. On bout 4: advance 29%/30% -> **52%/54%**, retreat 29%/39% -> **50%/67%**,
and the 556 direction confusions fall to 174 (-69%), which is the mechanism check the fix was
predicted to pass. Parry is untouched (1 correct of 211 before, 3 of 207 after) — as expected;
this was never a parry problem.

**Why this was missed for so long:** at 266-874 windows the inverted runs are rare enough to
look like ordinary confusion. It took 26 minutes of labelled footage to make a tracking bug
distinguishable from a learning problem. That is the strongest argument yet for continuous
labelled data — not as training material, but as instrumentation.

Transcription notes (source table had a few slips, all recorded rather than silently fixed):
two malformed timestamps (`20.53.520`, `22:118.967`) read as `20:53.520` and `22:18.967`; one
right-fencer blade action at 1360.045 s dropped because its footwork cell was blank; five
overlaps of 0.01-0.14 s clamped by truncating the earlier interval (the shortest real interval
in the bout is 0.30 s, so a clamp that size cannot move a label onto the wrong action).

**PARRY IS CLOSED.** 0% in both architectures, with 47 parry clips (more than advance's 35 or
retreat's 33). Not a data problem, not a schema problem, not a prior problem, not a pooling
problem. Do not spend more labelling on it.

**On more interval labels:** they are NOT training data — the model trains only from
`data/clips/`. 20 minutes of intervals would not reach the model at all as things stand. Where
they WOULD pay: (1) a contiguous stretch of ORDINARY footage to firm up CLASS_PRIOR, which
still transfers asymmetrically (bout2-prior -> bout1 = 24.4%); (2) training on continuous
windows instead of hand-cut clips, which would end the clip-cutting artifact class that has
produced three false results here.
**The classes are not mutually exclusive.** They are two near-orthogonal tracks:
- FOOTWORK (legs): advance / retreat / lunge / walking / neutral
- BLADE (arm): parry / extension / none
A fencer parries WHILE retreating. This is why parry has been the worst class since the start
and why no feature or architecture fix ever moved it. Do not try to suppress parry; the fix is
to label and predict the two tracks separately (and it lines up with Phase 5, which already
needs blade action and footwork independently for priority).

**BLADE MOTION ENERGY instead of blade DETECTION (2026-08-05).** Aaron: "the problem with
fast blades is that in the video sometimes it just disappears or it's only a blur." Right,
and that kills the detector approach on its own terms — there is nothing sharp in those
pixels, so more training frames cannot help. But blur is high frame-DIFFERENCE energy, so
the property that breaks a detector is the one a motion measure wants.

`scripts/blade_energy.py` measures mean/p99 frame-difference inside a box projected along
the forearm (camera pan removed first; an equal-area TORSO box as control).
`scripts/analyze_blade_energy.py` scores it. Coverage is the immediate win:
**100% of tracked fencer-frames vs 1–2% for the detector.**

**RESULT: as implemented, it does not work.** Parry vs non-blade AUC, per interval:

| statistic | bout 2 (n=4 parries) | bout 1 (n=2) | **POOLED (n=6)** |
|---|---|---|---|
| box **mean**, blade/torso | 0.62 | 0.65 | **0.59** |
| box **p99**, blade/torso | **0.70** | 0.49 | **0.55** |

**This is the fourth time a metric here lied in the same way, and I walked into it live.**
On bout 2 alone, p99 beat the mean 0.70 vs 0.62, and I wrote a confident mechanism into this
file for why — a blade is a thin streak, averaging over a mostly-background box destroys it,
p99 asks the right question. The mechanism is plausible and the number did not replicate:
bout 1 gives p99 **0.49**, and pooled it is 0.55 versus the mean's 0.59. The reasoning was
retrofitted to one sample of four. Neither statistic is distinguishable from chance.

What survives:
- **Coverage, which is real and large.** The blade box is measurable on **100%** of tracked
  fencer-frames versus 1–2% for the detector. Whatever gets measured there, it gets measured
  everywhere, which the detector can never do on blurred frames.
- **The negative result itself** — this is cheap to have killed before wiring it into the
  model. Compare `stance_ratio`: best AUC of any feature tried (0.91) and it still made the
  model worse.

Before retrying, the two things most likely to be wrong with the measurement (untested):
the box is ~3× forearm long and mostly background/opponent, and pan compensation is a single
global x-shift estimated at 320×180, which will not cancel parallax on a moving camera.

This does NOT settle whether blade information helps — only that this measurement of it
does not. And 6 parry intervals cannot settle anything either way, which is the concrete
answer to "should I get more parry intervals": **yes, that is exactly what they resolve.**

**THE REFEREE IS BEING TRACKED AS A FENCER — real, measured, and NOT yet fixed
(2026-08-05).** Aaron: "in bout 1 the person detector was clipping onto the referee
silhouette in the middle a lot." Confirmed. `get_fencer_boxes` resolves >2 people by keeping
the **two highest-confidence** boxes, and on bout 1:

- 68% of frames contain more than two tall people, so the tiebreak runs constantly
- on **51% of those** it picks a different pair than horizontal separation would
- the box it keeps and separation rejects sits at median **x = 0.49** (dead centre) with
  median confidence **0.85**, displacing a real fencer at **0.71**; 64% are in the middle third

The mechanism is clear: confidence measures *resemblance to a standing person*. A referee is
still, upright and unoccluded; a fencer mid-lunge is blurred, horizontal and self-occluded.
So confidence systematically prefers the referee. That is the opposite of what is wanted.

**Two fixes were tried and BOTH made it worse.** Bout 1, RAW model call:

| rule | overall | advance recall | retreat recall |
|---|---|---|---|
| top-2 by confidence (current) | **45.9%** | 50% | 64% |
| all candidates → continuity via `_assign_boxes` | 35.7% | 11% | 13% |
| top-4 by confidence → continuity | 35.8% | 11% | 13% |

The second attempt tested the obvious explanation for the first — that removing the
confidence gate let junk 0.4-confidence blobs win on distance — and it changed **nothing**
(35.7 → 35.8). So the regression is in the assignment logic itself, not the candidate pool.
Best remaining guess, untested: the >2 branch chose the pair minimising total distance to
remembered hip positions with no left/right constraint and no hysteresis, so slots can swap
frame to frame and shred the 60-frame window. Hip-x distance is too weak an association
metric; this probably needs real IoU-based tracking rather than a nearest-centre rule.

**THEN I LOOKED AT THE FRAMES, which I should have done first.** `scripts/inspect_detections.py`
draws every tall detection on frames where the rules disagree (output in
`data/diagnostics/<stem>_detections/`). A single frame settled what three experiments could not:

| # | conf | x | height | what it actually is |
|---|---|---|---|---|
| 0 | 0.87 | 0.26 | 0.27 | left fencer |
| 1 | **0.85** | 0.11 | 0.32 | **foreground spectator silhouette** |
| 2 | 0.80 | 0.44 | **0.41** | the referee |
| 3 | 0.65 | 0.71 | 0.32 | right fencer |

The intruders are not only the referee, and they are **TALLER than the fencers** (0.32–0.41
vs 0.26–0.27) because they stand between the camera and the piste — so `MIN_BOX_H_FRAC`
cannot reach them, and raising it would delete the fencers first. Here the confidence rule
keeps #0 and #1, **drops the right fencer entirely**, and sorts the remaining pair by x so
the LEFT fencer lands in slot B. Slot B faces left, so net-forward inverts: that is the
advance→retreat sign flip, visible directly.

Two cues separate silhouettes from fencers, and **neither works alone** — over 1262 tall
detections, boxes running to the frame bottom have median brightness 51 in bout 1 (vs 101
for the rest) but 106 in bout 2, whose tighter framing puts real fencers' feet near the
bottom edge. Together (dark AND bottom-anchored) they flag 16.2% of bout 1 and 4.1% of bout
2 — the expected split, bout 1 being the wide shot. Fencers wear WHITE, which is what keeps
them clear. Brightness is judged RELATIVE to the brightest box in frame, since an absolute
cutoff also deletes a fencer in a dark patch and does not survive a change of venue.

**The filter works and still costs accuracy.** Bout 1 / bout 2, RAW:

| rule | bout 1 | bout 2 | bout 1 advance recall | windows scored (b1) |
|---|---|---|---|---|
| top-2 by confidence (shipped) | **45.9%** | **44.1%** | **50%** | 787 |
| continuity / capped / widest-pair | 35.7 / 35.8 / 34.1% | — | 11% | — |
| dark+bottom, absolute brightness | 42.4% | 42.3% | 40% | 874 |
| dark+bottom, relative brightness | 42.9% | — | 40% | 874 |

Coverage rises 11% (787→874 windows) because a silhouette in a slot produced a frozen
skeleton that the `MAX_FROZEN_FRAC` gate suppressed; real fencers there produce real
predictions. Total CORRECT calls go up (raw 361→371, viewer view 517→565) — accuracy falls
because the newly-covered windows are hard ones. But **advance recall drops 50%→40% on
essentially the same window set** (151→156), which coverage does not explain and I could not
account for. Re-fitting the class prior against the new pipeline does not recover it
(held-out 46.2% / 39.4%, ≈ the shipped prior's 42.4%).

So: the bug is real, visually confirmed, and fixable — but every fix measured so far costs
the headline metric. Worth suspecting that some tuning to date (prior, thresholds, gates)
has quietly adapted to the broken detector, in which case the detector fix has to come with
a re-tune rather than be judged on its own.

### Next step — two-track labels, then parry

1. **Two-track labels** (`fencer,start,end,footwork,blade`). Nothing downstream works
   without them: a blade feature has no target while `parry` and `retreat` compete for one
   slot. `scripts/upgrade_labels.py` lifts bouts 1–2 mechanically and writes `TODO` only
   where it genuinely cannot infer (27 and 32 rows) rather than guessing.
   `scripts/check_labels.py` validates either schema — catches overlapping intervals, which
   silently mis-score because `truth_at()` returns the first match.
2. **Blade p99 energy** as a feature, once there are enough parries to justify it.
3. **Window length.** Interval durations across both bouts: parry median **0.60 s**, lunge
   **0.69 s** (100% under 2 s), vs `WINDOW_LONG` = 60 frames = **2.0 s**, mean-pooled. An
   18-frame parry is diluted across 60 frames that are mostly retreat. This may matter more
   than any blade feature, and it is likely part of why lunge recall sits at 26% on bout 2
   despite 57% precision. Footwork is a 1–3 s phenomenon and blade action a 0.5 s one; they
   probably should not share a window length.

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
