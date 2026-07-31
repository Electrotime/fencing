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

### Phase 0 — Environment setup
**Goal:** working Python environment with all dependencies installed.

`requirements.txt` must include:
```
mediapipe>=0.10.0
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
streamlit>=1.28.0
roboflow>=1.1.0
matplotlib>=3.7.0
```

Verify with a smoke test: import all libraries and print versions. If anything fails to import, fix it before moving on.

---

### Phase 1 — Data collection scripts

**`scripts/extract_blade_frames.py`**
- Opens every .mp4 in `data/raw_video/` (skips an interlaced original when a `_deinterlaced` twin exists, so a match isn't sampled twice)
- Samples a fixed budget of ~400 frames total (`TARGET_TOTAL_FRAMES`), evenly spaced and split evenly across videos, saved as JPEGs to `data/blade_frames/`
  - Rationale: every-5th-frame on full matches yields ~9,300 frames — far more than Phase 3's 300-500 target, and highly redundant (6 fps, a blade barely moves frame-to-frame). Target-count sampling hits the labeling budget and maximizes pose diversity.
- Naming convention: `{video_stem}_f{frame_index:06d}.jpg` — include the source video stem so frames from different videos never collide
- Prints total frames saved
- Use `cv2.VideoCapture`

No ML yet. Just OpenCV file I/O.

---

### Phase 2 — Pose estimation pipeline

**`src/pose_pipeline.py`** must expose two functions:

```python
def extract_keypoints_from_frame(frame: np.ndarray) -> np.ndarray:
    """
    Takes a single BGR frame (H, W, 3).
    Returns np.ndarray of shape (33, 4) — 33 MediaPipe landmarks, each (x, y, z, visibility).
    Returns np.zeros((33, 4)) if no person is detected.
    """

def extract_keypoints_from_video(video_path: str) -> np.ndarray:
    """
    Processes an entire video file.
    Returns np.ndarray of shape (N_frames, 33, 4).
    Fills failed frames with zeros.
    """
```

**`scripts/process_clips.py`**
- Iterates over every clip in `data/clips/<action>/`
- Calls `extract_keypoints_from_video` on each
- Saves result as `data/keypoints/<action>/<clip_name>.npy`
- Prints progress (clip name, output shape)

**`src/person_detector.py`** — required before pose estimation can work on match footage:

MediaPipe Pose detects only ONE person per frame. Fencing video has two. You must crop each fencer out of the frame separately before running pose estimation.

Use pretrained YOLOv8n (COCO) — it already knows what a person looks like, no training needed:

```python
from ultralytics import YOLO

def get_fencer_crops(frame: np.ndarray, model: YOLO) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Detects persons in frame using pretrained YOLOv8n (COCO class 0 = person).
    Returns (crop_A, crop_B) where A is the left fencer and B is the right fencer.
    Sorts detections left-to-right by bounding box x-center.
    Returns None for a slot if fewer than 2 persons are detected.
    """
```

Load with `YOLO("yolov8n.pt")` — this downloads COCO weights automatically, no extra training.

For training clips (Phase 2), each clip should already contain a single fencer (pre-cropped or filmed in isolation). The person detector is primarily needed in Phase 6 (real-time demo) and for any full-match preprocessing.

**MediaPipe configuration to use (Tasks API — `mp.solutions` was removed in mediapipe 0.10.14+):**

The model file must be downloaded once:
```python
# download_pose_model() in src/pose_pipeline.py handles this automatically.
# Model: pose_landmarker_full.task (~27 MB) — equivalent to old model_complexity=1.
```

Landmarker options:
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

### Phase 4 — Action recognition model

**`src/action_model.py`** must contain:

#### Constants
```python
CLASS_NAMES = ["advance", "lunge", "parry", "retreat"]
SEQ_LEN = 60       # frames per clip (pad/trim all clips to this)
INPUT_SIZE = 132   # 33 keypoints × 4 values (x, y, z, visibility). Face landmarks
                   # 1-10 stay in the tensor. NOTE (measured, 2026-07): MediaPipe
                   # reports them with visibility ~1.0 even under a fencing mask —
                   # it guesses their positions from head shape — so they are NOT
                   # zeroed by the carry-forward step; they just track the head.
HIDDEN_SIZE = 64   # spec said 2x128; measured at ~80 clips the small 1-layer net wins
N_AGG_FEATURES = 4 # engineered clip-level stats fed straight into the classifier head:
                   # net forward motion (WORLD travel = in-frame hip-x + camera pan,
                   # signed by facing), stance width p90, wrist speed p90, total travel.
                   # Measured: engineered feats vs keypoints alone +23 pts (51%->74%);
                   # combining hip-x with pan (vs pan alone) lifted advance/retreat
                   # direction accuracy 84%->94% and advance recall 78%->92%. The LSTM
                   # cannot rediscover these from 132 channels at this dataset size.
NUM_CLASSES = 4
```

#### Dataset class
```python
class FencingDataset(torch.utils.data.Dataset):
    """
    Loads all .npy files from data/keypoints/<class>/.
    Each sample: tensor of shape (SEQ_LEN, INPUT_SIZE).
    Each label: integer class index from CLASS_NAMES.
    Pads short clips with zeros at the end.
    Trims long clips from the end (keep the first SEQ_LEN frames) — the start of an
    action clip is the most distinctive part (initiation of a lunge, first foot step of
    an advance), so trimming from the end preserves the signal that matters.
    """
```

#### Model
```python
class ActionLSTM(nn.Module):
    """
    2-layer LSTM followed by a 2-layer MLP classifier.
    Input:  (batch, SEQ_LEN, INPUT_SIZE)
    Output: (batch, NUM_CLASSES) — raw logits, NOT softmax
    Use dropout=0.3 between layers.
    """
```

#### Training function
```python
def train_action_model(
    keypoints_dir: str,
    save_path: str,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2
) -> dict:
    """
    Full training loop.
    Returns dict with keys: train_losses, val_losses, val_accuracies.
    Saves best model (by val accuracy) to save_path.
    Prints epoch, loss, val accuracy each epoch.
    """
```

**`src/train_action.py`** — thin wrapper that calls `train_action_model` with default args and plots the loss curve using matplotlib.

**Evaluation requirement:** After training, print a classification report using `sklearn.metrics.classification_report`. If any class is below 70% recall, tell the user they need more clips for that class.

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

### Action class definitions
- **lunge**: explosive forward extension of the sword arm and front leg
- **parry**: defensive blade deflection, typically small wrist motion
- **advance**: step forward (front foot then rear foot)
- **retreat**: step backward (rear foot then front foot)

Additional classes to add once the base 4 are working:
- fleche (running attack)
- riposte (attack immediately after a parry)
- en-garde (static ready position — useful as a "no action" class)

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

**KNOWN OPEN BUG — the second fencer reads "retreat" through stoppages (2026-07-24).**
On the 40 s match segment the left fencer (slot A, facing right) gets a sane mix, while
the right fencer (slot B, facing left) reads `retreat` ~134/237 windows, `advance` 1 and
`neutral` 3. Ground-truthed to real errors: pulled the frames where A=neutral and
B=retreat, and the match clock is frozen at 2:34 with the fencers far apart — a stoppage
where both should read neutral/walking. Reproduces across 6 freshly-trained seeds, so it
is not the shipped checkpoint. What it is NOT (all measured, don't re-run):
- NOT the engineered features. Ablating each of the 6 to its neutral-class median leaves
  B at 134→148 retreats. Swapping pathways localises it: A's keypoints stay neutral-heavy
  with B's features, B's keypoints never go neutral with A's features. It's the LSTM path.
- NOT `nose_dir` instability. Facing sign is stable over the bout (A right 196/229,
  B left 214/237, only 4 flips each), and B's net-forward feature is correctly balanced
  (82 positive / 92 negative) — the feature is right, the classifier ignores it.
- NOT pose quality. demo A vs B are identical on size-in-frame (0.31/0.29), jitter
  (0.0079/0.0083), carried-forward joints (0%/0.4%), visibility (0.99/0.99).
- NOT facing coverage. Per-CLIP counts are balanced: neutral 12 right/11 left, and every
  class has ≥11 clips on its thinner side.
- NOT fixable by mirroring. Facing canonicalisation was tried BOTH ways and both lose:
  with the L/R landmark swap 81.8% vs 85.1% baseline; without the swap 80.6% vs 87.1%,
  and it merely moves the damage (facing-left 84%→85%, facing-right 91%→76%). Mirroring
  changes a fencer's handedness, and the sword-arm side is real signal.
What IS established: a real but modest out-of-sample facing gap — held-out accuracy 91%
facing right vs 84% facing left (the earlier in-sample check showed 89%/89% and was blind
to it, since it can't see a generalisation gap). 7 points cannot explain a 134:1 skew, so
the residual is most likely DOMAIN gap: every training clip is Aaron's own footage/venue/
camera, while the demo is broadcast footage of elite fencers. Best next lever is data from
the demo's own domain — neutral/idle clips cut from BROADCAST stoppages, both facings —
not another architecture change.

**RETRACTED (2026-07-24) — there is NO domain gap. Do not repeat this claim.** An earlier
version of this section said the training clips were "textbook-wide" while match footwork
was "compact", based on bout stance 0.30–0.36 vs train advance 0.45–0.91. That comparison
was confounded and the conclusion was wrong on two counts. (1) Aaron's clips are themselves
cut from broadcast matches — same source domain, and indeed same 1920x1080 / 29.97 fps as
the demo. (2) The comparison averaged ALL bout windows, but a bout window is a 2 s sliding
window over continuous video and most of them are idle, repositioning or stoppage, while
training clips are pure action. Restricting the bout to its ACTIVE windows (top quartile of
travel) the two line up almost exactly: stance 0.58 vs 0.54, crouch 0.55 vs 0.57, arm-reach
0.21 vs 0.21, net-forward 2.48 vs 1.87. Windows inside the advance band on all six features
go 19% → 40% once idle windows are dropped. The training data covers the bout fine.
Also NOT pose quality: re-extracting training clips through the demo's person-crop RAISES
jitter (0.0122 → 0.0158) instead of lowering it toward the demo's 0.008, and subject size is
comparable (train 0.19–0.40 of frame height, bout 0.29–0.31). No re-extraction needed.

**ROOT CAUSE — advance loses INSIDE its own feature region; it is the decision boundary,
not coverage.** Of the 89 bout windows sitting inside the training advance band on all six
features, the model calls lunge 55 (62%), retreat 30 (34%) and advance 4 (4%). The features
are right there and it still will not say advance; the median probability gap to the winner
is 0.341, far too wide for threshold tuning to paper over. Two concrete train/serve
mismatches feed this, both in how the WINDOW is built rather than in the model:
- Zero padding is a spurious class cue. Clip length tracks class (lunge/parry 24f → 60%
  zeros, advance 46f, retreat 48f, neutral/walking sliced → 0%), and the head mean-pools
  across the padding. Re-padding the same clips by holding the last frame drops in-sample
  accuracy 85% → 53% and flips 40% of calls, lunge→advance on 16 clips. The demo pads
  nothing, so the model's strongest cue is absent exactly when it matters.
- The action sits at the WRONG END. Training builds `kp[:SEQ_LEN]` = [action | zeros]; the
  demo feeds the last 60 frames of a live track = [context | action]. An LSTM is sequential,
  so this is not cosmetic.
Consequence worth remembering: val accuracy barely constrains demo behaviour. Models scoring
82–92% on validation span 1.9%–35.4% advance on the bout, because validation shares the
padding artifact and continuous video does not. Score candidate changes on `val@hold`
(same clips re-padded by holding the last frame) as a continuous-video proxy, not on val alone.
Four fixes tested and REJECTED (don't re-run):
- Per-frame world-motion channels into the LSTM (in.132→134, velocity + cumulative
  displacement). The LSTM is fed hip-centred frames so it cannot see translation, and
  direction rests on one scalar against 128 LSTM dims — but adding it LOSES: 85.4%→83.5%,
  advance recall 78%→60%. Mean-pooling makes per-frame velocity ≈ the existing net-forward
  scalar, so it is redundant noise plus extra capacity to overfit on 488 windows.
- Shorter demo windows / adding advance to FAST_CLASSES. `advance` is flat at 3–6% across
  60f/40f/25f/18f windows — it is not being diluted. Shortening just converts retreat into
  lunge and parry as displacement stops accumulating.
- Horizontal (x-axis) augmentation, u~U(0.55,1.10), to widen stance tolerance:
  val 86.3%→82.9%, advance recall 83%→67%, bout-advance 12%→4%. Loses on every axis.
- Reseeding / best-of-N. Bout-advance is ~5–7% for EVERY model. A 5-seed ensemble is worth
  having for other reasons (val 84.4%→85.6%, out-of-domain sd 4.1%→1.7%, range 1.9–20.2%
  →3.0–7.1%) but converges to 5% advance, not up. Note val accuracy barely constrains
  out-of-domain behaviour: models scoring 82–92% on validation span 1.9%–35.4% bout-advance.
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

**PER-FRAME MODEL (2026-07-29) — `ActionFrameLSTM`, `--frame-model`, kept ALONGSIDE.**
One label per frame instead of per window, so a window can hold an advance AND a lunge.
Needs no new annotation (each clip is a single action, so every real frame already carries
its label) and turns 488 windows into ~20k supervised frames, which also attacks the thin
transient classes directly. 12 seeds vs the window model: bout advance 9.7% → 14.3%, bout
lunge 42.3% → 31.0% (both ~2 sigma), costing ~3 pts of held-out accuracy (86% → 83%).
Shipped checkpoint is seed 7 (window acc 89.7%): bout advance 16%, lunge 49%, parry 5%.
Verified the walking calls against frames — all sampled ones are genuine stoppages, clock
frozen at 2:42/2:34 with fencers walking back to the lines. Not a failure mode.

**DO NOT ensemble the per-frame model** (members deliberately not shipped). 5 members gave
lunge 49% → 23%, the best lunge figure measured anywhere, but advance 16% → 8% and
parry 5% → **0%**, a class gone. Averaging dilutes the probability peaks of BRIEF actions,
and every transient class here is brief, so persistent classes take every frame. Ensembling
helps the window model and harms this one.

**Selection trap — do NOT pick a checkpoint by validation accuracy.** It has now chosen a
lunge-heavy checkpoint twice: window seed 8 → 52% bout lunge, per-frame seed 7 → 49%, against
sweep averages of 42% and 31%. Val accuracy is not merely uninformative about video behaviour,
it looks mildly ANTI-correlated with it. For the window model the ensemble sidesteps this; for
the per-frame model there is currently no good selection rule, which is a real open weakness.

**STILL OPEN — the A/B asymmetry that started all this survives everything.** On the same
footage with the per-frame model, fencer A reads advance=26% / lunge=37% while fencer B reads
advance=7% / lunge=62%. Every fix so far moved the aggregate and left the split intact. The
earlier facing investigation already ruled out nose_dir instability, pose quality and facing
coverage, and canonicalisation made things worse. This is a SEPARATE problem from
advance-vs-lunge and deserves its own investigation rather than more variants of this one.

**METHOD WARNING — bout-mix numbers are noisy; do not chase small differences.** Two runs of
the IDENTICAL current config measured advance 8% vs 14% and lunge 44% vs 32% on pure training
noise. Per-seed bout advance spans 0.4%-34% at 82-92% val. Several deltas quoted earlier in
this investigation ("slice-all 12%→17%", "rate features lunge 39%→27%") sit inside that band
and should not be treated as real. Only validation-side metrics (val, val@hold, artifact gap)
were stable enough to decide on.

Also tested and REJECTED (6 seeds each, all inside noise or worse — don't re-run): shorter
SEQ_LEN with every class sliced (24f/30f drop advance recall to 51%/73%); pre-padding so the
action ends at the window end (advance recall 83% → 63%); filling that pad with real neutral
context (→ 57%); rate-normalised length-invariant sum features (val +0.8, advance recall
−13); within-window RELATIVE stance/crouch/reach instead of absolute, i.e. "how much did this
fencer change" (val 84.6% → 85.2%, bout advance 14% → 10%); all three combined.

The remaining lever is data for the transient classes. Not because of any domain gap — there
isn't one — but because 35 advance + 33 retreat + 48 lunge clips do not determine a boundary
in a 128-dim representation, and the seed spread above is that underdetermination showing.
`advance` is the smallest class (35 clips → 35 windows, not sliceable; walking gets 232
windows from 27 clips), so it is the cheapest to improve. Same broadcast sources are fine.

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
