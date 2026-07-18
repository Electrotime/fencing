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

**Post-processing rules:**
1. Any keypoint with visibility < 0.5 → replace its x,y,z with the previous frame's values (carry-forward interpolation). For the first frame, use zeros.
2. Normalize all x,y coordinates relative to the hip midpoint (average of landmarks 23 and 24). After normalization, the hip midpoint is (0, 0).
3. Scale by dividing by the torso length (shoulder midpoint to hip midpoint) so the representation is size-invariant. (Originally shoulder width, landmarks 11-12 — changed 2026-07 after measurement: fencers stand side-on, so projected shoulder width collapses toward zero and amplified coordinates up to ~300x. Torso length is rotation-stable.)
4. Do NOT normalize z or visibility — keep them raw.

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
HIDDEN_SIZE = 128
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

**Updated 2026-07-18 — Phases 0–4 running. First action model trained (70% val acc on 53 clips); growing the clip dataset is the current focus.**

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

**Current focus: grow the clip dataset.** Target ~30-40 clips/class; priority parry, then
lunge/retreat. Known-weak clips to re-crop or replace: advance `New 11`/`New 12`, lunge
`New 5 (1)`/`New 8 (1)` (fencer too small in crop — MediaPipe misses or half-tracks them).
Clip-cutting rules learned the hard way: crop tight (fencer ≥ half frame height), keep the
action inside the first 2 s, cut every class with the same ~0.5 s lead-in / 0.3 s lead-out,
label by the clip's dominant action. Clip videos are gitignored (local/OneDrive only); the
extracted `data/keypoints/*.npy` are tracked in git. Banners with printed fencers can fool the
YOLO person detector (not MediaPipe pose), so eyeball auto-crops before trusting them.

When the user says something like "let's start" or "let's do Phase X", begin by:
1. Stating which phase you're working on
2. Listing the files you're about to create or modify
3. Writing the code
4. Telling the user exactly what command to run to test it

When you are unsure whether something will work (library version, hardware capability, API change), say so explicitly rather than guessing.
