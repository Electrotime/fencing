# FenceVision
**WIP**

Action recognition for fencing from ordinary broadcast video. FenceVision detects both fencers in a frame and classifies what each one is doing at 20 predictions per second, using only the video feed. No sensors, no instrumented equipment, no marked piste.

Held-out accuracy is 76.6% on a bout the model never trained on, compared to 16.7% for random guessing.

## Features

- Six-class action recognition: `advance`, `retreat`, `walking`, `neutral`, `lunge`, `parry`
- Simultaneous footwork and blade output, so a parry during a retreat is reported as both
- Opponent-aware classification: each fencer's features include their opponent's
- Rule-based parry gating that raises parry precision from 29% to 58%
- Annotated video output with per-fencer overlays
- Leave-one-bout-out evaluation scripts and per-feature ablation controls

## Installation

```bash
git clone https://github.com/Electrotime/fencing.git
cd fencing/fencing
pip install -r requirements.txt
```

Python 3.9+ is required. The shipped checkpoint is `models/action_opp5.pth`. Every path
below is relative to the inner `fencing/` directory, which is where the package, scripts
and data live.

## Usage

Run inference on a video segment and write an annotated output file:

```bash
python scripts/demo_video.py data/raw_video/5.mp4 out.mp4 --start 300 --end 320
```

Score a model against a labelled interval CSV:

```bash
python scripts/evaluate_labels.py data/raw_video/1.mp4 data/labels/bout1_intervals.csv \
    --model verify_h1_b5.pth --no-prior
```

Most scripts accept `--self-test` and document their evaluation setup in the module docstring.

## How it works

Detection runs in two stages. YOLOv8n locates the two fencers and returns a bounding box for each, then MediaPipe Pose extracts 33 landmarks per box. The landmarks are hip-centred and torso-normalised, so measurements taken from them do not depend on camera distance or framing.

Each fencer's normalised skeleton feeds two paths. A rolling 60-frame sequence (2 seconds of motion) goes into a 128-unit LSTM, and six engineered features are computed alongside it: net forward movement, stance width, wrist speed, total travel, arm reach, and knee angle. The LSTM output and the feature vector are concatenated at the classifier head, which produces probabilities over the six classes. A rule-based gate then adjusts parry predictions using the opponent's state.

Three choices account for most of the accuracy. First, the LSTM output is reduced with `last` rather than `mean`, worth 4 to 5 points: a parry lasts about 0.6 s inside a 2 s window, so averaging buries it under the rest of the window. Second, each fencer's feature vector is concatenated with their opponent's, as `[own(6) | opponent(6) | present(1)]`, because a retreat means something different when the other fencer is lunging. Third, normalising the skeletons keeps the posture features camera-invariant, which is why they generalise across venues while raw motion features do not (see [Cross-venue behaviour](#cross-venue-behaviour)).

## Parry detection

Parry is the hardest class: brief, small, and physically overlapping with footwork. The raw classifier ran at 29% precision, which makes an on-screen indicator worse than useless.

Across bouts 3 to 5, 86% of labelled parries have the opponent attacking at the same moment (76% lunging, 10% advancing with an extension). Both directions of that correlation are used:

| Rule | Condition | Effect on held-out bout 4 |
|---|---|---|
| Veto | Parry predicted, opponent not attacking, demote | Precision 29% -> 55% |
| Promote | Parry lost the argmax, opponent clearly lunging, call it | Recall 15% -> 29%, precision -> 58% |

Precision rises while recall doubles because the promoted windows are better than the average parry call: 29 of 45 are true parries. Of those 45, 37 were previously classified as `retreat`. Parrying while retreating is the common case, the legs dominate the pose signal, and the opponent's lunge is what breaks the tie.

## Results

All numbers are leave-one-bout-out. The model never trained on the bout it is scored on.

| Checkpoint | Configuration | Held-out bout 1 |
|---|---|---|
| `action_lstm` | Hand-cut clips, mean pooling | 43.4% |
| `action_cont` | Continuous windows, last pooling | 74.0% |
| `action_opp` | Plus opponent context | 74.6% |
| `action_opp5` | Plus a second venue | **76.6%** |

Per held-out bout, with the full decision path:

| Bout | Overall | Display accuracy¹ | Parry precision / recall |
|---|---|---|---|
| 1 | 76.6% | 83.2% | n/a |
| 4 | 72.7% | 79.8% | 58% / 29% |
| 5 (unseen venue) | 59.7% | 67.8% | 29% / 24% |

¹ The overlay renders `neutral` and `walking` identically, as "ready". Display accuracy scores what the viewer sees, so a neutral/walking mix-up is not counted as an error.

**Dataset:** 5 bouts across 2 venues, 1621 seconds of hand-labelled footage, 733 labelled intervals, 8430 training windows. Only 64 seconds are `parry`, which is the main source of difficulty.

## Evaluation protocol

Ten measured improvements were later retracted during development. The full log is in `CLAUDE.md`. The protocol below exists because of them.

- **Leave-one-bout-out, never a random split.** Windows are emitted every 5 frames from a 60-frame span, so adjacent windows share 92% of their frames and a random split reports fiction.
- **Shuffled control for every new feature.** Adding an input also widens the classifier head, so part of any gain is capacity rather than information. Permuting the new column within each bout and re-running separates the two. Blade energy scored +1.6 points and was positive on both held-out bouts; the shuffled control reproduced the entire effect.
- **Matched control for every new decision rule.** The parry promoter was scored against a rule promoting the same number of windows by own-probability alone: 37% precision against 58%. Same budget, one variable.
- **Pre-registered selection rules.** Tuning bout, confirmation bout, criterion and veto are written into the script docstring before it runs.
- **Absolute correct calls when coverage changes.** Scoring more windows means scoring harder ones. A 720p downscale raised accuracy from 76.6% to 77.7% while losing 2% of correct calls, purely by detecting fencers on fewer frames.
- **AUC is only valid for the unit it was measured on.** Blade energy separates parry intervals at 0.79 AUC and parry windows at chance, because a 0.6 s action inside a 2 s window mostly reports the window.

The retractions fell into three patterns: measuring how the training clips were cut rather than the fencing (correcting zero-padding dropped in-sample accuracy from 85% to 53%); generalising from a single bout (blade energy scored 0.70 AUC on bout 2 with n=4 parries and 0.49 on bout 1); and comparing a threshold against the wrong quantity (a branch marked unreachable turned out to be firing on real parries when instrumented).

## Cross-venue behaviour

The largest single error at an unfamiliar venue is `advance` misclassified as `walking`. The initial explanation was that a different camera height shifts the knee-angle measurement. Measurement ruled that out:

| Advance vs walking, AUC | Bout 1 | Bout 2 | Bout 3 | Bout 4 | Bout 5 (venue B) |
|---|---|---|---|---|---|
| `crouch` (posture) | 0.92 | 0.87 | 0.95 | 0.86 | 0.79 |
| `stance` (posture) | 0.82 | 0.84 | 0.85 | 0.82 | 0.74 |
| `net_forward` (motion) | 0.73 | 0.69 | 0.78 | 0.73 | 0.62 |
| `total_travel` (motion) | 0.49 | 0.56 | 0.63 | 0.63 | 0.34 |

Crouch is the strongest feature at the new venue, with a knee-angle gap (18.7°) close to the familiar venue's (17.2°). The motion features are what break, and `total_travel` inverts rather than weakening.

The split follows from how the features are built. Posture features use hip-centred, torso-normalised skeletons and are camera-invariant by construction. Motion features use `world_vel = diff(hip_x) - pan/PAN_WIDTH`, where the pan term is an estimate of camera movement. Venue B's camera pans 1.7 times harder than venue A's and is still panning only 16% of the time against 28%.

## Known limitations

1. **Parry recall is 29%.** Precision is acceptable now, but two of every three real parries are still missed.
2. **Cross-venue accuracy is unmeasured for the shipped model.** The 58.1% figure is historical, from a four-bout model at what was then an unfamiliar venue. That measurement was spent when the venue entered the training set. A third venue is needed for a new one.
3. **Motion features degrade off-venue.** The fix is a better pan estimate or a camera-invariant reformulation, not more data.
4. **Broadcast filler is not filtered.** 28% of predictions over replays and crowd shots display a real action. Geometry-based gating caps at 36% precision, because a replay of a touch is geometrically identical to the touch.

## Project structure

```
fencing/
  src/         action_model.py (features, LSTM), pose_pipeline.py,
               person_detector.py, labels.py
  scripts/     demo_video.py (inference loop), train_shipping.py,
               evaluate_labels.py, sweep_parry_promote.py,
               venue_motion.py, bout_timeline.py
  data/        raw_video/, labels/ (interval CSVs), train_continuous/ (cached windows)
  models/      action_opp5.pth (shipped checkpoint)
  CLAUDE.md    full findings log, including retracted results
```

## Built with

Python, PyTorch, MediaPipe Pose, Ultralytics YOLOv8, OpenCV, NumPy
