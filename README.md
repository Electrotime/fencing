# FenceVision
**WIP, things might NOT work**

Action recognition for fencing from ordinary broadcast video. FenceVision detects both fencers in a frame and classifies what each one is doing at 20 predictions per second, using only the video feed. No sensors, no instrumented equipment, no marked piste.

Held-out accuracy is 80.2% on a bout the model never trained on, and 66-70% at venues it has never seen, compared to 16.7% for random guessing.

## Features

- Six-class action recognition: `advance`, `retreat`, `walking`, `neutral`, `lunge`, `parry`
- Simultaneous footwork and blade output, so a parry during a retreat is reported as both
- Opponent-aware classification: each fencer's features include their opponent's
- Rule-based parry gating that raises parry precision from 29% to 56%
- Mirror augmentation for left- and right-handed fencers, worth 7 points against a matched control
- Annotated video output with per-fencer overlays
- Leave-one-bout-out evaluation scripts and per-feature ablation controls

## Installation

```bash
git clone https://github.com/Electrotime/fencing.git
cd fencing/fencing
pip install -r requirements.txt
```

Python 3.9+ is required. The shipped checkpoint is `models/action_mirror7.pth`. Every path
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

Most scripts accept `--self-test`, which runs their built-in assertions without touching any video.

## How it works

Detection runs in two stages. YOLOv8n locates the two fencers and returns a bounding box for each, then MediaPipe Pose extracts 33 landmarks per box. The landmarks are hip-centred and torso-normalised, so measurements taken from them do not depend on camera distance or framing.

Each fencer's normalised skeleton feeds two paths. A rolling 60-frame sequence (2 seconds of motion) goes into a 128-unit LSTM, and six engineered features are computed alongside it: net forward movement, stance width, wrist speed, total travel, arm reach, and knee angle. The LSTM output and the feature vector are concatenated at the classifier head, which produces probabilities over the six classes. A rule-based gate then adjusts parry predictions using the opponent's state.

Three choices account for most of the accuracy. First, the LSTM output is reduced with `last` rather than `mean`, worth 4 to 5 points: a parry lasts about 0.6 s inside a 2 s window, so averaging buries it under the rest of the window. Second, each fencer's feature vector is concatenated with their opponent's, as `[own(6) | opponent(6) | present(1)]`, because a retreat means something different when the other fencer is lunging. Third, normalising the skeletons keeps the posture features camera-invariant, which is why they generalise across venues while raw motion features do not (see [Cross-venue behaviour](#cross-venue-behaviour)).

A fourth, added later, is mirror augmentation. Normalisation already removes translation and scale, so cropping or zooming the video produces near-identical tensors, but it does not remove handedness: which arm extends toward the opponent survives every normalisation step. The training corpus happened to contain only one handedness in its right-hand slot, and a left-handed fencer at a new venue scored 35% while their opponent scored 71%. Mirroring the pose sequences fixed that specific gap and helped generally, because the six engineered features are provably mirror-invariant, so only the sequence the LSTM reads is flipped.

## Parry detection

Parry is the hardest class: brief, small, and physically overlapping with footwork. The raw classifier ran at 29% precision, which makes an on-screen indicator worse than useless.

Across bouts 3 to 5, 86% of labelled parries have the opponent attacking at the same moment (76% lunging, 10% advancing with an extension). Both directions of that correlation are used:

| Rule | Condition | Effect on held-out bout 4 |
|---|---|---|
| Veto | Parry predicted, opponent not attacking, demote | Precision 29% -> 54% |
| Promote | Parry lost the argmax, opponent clearly lunging, call it | Recall 25% -> 41%, precision 54% -> 56% |

Precision rises while recall grows because the promoted windows are better than the average parry call, not worse. Most of them were previously classified as `retreat`: parrying while retreating is the common case, the legs dominate the pose signal, and the opponent's lunge is what breaks the tie.

Two alternatives were tested against this rule and both lost. Promoting by the model's own parry probability alone, matched on the number of promotions, reaches 36% precision against 49%: the opponent's state is doing the work, not the lower threshold. A dedicated binary parry head, trained on a balanced problem and again matched on promotion count, is worse than the six-way model's own parry probability on all four bouts that have enough parries to measure. The six-way probability appears to encode "parry rather than retreat", which is exactly the comparison the rule needs, and a parry-versus-everything head discards it.

## Results

All numbers are leave-one-bout-out. The model never trained on the bout it is scored on.

| Checkpoint | Configuration | Held-out bout 1 |
|---|---|---|
| `action_lstm` | Hand-cut clips, mean pooling | 43.4% |
| `action_cont` | Continuous windows, last pooling | 74.0% |
| `action_opp` | Plus opponent context | 74.6% |
| `action_opp5` | Plus a second venue | 76.6% |
| `action_mirror` | Plus a third venue and mirror augmentation | **80.2%** |

The shipped checkpoint, `action_mirror7`, adds a fourth venue on top of that recipe. It is not in the table because it cannot be scored honestly on any bout in this corpus: all seven are in its training set. Its justification is a matched A/B instead. Adding the seventh bout is worth +6.6, -0.1, +1.1 and +2.4 points on four separately held-out bouts, a mean of +2.5 and never negative beyond seed noise.

Per held-out bout, with the full decision path:

| Bout | Overall | Display accuracy¹ | Parry precision / recall |
|---|---|---|---|
| 1 | 80.2% | 81.7% | 25% / 25%² |
| 4 | 73.4% | 79.0% | 56% / 41% |
| 5 | 72.0% | 75.8% | 22% / 24% |
| 7 (unseen venue) | 68.8% | 74.3% | 54% / 18% |

¹ The overlay renders `neutral` and `walking` identically, as "ready". Display accuracy scores what the viewer sees, so a neutral/walking mix-up is not counted as an error.

² Bout 1 contains 8 parry windows, too few to mean anything. Bouts 4 and 7 carry 207 and 204.

**Dataset:** 7 bouts across 4 venues, 3050 seconds of hand-labelled footage, 1365 labelled intervals, 16300 training windows. Only 126 seconds are `parry`, which is the main source of difficulty.

## Evaluation protocol

Eleven measured improvements were later retracted during development. The protocol below exists because of them.

- **Leave-one-bout-out, never a random split.** Windows are emitted every 5 frames from a 60-frame span, so adjacent windows share 92% of their frames and a random split reports fiction.
- **Shuffled control for every new feature.** Adding an input also widens the classifier head, so part of any gain is capacity rather than information. Permuting the new column within each bout and re-running separates the two. Blade energy scored +1.6 points and was positive on both held-out bouts; the shuffled control reproduced the entire effect.
- **Matched control for every new decision rule.** The parry promoter was scored against a rule promoting the same number of windows by own-probability alone: 36% precision against 49%. Same budget, one variable.
- **Pre-registered selection rules.** Tuning bout, confirmation bout, criterion and veto are fixed before the script runs. This is what kept the shipped parry thresholds unchanged when they were re-swept against a new checkpoint: a lower setting won on the tuning bout, then tied on the confirmation bout while costing precision, so it was not adopted.
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

A third venue then showed that most of what looked like a venue effect was not one. Scoring the two fencers separately instead of scoring the bout as a whole, one of them transferred at 71% while the other collapsed to 35% — and the difference was handedness, not the camera. Mirror augmentation recovered it, and against a control trained on the same doubled window count with identical rather than mirrored copies, mirroring is positive or neutral on every bout for a mean of 7 points. Reporting it against the control matters: on one bout the doubling alone costs 4.3 points and mirroring lands on exactly the same number, so without the control that bout would read as a regression caused by mirroring.

What remains after that is a genuine but cheap domain gap. Training on the first half of an unseen venue and testing on its second half:

| Target footage labelled | Venue C only | Plus the other venues |
|---|---|---|
| none | — | 54.1% |
| 35 s | 44.9% | 63.4% |
| 78 s | 60.5% | 66.5% |
| 199 s | 65.5% | 69.5% |
| 393 s | 68.5% | 69.3% |

Roughly 35 seconds of labelled footage from a new venue is worth 9 points, and the curve is flat past about three minutes. The existing corpus is what makes that work: at the smallest slice, target footage alone reaches 27% while the same slice combined with the other venues reaches 61%. Other venues are a prior, not a substitute. The practical consequence is that deploying at a new venue is a labelling task with a small known price rather than a research problem.

## Known limitations

1. **Parry recall is 41%.** Precision is acceptable now, but most real parries are still missed, and neither more labels, a separate blade head, nor a dedicated binary parry head has moved it. The remaining ideas all need a better view of the blade rather than better use of the current one.
2. **Cross-venue costs 6 to 10 points, and the price varies by venue.** Two venues have now been held out from the same training set and scored independently: 66.4% and 70.2%, against 75.5-80.2% on a familiar bout. Roughly a minute of labelled footage from the target venue closes most of the gap. Adding a third venue to training does not improve transfer to a fourth, so venue diversity in training is not the lever.
3. **Motion features degrade off-venue.** The fix is a better pan estimate or a camera-invariant reformulation, not more data.
4. **Broadcast filler is not filtered.** 28% of predictions over replays and crowd shots display a real action. Geometry-based gating caps at 36% precision, because a replay of a touch is geometrically identical to the touch.
5. **One fencer's rear arm is invisible to the camera.** Where the sword arm is hidden behind the torso, accuracy falls from 81.5% to 47.8% for the same fencer. This is a limit of the camera position rather than of the model, and suppressing predictions when the arm is hidden was tested and does not generalise across bouts.

## Project structure

```
fencing/
  src/         action_model.py (features, LSTM), pose_pipeline.py,
               person_detector.py, labels.py
  scripts/     demo_video.py (inference loop), train_shipping.py,
               evaluate_labels.py, sweep_parry_promote.py,
               venue_motion.py, bout_timeline.py
  data/        raw_video/, labels/ (interval CSVs), train_continuous/ (cached windows)
  models/      action_mirror7.pth (shipped checkpoint)
```

## Built with

Python, PyTorch, MediaPipe Pose, Ultralytics YOLOv8, OpenCV, NumPy
