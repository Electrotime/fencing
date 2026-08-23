# FenceVision
**THIS IS STILL WIP**

Action recognition for fencing from ordinary broadcast video. FenceVision detects both fencers in a frame and classifies what each one is doing at 20 predictions per second, using only the video feed. No sensors, no instrumented equipment, no marked piste.

Held-out accuracy is 80.2% on a bout the model never trained on, and 66-70% at venues it has never seen, compared to 16.7% for random guessing.

On the halts where both scoring lamps fire and the referee must award the touch on *right of way*, the model's action probabilities predict that decision at **0.76 AUC** (95% CI [0.57, 0.92], p = 0.003) across 38 halts in four bouts never used to select it — a pre-registered result that survived one failed replication before confirming. A companion pipeline reads the broadcast scoreboard to recover touch times and lamp colours automatically, at 104/104 on four broadcasters.

## Features

- Six-class action recognition: `advance`, `retreat`, `walking`, `neutral`, `lunge`, `parry`
- Simultaneous footwork and blade output, so a parry during a retreat is reported as both
- Opponent-aware classification: each fencer's features include their opponent's
- A learned two-term decision rule for parry, worth +13 points of recall at unchanged overall accuracy
- Mirror augmentation for left- and right-handed fencers, worth 7 points against a matched control
- Annotated video output with per-fencer overlays
- Leave-one-bout-out evaluation scripts and per-feature ablation controls
- Scoreboard and lamp reading with no OCR, recovering halt times, which lamps fired, and the scorer
- A pre-registered right-of-way test, reported with its failed replication and its failed rival hypothesis

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

Across bouts 3 to 5, 86% of labelled parries have the opponent attacking at the same moment (76% lunging, 10% advancing with an extension). That asymmetry is what makes the opponent's state usable: a parry is a response and barely happens unattacked, while a lunge only draws one about half the time. So the rule runs one way — the opponent's attack conditions parry, never the reverse.

The model is not blind to parry. Its parry probability separates parry windows at 0.82 to 0.88 AUC, so the ranking is good and the difficulty is extracting it at a 4% base rate. What changed the result was the *shape* of the decision rule.

The first version was a rectangle: call it a parry when the model's own parry probability clears one threshold **and** the opponent's lunge probability clears another. Both conditions had to hold independently. Replacing that with a single linear boundary, fitted on the same two numbers, lets them trade off — overwhelming parry evidence can carry a quiet opponent, and a committed attack can carry weak parry evidence:

    5.0858 * p_parry  +  4.0640 * opponent_lunge  >=  3.2708

| Rule | Held-out bout 4 | Mean over four held-out bouts |
|---|---|---|
| Raw classifier, no rule | 39% precision / 30% recall | 0.24 F1 |
| Rectangle (two thresholds) | 56% / 41% | 0.31 F1 |
| **Linear boundary** | **50% / 52%** | **0.38 F1** |

Parry F1 improves on all four bouts and overall accuracy is unchanged. The line passes almost exactly through the hand-tuned corner — at an opponent lunge of 0.60 it wants a parry probability of 0.164, against the 0.15 that was chosen by hand — so the old thresholds were a good point on a shape that simply could not express a trade-off.

The promoted windows are better than the average parry call, not worse. Most were previously classified as `retreat`: parrying while retreating is the common case, the legs dominate the pose signal, and the opponent's lunge is what breaks the tie.

Re-tuning the rectangle's two thresholds instead, chosen the same leave-one-bout-out way, gains nothing at all — 0.31 F1, identical to the shipped pair. So the improvement is the shape of the rule, not fresher numbers. Adding all twelve class probabilities, or lagged opponent history, does not beat the two features the rule already uses.

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
| 4 | 73.4% | 79.0% | 50% / 52% |
| 5 | 71.8% | 75.8% | 25% / 43% |
| 6 (unseen venue) | 73.7% | — | 36% / 26% |
| 7 (unseen venue) | 69.1% | 74.3% | 51% / 30% |

¹ The overlay renders `neutral` and `walking` identically, as "ready". Display accuracy scores what the viewer sees, so a neutral/walking mix-up is not counted as an error.

² Bout 1 contains 8 parry windows, too few to mean anything. Bouts 4 and 7 carry 207 and 204.

**Dataset:** 7 bouts across 4 venues, 3050 seconds of hand-labelled footage, 1365 labelled intervals, 16300 training windows. Only 126 seconds are `parry`, which is the main source of difficulty.

**Touch outcomes** are labelled separately from actions: bouts 4-7 exhaustively (159 halts, every stoppage including off-target), bouts 8-9 for contested halts only (22 two-lamp halts), which is the 26% of the work the scoreboard reader cannot do itself. Nine bouts of video in total; the action model still trains on seven.

## Reading the scoreboard

A second pipeline reads the broadcast graphics instead of the fencers. It recovers when a halt happened, which lamps fired, and who was awarded the touch — with no OCR and no hand labels. The digits are fixed-position, so comparing digit *images* is cheaper and more accurate than recognising characters.

Every production renders the scoring lamps differently: pill borders on one, the venue's LED strip on another, wide bars, score pills, permanent chevrons, a sliding name banner, an AR overlay projected on the piste. Locating the indicator is the only per-broadcast setup, and it is done by averaging frames in a window around known halts and subtracting the rest of the video, which puts the lamp at the peak of the difference.

Four things had to be right, and each came from looking at frames rather than tuning:

- **`min(B,G,R)`, not grayscale.** The indicator glows with the lamp colour on every touch, and in grayscale that glow is bright enough to read as a score change. Digits are white and the glow is saturated, so the minimum channel keeps one and drops the other.
- **A presence anchor.** The scorebug slides out and a sponsor bar slides through the same band. Without checking the timer panel, those animations are indistinguishable from touches.
- **Removing the always-lit furniture.** The score pill contains a bright specular arc. Left in, it is most of the mask, and swapping `0` for `5` moves under 4% of pixels — beneath compression noise.
- **Novelty.** A score only rises, one at a time, so a digit image a side has already held cannot be its new score. This alone cut spurious detections from 62 to 7.

Validated against hand-labelled touches on four bouts and four broadcasters:

| | |
|---|---|
| scoring touches found | **104 / 104** |
| lamp colour correct | **104 / 104** |
| single-lamp halts, "that side scores" | **56 / 56** |

The lockout matters more than the rulebook implies. Foil locks out at 300 ms, but the graphic lags: across 38 two-lamp halts the second lamp trails the first by a median of 0.20 s, a 90th percentile of 1.20 s, and a maximum of 1.80 s. A detector closing its window at 0.5 s reads a third of all doubles as singles.

Two things it does not do. Off-target hits show a **white** lamp: one production gives them a dedicated bar and the reader finds them on 8 of 10 halts, another has no per-side white indicator and brightens globally at every stoppage, where four separate statistics all failed to separate off-target from valid. And **replays re-fire the graphic**, so the broadcast showing a touch again looks like a second touch — 48 of 51 false positives on one bout sit within 30 s of the real halt.

The practical output is that hand-labelling drops to the contested halts only: **41 calls against 159 hand-labelled halts, 26% of the work.**

## Right of way

In foil, when both lamps fire the touch is awarded on *priority*: to whoever was attacking, and if both were, to whoever went forward first. That makes contested halts the only ones where an action model can contribute at all — a single lamp already names the scorer, and 54% of touches in this corpus are single-lamp.

A pre-registered test asks whether the model's action probabilities carry that decision. The feature is the difference between the two fencers' peak `advance` probability, averaged across four lookback windows so that no window length is chosen after the fact.

| bouts | role | n | AUC | one-sided p |
|---|---|---|---|---|
| 4, 7 | discovery | 19 | 0.83 | — |
| 5, 6 | confirmation | 13 | 0.75 | 0.084 |
| 8, 9 | confirmation | 14 | 0.80 | 0.036 |
| **5, 6, 8, 9** | **pooled** | **38** | **0.76** | **0.0033** |

95% CI [0.57, 0.92], which excludes chance. The pooled row is the result, not bouts 8–9 alone: the hypothesis was tested twice on held-out data, missing once and hitting once, and reporting only the hit would be selection. It survives correction for both registered features, holds under leave-one-bout-out on all four bouts, and every one of the six bouts is individually above chance (0.68 to 0.96).

Three things it is not:

1. **Not a decision rule.** A threshold fitted on the discovery bouts scores 82% there and **50%** on confirmation — worse than always picking the more common side. Per-bout offsets range from -0.204 to +0.029, so the ranking transfers and the boundary does not. Z-scoring within a bout, which needs no labels, gives 71% against a 58% baseline — on 38 halts, a margin of five calls.
2. **Not order.** The rule turns on who moved *first*, so a second feature was registered *before the confirmation data existed*: the time-centroid of each fencer's advance probability. It scored **0.38**, below chance. The model sees who is attacking, not who started; a 2-second window smears an onset the referee resolves in tenths of a second.
3. **Not deployable.** The priority label is *derived from* the scoreboard, and the contested subset is *defined by* the lamps. This measures that pose carries information about right of way. It does not replace reading the scoreboard, and a rule that needs the lamps to know which halts to apply to cannot be used where the lamps are missing.

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

1. **Parry recall is 26 to 52% depending on the bout.** Precision is acceptable now, but most real parries are still missed, and neither more labels, a separate blade head, nor a dedicated binary parry head has moved it. The remaining ideas all need a better view of the blade rather than better use of the current one. Broadcast and standard videos are limited by a low FPS and high shutter speed, causing blurring or disappearing blades, which cannot be accurately used to detect a parry. 
2. **Cross-venue costs 6 to 10 points, and the price varies by venue.** Two venues have now been held out from the same training set and scored independently: 66.4% and 70.2%, against 75.5-80.2% on a familiar bout. Roughly a minute of labelled footage from the target venue closes most of the gap. Adding a third venue to training does not improve transfer to a fourth, so venue diversity in training is not the lever.
3. **Motion features degrade off-venue.** The fix is a better pan estimate or a camera-invariant reformulation, not more data.
4. **Broadcast filler is not filtered.** 28% of predictions over replays and crowd shots display a real action. Geometry-based gating caps at 36% precision, because a replay of a touch is geometrically identical to the touch.
5. **One fencer's rear arm is invisible to the camera.** Where the sword arm is hidden behind the torso, accuracy falls from 81.5% to 47.8% for the same fencer. This is a limit of the camera position rather than of the model, and suppressing predictions when the arm is hidden was tested and does not generalise across bouts.
6. **Off-target lamp detection does not transfer between broadcasts.** One production gives the white lamp its own indicator and it reads at 8/10; another has none and brightens globally at every halt, where four separate statistics failed. Locating any lamp needs either labels to contrast against or a human to look — two attempts at unsupervised localisation both found LED floods and permanent graphics instead.
7. **The right-of-way result is a measurement, not a capability.** Its label is derived from the scoreboard and its subset is defined by the lamps, so it cannot be applied where the lamps are unreadable. No transferable decision threshold exists: the ranking generalises, the boundary does not.

## Project structure

```
fencing/
  src/         action_model.py (features, LSTM), pose_pipeline.py,
               person_detector.py, labels.py
  scripts/     demo_video.py (inference loop), train_shipping.py,
               evaluate_labels.py, sweep_parry_promote.py,
               read_scoreboard.py (lamps, halts, score digits),
               check_touches.py (touch-table validator),
               label_worklist.py (contested-halt worklist),
               exp_touch_probe.py / exp_contested.py (right-of-way tests),
               venue_motion.py, bout_timeline.py
  data/        raw_video/, labels/ (interval CSVs), train_continuous/ (cached windows)
  models/      action_mirror7.pth (shipped checkpoint)
```

## Built with

Python, PyTorch, MediaPipe Pose, Ultralytics YOLOv8, OpenCV, NumPy
