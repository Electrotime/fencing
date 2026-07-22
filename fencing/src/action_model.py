"""Phase 4: LSTM that watches a keypoint sequence and names the fencing action.

Hybrid design (measured, 2026-07): the LSTM reads the raw keypoint sequence, and
four engineered clip-level numbers go straight into the classifier head. Those
four carry signals the LSTM provably couldn't dig out of 132 channels with a
dataset this size -- adding them took validation accuracy from ~51% to ~74% and
retreat recall from ~3% to ~67% in 10-seed ablations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

CLASS_NAMES = ["advance", "lunge", "parry", "retreat"]
SEQ_LEN = 60          # every clip gets padded/trimmed to this many frames
INPUT_SIZE = 132      # 33 landmarks x 4 values, flattened per frame
N_AGG_FEATURES = 4    # engineered clip-level features fed straight into the head
HIDDEN_SIZE = 64      # measured: this small net beats the original 2x128 at 83 clips
NUM_CLASSES = 4
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
MIN_RECALL = 0.70     # below this, a class needs more training clips

NOSE = 0
WRIST_LEFT, WRIST_RIGHT = 15, 16
ANKLE_LEFT, ANKLE_RIGHT = 27, 28


def _pick_device() -> torch.device:
    """cuda on the gaming pc, mps on the mac, cpu otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


PAN_WIDTH = 320.0     # px width the pan was measured at (see pose_pipeline.PAN_DOWNSCALE)
FORWARD_SCALE = 10.0  # lifts fraction-of-width motion into a ~[-3, 3] feature range


def _engineered_features(kp: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Four clip-level numbers that decide the classes the raw sequence can't.

    kp: (n, 33, 4) normalized keypoints with the lock-on zeros already stripped.
    motion: (n, 2) = [background pan px, raw hip-x fraction], aligned to the clip end.

    - net forward motion: the fencer's WORLD travel = how she crossed the frame
      (hip-x) plus how far the camera panned to follow her, signed by facing
      direction. + advancing, - retreating. Combining both terms (not pan alone)
      lifted advance/retreat direction accuracy 84% -> 94% across camera styles.
    - stance width p90: the lunge's wide split, robust to single glitch frames.
    - wrist speed p90: blade-hand activity, the parry signature.
    - total travel: lots of it = footwork, little = blade action on the spot.
    """
    n = len(kp)
    if n < 2:
        return np.zeros(N_AGG_FEATURES, dtype=np.float32)

    # align the motion track to the stripped keypoints (strip removed leading
    # frames, so the END of the motion array lines up with the END of the clip)
    motion = np.atleast_2d(motion)
    if motion.shape[1] != 2:  # tolerate a legacy 1-col pan file
        motion = np.stack([motion.ravel(), np.zeros(len(motion.ravel()))], axis=1)
    pan = np.zeros(n, dtype=np.float32)
    hip_x = np.zeros(n, dtype=np.float32)
    m = min(n, len(motion))
    if m:
        pan[:m] = motion[len(motion) - m:, 0]
        hip_x[:m] = motion[len(motion) - m:, 1]
    if n >= 3:  # de-spike, same as the keypoints
        pan[1:-1] = np.median(np.stack([pan[:-2], pan[1:-1], pan[2:]]), axis=0)
        hip_x[1:-1] = np.median(np.stack([hip_x[:-2], hip_x[1:-1], hip_x[2:]]), axis=0)

    # per-frame world velocity = fencer's frame motion + camera motion (-pan).
    # both in fraction-of-width units so they add on the same scale.
    world_vel = np.diff(hip_x) - pan[1:] / PAN_WIDTH
    nose_dir = float(np.sign(np.median(kp[:, NOSE, 0])) or 1.0)  # which way she faces
    forward = world_vel * nose_dir * FORWARD_SCALE               # + advancing, - retreating

    stance = np.abs(kp[:, ANKLE_LEFT, 0] - kp[:, ANKLE_RIGHT, 0])
    wrist_step = np.maximum(
        np.linalg.norm(np.diff(kp[:, WRIST_LEFT, :2], axis=0), axis=1),
        np.linalg.norm(np.diff(kp[:, WRIST_RIGHT, :2], axis=0), axis=1),
    )

    return np.array([
        float(np.clip(forward.sum(), -3.0, 3.0)),
        float(np.percentile(stance, 90)),
        float(np.clip(np.percentile(wrist_step, 90), 0, 3)),
        float(np.clip(np.abs(forward).sum(), 0, 6.0)),
    ], dtype=np.float32)


class FencingDataset(Dataset):
    """All the keypoint clips under keypoints_dir/<action>/, one sample per clip.

    Every sample is (sequence, engineered_features, label):
      sequence  (SEQ_LEN, INPUT_SIZE) -- padded/trimmed keypoints
      engineered (N_AGG_FEATURES,)    -- see _engineered_features
    Short clips get zero-padded at the end. Long clips keep their first SEQ_LEN
    frames, since the start of an action is the part that identifies it.
    """

    def __init__(self, keypoints_dir: str | Path) -> None:
        keypoints_dir = Path(keypoints_dir)
        self.samples: list[tuple[Path, int]] = []
        for label, action in enumerate(CLASS_NAMES):
            for npy in sorted((keypoints_dir / action).glob("*.npy")):
                if npy.name.endswith(".pan.npy"):
                    continue  # companion files, not samples
                self.samples.append((npy, label))
        if not self.samples:
            raise FileNotFoundError(
                f"no .npy files under {keypoints_dir} - run scripts/process_clips.py first")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        path, label = self.samples[idx]
        kp = np.load(path).astype(np.float32)        # (n_frames, 33, 4)
        motion_path = path.with_name(path.stem + ".pan.npy")   # (n, 2): [pan, hip-x]
        motion = (np.load(motion_path).astype(np.float32)
                  if motion_path.exists() else np.zeros((len(kp), 2), dtype=np.float32))

        # mediapipe can take a while to find the fencer at the start of a clip,
        # which leaves all-zero frames up front. drop those so zeros only ever
        # appear as padding at the end, never as fake "action" at the start
        real = np.any(kp.reshape(len(kp), -1) != 0, axis=1)
        if real.any():
            kp = kp[np.argmax(real):]

        agg = _engineered_features(kp, motion)

        flat = kp.reshape(len(kp), -1)               # (n_frames, 132)
        if len(flat) >= SEQ_LEN:
            flat = flat[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - len(flat), INPUT_SIZE), dtype=np.float32)
            flat = np.concatenate([flat, pad])
        return torch.from_numpy(flat), torch.from_numpy(agg), label


class ActionLSTM(nn.Module):
    """LSTM over the keypoint sequence, engineered clip stats joined at the head.

    (batch, SEQ_LEN, INPUT_SIZE) + (batch, N_AGG_FEATURES) -> (batch, NUM_CLASSES)
    raw logits. No softmax here, CrossEntropyLoss wants the logits as-is.
    Mean-pooled over time (a lunge is a brief peak somewhere in the window, the
    final hidden state alone tended to forget it).
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + N_AGG_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor, agg: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(torch.cat([out.mean(dim=1), agg], dim=-1))


def train_action_model(
    keypoints_dir: str | Path,
    save_path: str | Path,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
) -> dict:
    """Full training loop. Keeps whichever epoch scored best on validation.

    Returns {"train_losses": [...], "val_losses": [...], "val_accuracies": [...]}.
    """
    device = _pick_device()
    dataset = FencingDataset(keypoints_dir)

    n_val = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    if n_train < 1:
        raise ValueError(f"only {len(dataset)} clip(s) total, not enough to train "
                         "and still hold some out for validation")
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    model = ActionLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.CrossEntropyLoss()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict = {"train_losses": [], "val_losses": [], "val_accuracies": []}
    best_acc = -1.0

    print(f"{len(dataset)} clips ({n_train} train / {n_val} val), device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, agg, yb in train_dl:
            x, agg, yb = x.to(device), agg.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, agg), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        train_loss = running / n_train

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, agg, yb in val_dl:
                x, agg, yb = x.to(device), agg.to(device), yb.to(device)
                logits = model(x, agg)
                val_loss += loss_fn(logits, yb).item() * len(x)
                correct += int((logits.argmax(1) == yb).sum())
        val_loss /= n_val
        val_acc = correct / n_val

        history["train_losses"].append(train_loss)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        note = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            note = "  <- best so far, saved"
        print(f"epoch {epoch:3d}/{epochs}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  val acc {val_acc:.1%}{note}")

    _print_val_report(save_path, val_dl, device)
    return history


def _print_val_report(weights_path: Path, val_dl: DataLoader, device: torch.device) -> None:
    """Reload the best weights and show per-class precision/recall on validation."""
    # imported here so loading this module for inference doesn't need sklearn
    from sklearn.metrics import classification_report, recall_score

    model = ActionLSTM().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    with torch.no_grad():
        for x, agg, yb in val_dl:
            logits = model(x.to(device), agg.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(yb.tolist())

    labels = list(range(NUM_CLASSES))
    print("\nValidation report (best checkpoint):")
    print(classification_report(y_true, y_pred, labels=labels,
                                target_names=CLASS_NAMES, zero_division=0))

    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    weak = [name for name, r in zip(CLASS_NAMES, recalls) if r < MIN_RECALL]
    if weak:
        print(f"recall under {MIN_RECALL:.0%} for: {', '.join(weak)} - "
              "collect more clips for those actions and retrain")


if __name__ == "__main__":
    import shutil
    import tempfile

    # test 1: model shapes
    model = ActionLSTM()
    fake_seq = torch.randn(4, SEQ_LEN, INPUT_SIZE)
    fake_agg = torch.randn(4, N_AGG_FEATURES)
    out = model(fake_seq, fake_agg)
    assert out.shape == (4, NUM_CLASSES), out.shape
    print("test 1 ok: model maps (4, 60, 132) + (4, 4) -> (4, 4)")

    # test 2: dataset padding/trimming with synthetic clips of awkward lengths
    tmp = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(0)
        for action in CLASS_NAMES:
            (tmp / action).mkdir()
            for i, n_frames in enumerate([20, 60, 90]):
                np.save(tmp / action / f"fake_{i}.npy",
                        rng.normal(size=(n_frames, 33, 4)).astype(np.float32))
                if i == 0:  # give one clip a motion track, others test the missing path
                    np.save(tmp / action / f"fake_{i}.pan.npy",
                            rng.normal(size=(n_frames, 2)).astype(np.float32))

        ds = FencingDataset(tmp)
        assert len(ds) == 12, f"pan companions must not count as samples, got {len(ds)}"
        for i in range(len(ds)):
            x, agg, yb = ds[i]
            assert x.shape == (SEQ_LEN, INPUT_SIZE), x.shape
            assert agg.shape == (N_AGG_FEATURES,), agg.shape
            assert 0 <= yb < NUM_CLASSES
        short, _, _ = ds[0]  # fake_0 is 20 frames
        assert torch.all(short[20:] == 0)
        print("test 2 ok: 12 fake clips -> (60, 132) + (4,), pan companions excluded")

        # test 2b: leading empty frames (slow mediapipe lock-on) get stripped
        laggy = np.zeros((40, 33, 4), dtype=np.float32)
        laggy[10:] = rng.normal(size=(30, 33, 4))
        np.save(tmp / "advance" / "fake_laggy.npy", laggy)
        ds2 = FencingDataset(tmp)
        idx = [p.stem for p, _ in ds2.samples].index("fake_laggy")
        x, agg, _ = ds2[idx]
        assert torch.any(x[0] != 0), "first frame should be real data, not zeros"
        assert torch.all(x[30:] == 0), "30 real frames -> the rest is padding"
        print("test 2b ok: leading zero frames get stripped, padding moves to the end")

        # test 3: training loop runs end to end and saves a checkpoint
        hist = train_action_model(tmp, tmp / "model.pth", epochs=2, batch_size=4)
        assert len(hist["train_losses"]) == 2
        assert (tmp / "model.pth").exists()
        print("test 3 ok: 2 training epochs on fake data, checkpoint saved")
        print("(the report above is on random noise, ignore the numbers)")
    finally:
        shutil.rmtree(tmp)

    print("\nall good")
