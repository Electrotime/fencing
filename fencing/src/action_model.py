"""Phase 4: LSTM that watches a keypoint sequence and names the fencing action."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

CLASS_NAMES = ["advance", "lunge", "parry", "retreat"]
SEQ_LEN = 60       # every clip gets padded/trimmed to this many frames
INPUT_SIZE = 132   # 33 landmarks x 4 values, flattened per frame
HIDDEN_SIZE = 128
NUM_CLASSES = 4
DROPOUT = 0.3
MIN_RECALL = 0.70  # below this, a class needs more training clips


def _pick_device() -> torch.device:
    """cuda on the gaming pc, mps on the mac, cpu otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class FencingDataset(Dataset):
    """All the .npy keypoint clips under keypoints_dir/<action>/, one sample per clip.

    Every sample comes out as (SEQ_LEN, INPUT_SIZE). Short clips get zero-padded
    at the end. Long clips keep their first SEQ_LEN frames, since the start of an
    action (the launch of a lunge, the first step of an advance) is the part that
    actually identifies it.
    """

    def __init__(self, keypoints_dir: str | Path) -> None:
        keypoints_dir = Path(keypoints_dir)
        self.samples: list[tuple[Path, int]] = []
        for label, action in enumerate(CLASS_NAMES):
            for npy in sorted((keypoints_dir / action).glob("*.npy")):
                self.samples.append((npy, label))
        if not self.samples:
            raise FileNotFoundError(
                f"no .npy files under {keypoints_dir} - run scripts/process_clips.py first")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        kp = np.load(path).astype(np.float32)   # (n_frames, 33, 4)
        kp = kp.reshape(len(kp), -1)            # (n_frames, 132)
        # mediapipe can take a while to find the fencer at the start of a clip,
        # which leaves all-zero frames up front. drop those so zeros only ever
        # appear as padding at the end, never as fake "action" at the start
        real = np.any(kp != 0, axis=1)
        if real.any():
            kp = kp[np.argmax(real):]
        if len(kp) >= SEQ_LEN:
            kp = kp[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - len(kp), INPUT_SIZE), dtype=np.float32)
            kp = np.concatenate([kp, pad])
        return torch.from_numpy(kp), label


class ActionLSTM(nn.Module):
    """2-layer LSTM over the sequence, then a small MLP on the final hidden state.

    (batch, SEQ_LEN, INPUT_SIZE) -> (batch, NUM_CLASSES) raw logits.
    No softmax here, CrossEntropyLoss wants the logits as-is.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, num_layers=2,
                            batch_first=True, dropout=DROPOUT)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return self.head(h_n[-1])  # final hidden state of the top layer


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
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict = {"train_losses": [], "val_losses": [], "val_accuracies": []}
    best_acc = -1.0

    print(f"{len(dataset)} clips ({n_train} train / {n_val} val), device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device) 
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        train_loss = running / n_train
        
        model.eval() 
        val_loss = 0.0  
        
        correct = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                val_loss += loss_fn(logits, y).item() * len(x)
                
                correct += int((logits.argmax(1) == y).sum())
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
        for x, y in val_dl:
            logits = model(x.to(device))
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_true.extend(y.tolist())

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
    fake_batch = torch.randn(4, SEQ_LEN, INPUT_SIZE)
    out = model(fake_batch)
    assert out.shape == (4, NUM_CLASSES), out.shape
    print("test 1 ok: model maps (4, 60, 132) -> (4, 4)")
    
    # test 2: dataset padding/trimming with synthetic clips of awkward lengths             
    tmp = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(0)
        for action in CLASS_NAMES: 
            
            (tmp / action).mkdir() 
            for i, n_frames in enumerate([20, 60, 90]):
                np.save(tmp / action / f"fake_{i}.npy",
                        rng.normal(size=(n_frames, 33, 4)).astype(np.float32))
        
        ds = FencingDataset(tmp)
        assert len(ds) == 12
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.shape == (SEQ_LEN, INPUT_SIZE), x.shape
            assert 0 <= y < NUM_CLASSES 
        # short clip: the padded tail must be zeros 
        short, _ = ds[0]  # fake_0 is 20 frames
        
        
        assert torch.all(short[20:] == 0)
        print("test 2 ok: 12 fake clips all come out (60, 132), padding is zeros")

        # test 2b: leading empty frames (slow mediapipe lock-on) get stripped
        laggy = np.zeros((40, 33, 4), dtype=np.float32)
        laggy[10:] = rng.normal(size=(30, 33, 4))
        np.save(tmp / "advance" / "fake_laggy.npy", laggy)
        ds2 = FencingDataset(tmp)
        x, _ = ds2[[p.stem for p, _ in ds2.samples].index("fake_laggy")]
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
