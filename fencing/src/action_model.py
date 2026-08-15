"""Phase 4: LSTM that watches a keypoint sequence and names the fencing action."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

CLASS_NAMES = ["advance", "lunge", "parry", "retreat", "neutral", "walking"]
SEQ_LEN = 60          # every clip gets padded/trimmed to this many frames
INPUT_SIZE = 132      # 33 landmarks x 4 values, flattened per frame
N_AGG_FEATURES = 6    # engineered clip-level features fed straight into the head
N_AGG_WIDE = 2 * N_AGG_FEATURES + 1
HIDDEN_SIZE = 128     # re-tuned at 488 windows (h64 won at 83 samples; capacity pays
                      # off again with 6x the data: +1.5 pts, lunge ~100%)
NUM_CLASSES = 6
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4
MIN_RECALL = 0.70     # below this, a class needs more training clips

SLICEABLE_CLASSES = {"neutral", "walking"}
SLICE_STRIDE = 30     # 1 s hop between windows
MIN_SLICE = 20        # don't emit a window with fewer real frames than this

NOSE = 0
SHOULDER_LEFT, SHOULDER_RIGHT = 11, 12
HIP_LEFT, HIP_RIGHT = 23, 24
KNEE_LEFT, KNEE_RIGHT = 25, 26
WRIST_LEFT, WRIST_RIGHT = 15, 16
ANKLE_LEFT, ANKLE_RIGHT = 27, 28


def _knee_angle(hip: np.ndarray, knee: np.ndarray, ankle: np.ndarray) -> np.ndarray:
    """Per-frame knee angle in degrees for one leg (hip-knee-ankle)."""
    a = hip - knee
    b = ankle - knee
    cos = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def _first_mover(kp: np.ndarray, nose_dir: float) -> float:
    """NOT WIRED IN -- kept as a documented dead end. See the warning at the end."""
    if len(kp) < 6:
        return 0.0
    ax = kp[:, ANKLE_LEFT, 0] * nose_dir
    bx = kp[:, ANKLE_RIGHT, 0] * nose_dir
    front, back = (ax, bx) if np.mean(ax) >= np.mean(bx) else (bx, ax)
    vf, vb = np.diff(front), np.diff(back)
    both = np.abs(np.concatenate([vf, vb]))
    thresh = max(1e-4, 0.5 * float(np.percentile(both, 75)))
    moved_f, moved_b = np.abs(vf) > thresh, np.abs(vb) > thresh
    fi = int(np.argmax(moved_f)) if moved_f.any() else len(vf)
    bi = int(np.argmax(moved_b)) if moved_b.any() else len(vb)
    if fi == bi:
        return 0.0
    return 1.0 if fi < bi else -1.0


def _pick_device() -> torch.device:
    """cuda on the gaming pc, mps on the mac, cpu otherwise."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


PAN_WIDTH = 320.0     # px width the pan was measured at (see pose_pipeline.PAN_DOWNSCALE)
FORWARD_SCALE = 10.0  # lifts fraction-of-width motion into a ~[-3, 3] feature range


def wide_agg(own: np.ndarray, opponent: np.ndarray | None) -> np.ndarray:
    """[own(6) | opponent(6) | present(1)] -- the ONLY definition of this layout."""
    own = np.asarray(own, dtype=np.float32).reshape(-1)
    if opponent is None:
        return np.concatenate([own, np.zeros(N_AGG_FEATURES, np.float32),
                               np.zeros(1, np.float32)]).astype(np.float32)
    opp = np.asarray(opponent, dtype=np.float32).reshape(-1)
    return np.concatenate([own, opp, np.ones(1, np.float32)]).astype(np.float32)


def _engineered_features(kp: np.ndarray, motion: np.ndarray) -> np.ndarray:
    """Seven clip-level numbers that decide the classes the raw sequence can't."""
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
    # wrist forward of the shoulder along the facing direction, best of both arms
    reach_l = (kp[:, WRIST_LEFT, 0] - kp[:, SHOULDER_LEFT, 0]) * nose_dir
    reach_r = (kp[:, WRIST_RIGHT, 0] - kp[:, SHOULDER_RIGHT, 0]) * nose_dir
    arm_reach = np.percentile(np.maximum(reach_l, reach_r), 90)

    # crouch from the more-bent knee: straight ~180 deg -> 0, deep bend ~120 -> 1
    knee = np.minimum(
        _knee_angle(kp[:, HIP_LEFT, :2], kp[:, KNEE_LEFT, :2], kp[:, ANKLE_LEFT, :2]),
        _knee_angle(kp[:, HIP_RIGHT, :2], kp[:, KNEE_RIGHT, :2], kp[:, ANKLE_RIGHT, :2]),
    )
    crouch = (180.0 - float(np.median(knee))) / 60.0

    return np.array([
        float(np.clip(forward.sum(), -3.0, 3.0)),
        float(np.percentile(stance, 90)),
        float(np.clip(np.percentile(wrist_step, 90), 0, 3)),
        float(np.clip(np.abs(forward).sum(), 0, 6.0)),
        float(np.clip(arm_reach, -1.5, 2.5)),
        float(np.clip(crouch, 0.0, 1.5)),
    ], dtype=np.float32)


def _load_clip(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a clip's keypoints + aligned motion track, leading lock-on zeros stripped."""
    kp = np.load(path).astype(np.float32)            # (n, 33, 4)
    motion_path = path.with_name(path.stem + ".pan.npy")  # (n, 2): [pan, hip-x]
    motion = (np.load(motion_path).astype(np.float32)
              if motion_path.exists() else np.zeros((len(kp), 2), dtype=np.float32))
    m = min(len(kp), len(motion))
    kp, motion = kp[:m], motion[:m]
    real = np.any(kp.reshape(len(kp), -1) != 0, axis=1)
    if real.any():
        s = int(np.argmax(real))
        kp, motion = kp[s:], motion[s:]              # strip leading all-zero frames
    return kp, motion


class FencingDataset(Dataset):
    """Keypoint clips under keypoints_dir/<action>/ as (sequence, features, label)."""

    def __init__(self, keypoints_dir: str | Path) -> None:
        keypoints_dir = Path(keypoints_dir)
        # each sample: (path, label, start) -- start None = whole clip / first window
        self.samples: list[tuple[Path, int, int | None]] = []
        self.groups: list[int] = []
        self.labels: list[int] = []
        group = 0
        for label, action in enumerate(CLASS_NAMES):
            for npy in sorted((keypoints_dir / action).glob("*.npy")):
                if npy.name.endswith(".pan.npy"):
                    continue
                if action in SLICEABLE_CLASSES:
                    kp, _ = _load_clip(npy)
                    starts = list(range(0, max(1, len(kp) - MIN_SLICE + 1), SLICE_STRIDE)) or [0]
                    for st in starts:
                        self.samples.append((npy, label, st))
                        self.groups.append(group)
                        self.labels.append(label)
                else:
                    self.samples.append((npy, label, None))
                    self.groups.append(group)
                    self.labels.append(label)
                group += 1
        if not self.samples:
            raise FileNotFoundError(
                f"no .npy files under {keypoints_dir} - run scripts/process_clips.py first")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        path, label, start = self.samples[idx]
        kp, motion = _load_clip(path)
        if start is not None:                         # sliced window
            kp, motion = kp[start:start + SEQ_LEN], motion[start:start + SEQ_LEN]

        agg = _engineered_features(kp, motion)

        flat = kp.reshape(len(kp), -1)                # (n_frames, 132)
        n_real = min(len(flat), SEQ_LEN)              # frames before the padding starts
        if len(flat) >= SEQ_LEN:
            flat = flat[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - len(flat), INPUT_SIZE), dtype=np.float32)
            flat = np.concatenate([flat, pad])
        return (torch.from_numpy(flat), torch.from_numpy(agg),
                torch.tensor(n_real, dtype=torch.long), label)


def group_stratified_split(labels: list[int], groups: list[int], val_per_class: int,
                           seed: int) -> tuple[list[int], list[int]]:
    """Hold out val_per_class whole CLIPS per class; return (train_idx, val_idx)."""
    labels_arr = np.array(labels)
    groups_arr = np.array(groups)
    rng = np.random.default_rng(seed)
    val_groups: set[int] = set()
    for c in range(NUM_CLASSES):
        cls_groups = np.unique(groups_arr[labels_arr == c])
        rng.shuffle(cls_groups)
        val_groups.update(cls_groups[:val_per_class].tolist())
    val_idx = [i for i, g in enumerate(groups) if g in val_groups]
    train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
    return train_idx, val_idx


class ActionLSTM(nn.Module):
    """LSTM over the keypoint sequence, engineered clip stats joined at the head."""

    POOL_MODES = ("mean", "max", "last")

    def __init__(self, pool: str = "mean", n_agg: int = N_AGG_FEATURES) -> None:
        """`pool` selects the time reduction. DEFAULTS TO "mean" for a reason:"""
        super().__init__()
        if pool not in self.POOL_MODES:
            raise ValueError(f"pool must be one of {self.POOL_MODES}, got {pool!r}")
        self.pool = pool
        self.n_agg = n_agg
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + n_agg, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor, agg: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)
        if lengths is None:
            # caller fed a full window with no length; treat every frame as real
            lengths = torch.full((x.shape[0],), x.shape[1], device=x.device)
        steps = torch.arange(x.shape[1], device=x.device)[None, :]
        mask = (steps < lengths[:, None].to(x.device)).unsqueeze(-1).to(out.dtype)
        if self.pool == "mean":
            pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        elif self.pool == "max":
            # -inf on padding: zeros would otherwise beat negative activations
            pooled = out.masked_fill(mask == 0, float("-inf")).max(dim=1).values
        else:  # last real timestep
            idx = (lengths.to(out.device) - 1).clamp(min=0)
            pooled = out[torch.arange(out.shape[0], device=out.device), idx]
        return self.head(torch.cat([pooled, agg], dim=-1))


class ActionFrameLSTM(nn.Module):
    """Per-FRAME classifier: (batch, T, INPUT_SIZE) + agg -> (batch, T, NUM_CLASSES)."""

    def __init__(self) -> None:
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(HIDDEN_SIZE + N_AGG_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor, agg: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.lstm(x)                              # (B, T, HIDDEN)
        agg_t = agg[:, None, :].expand(-1, out.shape[1], -1)
        return self.head(torch.cat([out, agg_t], dim=-1))  # (B, T, NUM_CLASSES)


def frame_logits_to_window(logits: torch.Tensor, lengths: torch.Tensor | None = None,
                           mode: str = "last") -> torch.Tensor:
    """(B, T, C) per-frame logits -> (B, C) one call per window."""
    b, t, c = logits.shape
    if lengths is None:
        lengths = torch.full((b,), t, dtype=torch.long, device=logits.device)
    idx = (lengths.to(logits.device).clamp(1, t) - 1)
    if mode == "last":
        return logits[torch.arange(b, device=logits.device), idx]
    steps = torch.arange(t, device=logits.device)[None, :]
    mask = steps < lengths.to(logits.device)[:, None]
    pred = logits.argmax(-1)
    out = torch.zeros(b, c, device=logits.device)
    for i in range(b):
        counts = torch.bincount(pred[i][mask[i]], minlength=c).float()
        out[i] = counts
    return out


class ActionEnsemble(nn.Module):
    """Several ActionLSTMs averaged in probability space. Same call signature."""

    def __init__(self, members: list["ActionLSTM"]) -> None:
        super().__init__()
        if not members:
            raise ValueError("ActionEnsemble needs at least one member")
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor, agg: torch.Tensor,
                lengths: torch.Tensor | None = None) -> torch.Tensor:
        probs = torch.stack([torch.softmax(m(x, agg, lengths), dim=-1)
                             for m in self.members]).mean(0)
        return torch.log(probs.clamp_min(1e-12))


def load_action_model(weights_path: str | Path,
                      device: torch.device | None = None,
                      cls: type[nn.Module] = ActionLSTM) -> nn.Module:
    """Load the ensemble if members sit beside the checkpoint, else the single one."""
    weights_path = Path(weights_path)
    device = device or _pick_device()
    members = sorted(weights_path.parent.glob(f"{weights_path.stem}.m*.pth"))

    def _one(p: Path) -> nn.Module:
        m = cls()
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        return m

    if len(members) >= 2:
        model = ActionEnsemble([_one(p) for p in members]).to(device)
        model.eval()
        print(f"loaded {len(members)}-model ensemble from {weights_path.parent}")
        return model
    model = _one(weights_path).to(device)
    model.eval()
    return model


def train_action_model(
    keypoints_dir: str | Path,
    save_path: str | Path,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
    quiet: bool = False,
) -> dict:
    """Full training loop. Keeps whichever epoch scored best on validation."""
    torch.manual_seed(seed)
    device = _pick_device()
    dataset = FencingDataset(keypoints_dir)

    # hold out ~val_split of each class's CLIPS (not windows), stratified
    group_counts = [len(np.unique(np.array(dataset.groups)[np.array(dataset.labels) == c]))
                    for c in range(NUM_CLASSES)]
    present = [g for g in group_counts if g > 0]
    if not present:
        raise ValueError("no samples found")
    val_per_class = max(1, int(round(val_split * min(present))))
    train_idx, val_idx = group_stratified_split(dataset.labels, dataset.groups, val_per_class, seed)
    if not train_idx or not val_idx:
        raise ValueError(f"not enough clips to train and validate (per-class clips: {group_counts})")
    n_train, n_val = len(train_idx), len(val_idx)
    train_dl = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    model = ActionLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    # inverse-frequency class weights: slicing turns walking/neutral into many more
    # windows than the transient classes, so weight the loss back toward balance
    tr_labels = np.array([dataset.labels[i] for i in train_idx])
    freq = np.array([max(1, int(np.sum(tr_labels == c))) for c in range(NUM_CLASSES)])
    class_w = torch.tensor(freq.sum() / (NUM_CLASSES * freq), dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict = {"train_losses": [], "val_losses": [], "val_accuracies": []}
    best_acc = -1.0

    print(f"{len(dataset)} clips ({n_train} train / {n_val} val), device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, agg, ln, yb in train_dl:
            x, agg, yb = x.to(device), agg.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x, agg, ln), yb)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        train_loss = running / n_train

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for x, agg, ln, yb in val_dl:
                x, agg, yb = x.to(device), agg.to(device), yb.to(device)
                logits = model(x, agg, ln)
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
        if not quiet or note or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs}  train loss {train_loss:.4f}  "
                  f"val loss {val_loss:.4f}  val acc {val_acc:.1%}{note}")

    if not quiet:
        _print_val_report(save_path, val_dl, device)
    return history


def train_frame_action_model(
    keypoints_dir: str | Path,
    save_path: str | Path,
    epochs: int = 80,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    seed: int = 42,
    quiet: bool = False,
) -> dict:
    """Train the per-frame ActionFrameLSTM. Same data, same splits, per-frame loss."""
    torch.manual_seed(seed)
    device = _pick_device()
    dataset = FencingDataset(keypoints_dir)

    group_counts = [len(np.unique(np.array(dataset.groups)[np.array(dataset.labels) == c]))
                    for c in range(NUM_CLASSES)]
    present = [g for g in group_counts if g > 0]
    if not present:
        raise ValueError("no samples found")
    val_per_class = max(1, int(round(val_split * min(present))))
    train_idx, val_idx = group_stratified_split(dataset.labels, dataset.groups,
                                                val_per_class, seed)
    if not train_idx or not val_idx:
        raise ValueError(f"not enough clips to train and validate (per-class: {group_counts})")
    train_dl = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(Subset(dataset, val_idx), batch_size=batch_size)

    model = ActionFrameLSTM().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)

    frame_freq = np.zeros(NUM_CLASSES, dtype=np.float64)
    for i in train_idx:
        frame_freq[dataset.labels[i]] += float(dataset[i][2])
    frame_freq = np.maximum(frame_freq, 1.0)
    class_w = torch.tensor(frame_freq.sum() / (NUM_CLASSES * frame_freq),
                           dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=class_w, reduction="none")

    def _targets(ln: torch.Tensor, yb: torch.Tensor, t: int):
        y = yb[:, None].expand(-1, t).to(device)
        steps = torch.arange(t, device=device)[None, :]
        return y, (steps < ln.to(device)[:, None]).float()

    def _evaluate():
        model.eval()
        frame_ok = frame_n = win_ok = win_n = 0
        with torch.no_grad():
            for x, agg, ln, yb in val_dl:
                x, agg, yb = x.to(device), agg.to(device), yb.to(device)
                logits = model(x, agg)
                y, mask = _targets(ln, yb, x.shape[1])
                pred = logits.argmax(-1)
                frame_ok += float(((pred == y).float() * mask).sum())
                frame_n += float(mask.sum())
                vote = frame_logits_to_window(logits, ln, mode="vote").argmax(-1)
                win_ok += int((vote == yb).sum())
                win_n += len(yb)
        return frame_ok / max(frame_n, 1.0), win_ok / max(win_n, 1)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    history: dict = {"train_losses": [], "val_accuracies": [], "frame_accuracies": []}
    best_acc = -1.0

    print(f"{len(dataset)} windows ({len(train_idx)} train / {len(val_idx)} val), "
          f"per-frame, device: {device}")
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for x, agg, ln, yb in train_dl:
            x, agg, yb = x.to(device), agg.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(x, agg)
            y, mask = _targets(ln, yb, x.shape[1])
            per = loss_fn(logits.reshape(-1, NUM_CLASSES), y.reshape(-1)).reshape(y.shape)
            loss = (per * mask).sum() / mask.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            running += loss.item() * len(x)
        train_loss = running / len(train_idx)

        frame_acc, val_acc = _evaluate()
        history["train_losses"].append(train_loss)
        history["val_accuracies"].append(val_acc)
        history["frame_accuracies"].append(frame_acc)

        note = ""
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            note = "  <- best so far, saved"
        if not quiet or note or epoch % 10 == 0 or epoch == epochs:
            print(f"epoch {epoch:3d}/{epochs}  loss {train_loss:.4f}  "
                  f"frame acc {frame_acc:.1%}  window acc {val_acc:.1%}{note}")
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
        for x, agg, ln, yb in val_dl:
            logits = model(x.to(device), agg.to(device), ln)
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
    print(f"test 1 ok: model maps (4, {SEQ_LEN}, {INPUT_SIZE}) + (4, {N_AGG_FEATURES}) "
          f"-> (4, {NUM_CLASSES})")

    # test 2: dataset shapes; sliceable classes yield multiple windows per clip
    tmp = Path(tempfile.mkdtemp())
    try:
        rng = np.random.default_rng(0)
        for action in CLASS_NAMES:
            (tmp / action).mkdir()
            for i, n_frames in enumerate([20, 60, 90]):
                np.save(tmp / action / f"fake_{i}.npy",
                        rng.normal(size=(n_frames, 33, 4)).astype(np.float32))
                if i == 0:  # one clip gets a motion track, others test the missing path
                    np.save(tmp / action / f"fake_{i}.pan.npy",
                            rng.normal(size=(n_frames, 2)).astype(np.float32))

        ds = FencingDataset(tmp)
        for i in range(len(ds)):
            x, agg, ln, yb = ds[i]
            assert x.shape == (SEQ_LEN, INPUT_SIZE), x.shape
            assert agg.shape == (N_AGG_FEATURES,), agg.shape
            assert 1 <= int(ln) <= SEQ_LEN, ln
            assert 0 <= yb < NUM_CLASSES
        adv, wlk = CLASS_NAMES.index("advance"), CLASS_NAMES.index("walking")
        n_adv = sum(1 for _, l, _ in ds.samples if l == adv)
        n_wlk = sum(1 for _, l, _ in ds.samples if l == wlk)
        assert n_adv == 3, f"transient class = one sample per clip, got {n_adv}"
        assert n_wlk > 3, f"sliceable class should slice into more windows, got {n_wlk}"
        wlk_groups = {g for g, (_, l, _) in zip(ds.groups, ds.samples) if l == wlk}
        assert len(wlk_groups) == 3, "3 walking clips must stay 3 groups (no leakage)"
        print(f"test 2 ok: shapes valid; walking sliced 3 clips -> {n_wlk} windows / 3 groups")

        short, _, short_len, _ = ds[0]  # advance/fake_0, 20 frames -> padded tail
        assert torch.all(short[20:] == 0)
        assert int(short_len) == 20, f"real-frame count should be 20, got {int(short_len)}"
        print("test 2b ok: short clip zero-padded at the end, length reported as 20")

        # test 2b2: pooling must ignore the padding, so scribbling over the pad
        # region cannot change the answer when the length is passed
        m = ActionLSTM()
        m.eval()
        x0, a0 = short[None].clone(), ds[0][1][None]
        scribbled = x0.clone()
        scribbled[:, 20:] = torch.randn_like(scribbled[:, 20:])
        with torch.no_grad():
            same = m(x0, a0, short_len[None])
            also = m(scribbled, a0, short_len[None])
            unmasked = m(scribbled, a0)
        assert torch.allclose(same, also, atol=1e-6), "masked pooling leaked padding"
        assert not torch.allclose(same, unmasked, atol=1e-6), "unmasked path should differ"
        print("test 2b2 ok: masked pooling ignores whatever sits in the padded tail")

        # test 2c: leading empty frames (slow mediapipe lock-on) get stripped
        laggy = np.zeros((40, 33, 4), dtype=np.float32)
        laggy[10:] = rng.normal(size=(30, 33, 4))
        np.save(tmp / "advance" / "fake_laggy.npy", laggy)
        ds2 = FencingDataset(tmp)
        idx = [p.stem for p, _, _ in ds2.samples].index("fake_laggy")
        x, agg, _, _ = ds2[idx]
        assert torch.any(x[0] != 0), "first frame should be real data, not zeros"
        assert torch.all(x[30:] == 0), "30 real frames -> the rest is padding"
        print("test 2c ok: leading zero frames stripped, padding moves to the end")

        # test 2d: group-stratified split holds out whole clips, no window leakage
        tr, va = group_stratified_split(ds.labels, ds.groups, val_per_class=1, seed=0)
        tr_groups = {ds.groups[i] for i in tr}
        va_groups = {ds.groups[i] for i in va}
        assert tr_groups.isdisjoint(va_groups), "a clip's windows must not span the split"
        print("test 2d ok: group split keeps each clip entirely in train or val")

        # test 3: training loop runs end to end and saves a checkpoint
        hist = train_action_model(tmp, tmp / "model.pth", epochs=2, batch_size=4)
        assert len(hist["train_losses"]) == 2
        assert (tmp / "model.pth").exists()
        print("test 3 ok: 2 training epochs on fake data, checkpoint saved")
        print("(the report above is on random noise, ignore the numbers)")
    finally:
        shutil.rmtree(tmp)

    print("\nall good")
