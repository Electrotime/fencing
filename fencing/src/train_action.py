"""Train the action LSTM with the default settings and save a training plot.

Run from project root:  python src/train_action.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # just save the png, don't try to open a window
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.action_model import train_action_model

KEYPOINTS_DIR = PROJECT_ROOT / "data" / "keypoints"
MODEL_PATH = PROJECT_ROOT / "models" / "action_lstm.pth"
PLOT_PATH = PROJECT_ROOT / "models" / "action_lstm_training.png"

TRAIN_COLOR = "#2a78d6"  # blue
VAL_COLOR = "#1baf7a"    # aqua
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def _tidy(ax: plt.Axes) -> None:
    """Quiet down the axes so the data is the loudest thing in the plot."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color="#e5e4e0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SOFT)


def main() -> None:
    try:
        history = train_action_model(KEYPOINTS_DIR, MODEL_PATH)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(f"can't train yet: {e}")
    epochs = range(1, len(history["train_losses"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, history["train_losses"], color=TRAIN_COLOR, linewidth=2, label="train")
    ax_loss.plot(epochs, history["val_losses"], color=VAL_COLOR, linewidth=2, label="validation")
    ax_loss.set_xlabel("epoch", color=INK_SOFT)
    ax_loss.set_ylabel("cross-entropy loss", color=INK_SOFT)
    ax_loss.set_title("Loss", color=INK)
    ax_loss.legend(frameon=False, labelcolor=INK)
    _tidy(ax_loss)

    ax_acc.plot(epochs, [a * 100 for a in history["val_accuracies"]],
                color=TRAIN_COLOR, linewidth=2)
    ax_acc.set_xlabel("epoch", color=INK_SOFT)
    ax_acc.set_ylabel("%", color=INK_SOFT)
    ax_acc.set_title("Validation accuracy", color=INK)
    ax_acc.set_ylim(0, 100)
    _tidy(ax_acc)
    
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    best = max(history["val_accuracies"])
    print(f"\nbest validation accuracy: {best:.1%}")
    print(f"model -> {MODEL_PATH}")
    print(f"plot  -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
