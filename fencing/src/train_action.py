"""Train the action LSTM across several seeds and ship the strongest checkpoint.

Each seed gets a different train/val split and weight init; the winner's weights
end up in models/action_lstm.pth so the shipped model isn't hostage to one
unlucky split. Quote the MEAN accuracy across seeds, not the winner's number --
the winner is picked on its own val set, so its score runs optimistic.

Run from project root:  python src/train_action.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # just save the png, don't try to open a window
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.action_model import train_action_model

KEYPOINTS_DIR = PROJECT_ROOT / "data" / "keypoints"
MODEL_PATH = PROJECT_ROOT / "models" / "action_lstm.pth"
PLOT_PATH = PROJECT_ROOT / "models" / "action_lstm_training.png"

N_SEEDS = 10

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
    tmp_path = MODEL_PATH.with_name("action_lstm_candidate.pth")
    best_acc, best_seed, best_history, accs = -1.0, None, None, []

    for seed in range(N_SEEDS):
        print(f"=== seed {seed} ===")
        try:
            history = train_action_model(KEYPOINTS_DIR, tmp_path, seed=seed, quiet=True)
        except (FileNotFoundError, ValueError) as e:
            sys.exit(f"can't train yet: {e}")
        acc = max(history["val_accuracies"])
        accs.append(acc)
        if acc > best_acc:
            best_acc, best_seed, best_history = acc, seed, history
            shutil.copy2(tmp_path, MODEL_PATH)
            print(f"seed {seed}: {acc:.1%}  <- new best, weights kept\n")
        else:
            print(f"seed {seed}: {acc:.1%}\n")
    tmp_path.unlink(missing_ok=True)

    epochs = range(1, len(best_history["train_losses"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(11, 4))

    ax_loss.plot(epochs, best_history["train_losses"], color=TRAIN_COLOR, linewidth=2, label="train")
    ax_loss.plot(epochs, best_history["val_losses"], color=VAL_COLOR, linewidth=2, label="validation")
    ax_loss.set_xlabel("epoch", color=INK_SOFT)
    ax_loss.set_ylabel("cross-entropy loss", color=INK_SOFT)
    ax_loss.set_title(f"Loss (best seed {best_seed})", color=INK)
    ax_loss.legend(frameon=False, labelcolor=INK)
    _tidy(ax_loss)

    ax_acc.plot(epochs, [a * 100 for a in best_history["val_accuracies"]],
                color=TRAIN_COLOR, linewidth=2)
    ax_acc.set_xlabel("epoch", color=INK_SOFT)
    ax_acc.set_ylabel("%", color=INK_SOFT)
    ax_acc.set_title("Validation accuracy", color=INK)
    ax_acc.set_ylim(0, 100)
    _tidy(ax_acc)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)

    print(f"\nseeds: {N_SEEDS}   mean val acc {np.mean(accs):.1%} +/- {np.std(accs):.1%}"
          "   (quote this, not the max)")
    print(f"shipped checkpoint: seed {best_seed} at {best_acc:.1%}")
    print(f"model -> {MODEL_PATH}")
    print(f"plot  -> {PLOT_PATH}")


if __name__ == "__main__":
    main()
