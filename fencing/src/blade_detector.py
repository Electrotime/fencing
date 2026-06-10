"""Phase 3 — blade detection and tip extraction using a fine-tuned YOLO model."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

MIN_BLADE_CONFIDENCE = 0.3
NULL_VELOCITY = (0.0, 0.0)
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def load_blade_model(weights_path: str | Path) -> YOLO:
    """Load YOLO weights from disk."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return YOLO(str(weights_path))


def get_blade_tip(frame: np.ndarray, model: YOLO) -> tuple[float, float] | None:
    """Return centroid of best blade detection as (x, y), or None if nothing found."""
    results = model(frame, conf=MIN_BLADE_CONFIDENCE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def get_blade_tip_trajectory(
    video_path: str | Path,
    model: YOLO,
) -> list[tuple[float, float] | None]:
    """Run blade detection on every frame of a video, return per-frame tip positions."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    trajectory: list[tuple[float, float] | None] = []

    with tqdm(total=total, desc=video_path.stem[:40], unit="frame") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            trajectory.append(get_blade_tip(frame, model))
            bar.update(1)

    cap.release()
    return trajectory


def compute_tip_velocity(
    trajectory: list[tuple[float, float] | None],
) -> list[tuple[float, float]]:
    """Frame-to-frame blade tip displacement. None frames get (0, 0)."""
    if not trajectory:
        return []

    velocities: list[tuple[float, float]] = [NULL_VELOCITY]
    for i in range(1, len(trajectory)):
        curr, prev = trajectory[i], trajectory[i - 1]
        if curr is None or prev is None:
            velocities.append(NULL_VELOCITY)
        else:
            velocities.append((float(curr[0] - prev[0]), float(curr[1] - prev[1])))

    return velocities


if __name__ == "__main__":
    WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "blade_yolo" / \
              "fencing_blade_v1" / "weights" / "best.pt"

    traj: list[tuple[float, float] | None] = [(10.0, 20.0), (12.0, 22.0), None, (15.0, 25.0)]
    vel = compute_tip_velocity(traj)
    assert len(vel) == 4
    assert vel[0] == (0.0, 0.0)
    assert vel[1] == (2.0, 2.0)
    assert vel[2] == (0.0, 0.0)
    assert vel[3] == (0.0, 0.0)
    print("Test 1 passed: compute_tip_velocity logic correct")

    if not WEIGHTS.exists():
        print(f"\nWeights not found at {WEIGHTS} — skipping Tests 2 & 3.")
    else:
        model = load_blade_model(WEIGHTS)

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        tip = get_blade_tip(blank, model)
        assert tip is None, f"Expected None on blank frame, got {tip}"
        print("Test 2 passed: no false detection on blank frame")
        print("Test 3: run get_blade_tip on a real frame to verify visually.")

    print("\nSmoke tests passed.")
