"""Blade detection with the fine-tuned YOLO model, plus point position/velocity helpers."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

MIN_BLADE_CONFIDENCE = 0.3
NULL_VELOCITY = (0.0, 0.0)


def load_blade_model(weights_path: str | Path) -> YOLO:
    """Load the trained blade weights."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    return YOLO(str(weights_path))


def get_blade_box(frame: np.ndarray, model: YOLO) -> tuple[float, float, float, float] | None:
    """Most confident blade box as (x1, y1, x2, y2), or None if no blade found."""
    results = model(frame, conf=MIN_BLADE_CONFIDENCE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    best_idx = int(boxes.conf.argmax())
    x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy()
    return (float(x1), float(y1), float(x2), float(y2))


def get_blade_boxes(frame: np.ndarray, model: YOLO, k: int = 2) -> list:
    """Up to k most confident blade boxes -- usually both fencers' blades."""
    results = model(frame, conf=MIN_BLADE_CONFIDENCE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    conf = boxes.conf.cpu().numpy()
    order = np.argsort(-conf)[:k]
    xy = boxes.xyxy.cpu().numpy()
    return [tuple(float(v) for v in xy[i]) for i in order]


def get_blade_centre(frame: np.ndarray, model: YOLO) -> tuple[float, float] | None:
    """Centre of the most confident blade box -- roughly the blade's MIDPOINT, not its tip."""
    box = get_blade_box(frame, model)
    if box is None:
        return None
    x1, y1, x2, y2 = box
    return (float((x1 + x2) / 2), float((y1 + y2) / 2))


def blade_tip_from_box(box, wrists) -> tuple[float, float] | None:
    """Tip estimate: the box corner farthest from the nearest wrist holding the blade.

    A blade lies along one diagonal of its box, so its two ends are opposite corners;
    the far one from the hand is the point.
    """
    if box is None or not wrists:
        return None
    x1, y1, x2, y2 = box
    corners = ((x1, y1), (x2, y2), (x1, y2), (x2, y1))
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    wx, wy = min(wrists, key=lambda w: (w[0] - cx) ** 2 + (w[1] - cy) ** 2)
    return max(corners, key=lambda c: (c[0] - wx) ** 2 + (c[1] - wy) ** 2)


def blade_tip_directed(box, wrist, direction) -> tuple[float, float] | None:
    """Box corner farthest along the arm's pointing direction -- stays on the blade axis."""
    if box is None or wrist is None or direction is None:
        return None
    x1, y1, x2, y2 = box
    ux, uy = direction
    wx, wy = wrist
    corners = ((x1, y1), (x2, y2), (x1, y2), (x2, y1))
    return max(corners, key=lambda c: (c[0] - wx) * ux + (c[1] - wy) * uy)


def blade_axis(frame, box, pad: int = 6):
    """Fit the blade's line inside its box by PCA on edge pixels: (end1, end2, linearity)."""
    if box is None:
        return None
    H, W = frame.shape[:2]
    x1 = max(0, int(box[0]) - pad); y1 = max(0, int(box[1]) - pad)
    x2 = min(W, int(box[2]) + pad); y2 = min(H, int(box[3]) + pad)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or min(crop.shape[:2]) < 8:
        return None
    g = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    ys, xs = np.nonzero(cv2.Canny(g, 50, 150))
    if len(xs) < 12:
        return None
    P = np.stack([xs, ys], 1).astype(np.float32)
    mu = P.mean(0)
    _, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
    d = Vt[0]
    t = (P - mu) @ d
    e1 = mu + d * float(t.min()); e2 = mu + d * float(t.max())
    return ((float(e1[0] + x1), float(e1[1] + y1)),
            (float(e2[0] + x1), float(e2[1] + y1)),
            float(S[0] / (S[1] + 1e-6)))


def blade_tip_linefit(frame, box, wrist, min_linearity: float = 1.5):
    """Far end of the fitted blade line from the hand; None when the fit is not line-like."""
    got = blade_axis(frame, box)
    if got is None or wrist is None or got[2] < min_linearity:
        return None
    e1, e2, _ = got
    wx, wy = wrist
    far = max((e1, e2), key=lambda e: (e[0] - wx) ** 2 + (e[1] - wy) ** 2)
    return far


def get_blade_tip(frame: np.ndarray, model: YOLO, wrists=None) -> tuple[float, float] | None:
    """Blade point. Without wrists there is nothing to orient by, so the centre is returned."""
    box = get_blade_box(frame, model)
    if box is None:
        return None
    if not wrists:
        x1, y1, x2, y2 = box
        return (float((x1 + x2) / 2), float((y1 + y2) / 2))
    return blade_tip_from_box(box, wrists)


def get_blade_centre_trajectory(
    video_path: str | Path,
    model: YOLO,
) -> list[tuple[float, float] | None]: 
    """Blade CENTRE for every frame of a video (None where nothing was found)."""
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
            trajectory.append(get_blade_centre(frame, model))
            bar.update(1)

    cap.release()
    return trajectory


def compute_tip_velocity(
    trajectory: list[tuple[float, float] | None],
) -> list[tuple[float, float]]:
    """How far the tip moved between frames. Frames with no detection get (0, 0)."""
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


def _self_test_tip():
    """The tip is the far diagonal end from the hand, and degrades to None safely."""
    box = (100.0, 100.0, 200.0, 200.0)
    assert blade_tip_from_box(box, [(95.0, 205.0)]) == (200.0, 100.0)
    assert blade_tip_from_box(box, [(205.0, 95.0)]) == (100.0, 200.0)
    assert blade_tip_from_box(box, []) is None
    assert blade_tip_from_box(None, [(1.0, 1.0)]) is None
    near = blade_tip_from_box(box, [(95.0, 205.0), (1000.0, 1000.0)])
    assert near == (200.0, 100.0), "must orient by the NEAREST wrist"
    print("blade tip self-test ok")


if __name__ == "__main__":
    WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "blade_yolo" / \
              "fencing_blade_v2" / "weights" / "best.pt"

    traj: list[tuple[float, float] | None] = [(10.0, 20.0), (12.0, 22.0), None, (15.0, 25.0)]
    vel = compute_tip_velocity(traj)
    assert len(vel) == 4
    assert vel[0] == (0.0, 0.0)
    assert vel[1] == (2.0, 2.0)
    assert vel[2] == (0.0, 0.0)
    assert vel[3] == (0.0, 0.0)
    print("test 1 ok: velocity math checks out")

    if not WEIGHTS.exists():
        print(f"\nno weights at {WEIGHTS}, skipping the model tests")
    else:
        model = load_blade_model(WEIGHTS)

        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        tip = get_blade_tip(blank, model)
        assert tip is None, f"expected None on a black frame, got {tip}"
        print("test 2 ok: nothing detected on a black frame (good)")
        print("test 3: run get_blade_tip on a real frame and eyeball the result")

    print("\ndone")
