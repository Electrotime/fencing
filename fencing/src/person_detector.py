"""Person detection using pretrained YOLOv8n — crops both fencers out of a full frame."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.4


def load_person_model() -> YOLO:
    """Load YOLOv8n with COCO weights (downloads ~6 MB on first run)."""
    return YOLO("yolov8n.pt")


def get_fencer_crops(
    frame: np.ndarray,
    model: YOLO,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect the two fencers and return (crop_A, crop_B) sorted left-to-right."""
    results = model(frame, classes=[PERSON_CLASS_ID], conf=MIN_CONFIDENCE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    # Keep top 2 by confidence to ignore coaches/referees at frame edges
    if len(xyxy) > 2:
        top2 = np.argsort(confs)[::-1][:2]
        xyxy = xyxy[top2]

    x_centers = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    xyxy = xyxy[np.argsort(x_centers)]

    h, w = frame.shape[:2]

    def _crop(box: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = box.astype(int)
        return frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]

    crop_a = _crop(xyxy[0]) if len(xyxy) >= 1 else None
    crop_b = _crop(xyxy[1]) if len(xyxy) >= 2 else None
    return crop_a, crop_b


if __name__ == "__main__":
    print("Loading YOLOv8n (COCO) — downloads on first run...")
    model = load_person_model()
    print("Model loaded.")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    crop_a, crop_b = get_fencer_crops(blank, model)
    assert crop_a is None and crop_b is None
    print("Smoke test passed: blank frame -> (None, None)")
