"""Find the two fencers in a frame with stock YOLOv8n and crop them out."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.4


def load_person_model() -> YOLO:
    """YOLOv8n with the stock COCO weights (auto-downloads ~6 MB the first time)."""
    return YOLO("yolov8n.pt")


def get_fencer_crops(
    frame: np.ndarray,
    model: YOLO,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Crop out both fencers. A is the fencer on the left, B is on the right.

    If only one person shows up, whichever half of the frame they're on decides
    the slot, and the other slot is None. A slot is also None if its box is
    useless (would crop down to nothing).
    """
    results = model(frame, classes=[PERSON_CLASS_ID], conf=MIN_CONFIDENCE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    # more than 2 people means a ref or a coach got picked up too, keep the 2 strongest
    if len(xyxy) > 2:
        top2 = np.argsort(confs)[::-1][:2]
        xyxy = xyxy[top2]

    x_centers = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    order = np.argsort(x_centers)
    xyxy = xyxy[order]
    x_centers = x_centers[order]

    h, w = frame.shape[:2]

    def _crop(box: np.ndarray) -> np.ndarray | None:
        x1, y1, x2, y2 = box.astype(int)
        crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
        return crop if crop.size > 0 else None

    if len(xyxy) == 1:
        # only found one person, so use their side of the frame to guess which
        # fencer it is instead of always calling them fencer A
        crop = _crop(xyxy[0])
        return (crop, None) if x_centers[0] < w / 2 else (None, crop)

    return _crop(xyxy[0]), _crop(xyxy[1])


if __name__ == "__main__":
    print("loading YOLOv8n (downloads the first time)...")
    model = load_person_model()
    print("loaded")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    crop_a, crop_b = get_fencer_crops(blank, model)
    assert crop_a is None and crop_b is None
    print("smoke test ok: black frame gives (None, None)")
