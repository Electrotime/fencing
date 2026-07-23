"""Find the two fencers in a frame with stock YOLOv8n and crop them out."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.4


def load_person_model() -> YOLO:
    """YOLOv8n with the stock COCO weights (auto-downloads ~6 MB the first time)."""
    return YOLO("yolov8n.pt")


def get_fencer_boxes(
    frame: np.ndarray,
    model: YOLO,
    min_h_frac: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find both fencers, return their xyxy boxes as (box_A, box_B). A = left, B = right.

    If only one person shows up, whichever half of the frame they're on decides
    the slot, and the other slot is None. min_h_frac drops detections shorter
    than that fraction of frame height -- printed fencers on backdrop banners
    trigger real person detections, and box size is what tells them apart.
    """
    results = model(frame, classes=[PERSON_CLASS_ID], conf=MIN_CONFIDENCE, verbose=False)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None, None

    xyxy = boxes.xyxy.cpu().numpy()
    confs = boxes.conf.cpu().numpy()

    if min_h_frac > 0:
        tall = (xyxy[:, 3] - xyxy[:, 1]) >= min_h_frac * frame.shape[0]
        xyxy, confs = xyxy[tall], confs[tall]
        if len(xyxy) == 0:
            return None, None

    # more than 2 people means a ref or a coach got picked up too, keep the 2 strongest
    if len(xyxy) > 2:
        top2 = np.argsort(confs)[::-1][:2]
        xyxy = xyxy[top2]

    x_centers = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
    order = np.argsort(x_centers)
    xyxy = xyxy[order]
    x_centers = x_centers[order]

    if len(xyxy) == 1:
        # only found one person, so use their side of the frame to guess which
        # fencer it is instead of always calling them fencer A
        return (xyxy[0], None) if x_centers[0] < frame.shape[1] / 2 else (None, xyxy[0])

    return xyxy[0], xyxy[1]


def crop_box(frame: np.ndarray, box: np.ndarray | None) -> np.ndarray | None:
    """Cut a box out of the frame. None if the box is missing or degenerate."""
    if box is None:
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box.astype(int)
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    return crop if crop.size > 0 else None


def get_fencer_crops(
    frame: np.ndarray,
    model: YOLO,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Crop out both fencers. A is the fencer on the left, B is on the right.

    Same slot logic as get_fencer_boxes; a slot is also None if its box would
    crop down to nothing.
    """
    box_a, box_b = get_fencer_boxes(frame, model)
    return crop_box(frame, box_a), crop_box(frame, box_b)


if __name__ == "__main__":
    print("loading YOLOv8n (downloads the first time)...")
    model = load_person_model()
    print("loaded")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    crop_a, crop_b = get_fencer_crops(blank, model)
    assert crop_a is None and crop_b is None
    print("smoke test ok: black frame gives (None, None)")
