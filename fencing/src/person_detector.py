"""Find the two fencers in a frame with stock YOLOv8n and crop them out."""

from __future__ import annotations

import os

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.4

FOREGROUND_MAX_REL_MEAN = 0.55  # this dark relative to the brightest box = silhouette
FOREGROUND_MIN_BOTTOM = 0.90    # box runs to the frame bottom, i.e. in front of the piste
FILTER_ENABLED = os.environ.get("FENCING_NO_SILHOUETTE_FILTER", "") != "1"


def load_person_model() -> YOLO:
    """YOLOv8n with the stock COCO weights (auto-downloads ~6 MB the first time)."""
    return YOLO("yolov8n.pt")


def get_fencer_boxes(
    frame: np.ndarray,
    model: YOLO,
    min_h_frac: float = 0.0,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Find both fencers, return their xyxy boxes as (box_A, box_B). A = left, B = right."""
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

    if len(xyxy) > 2 and FILTER_ENABLED:
        means = np.full(len(xyxy), np.nan)
        for i, b in enumerate(xyxy):
            x1, y1, x2, y2 = (int(v) for v in b)
            patch = frame[max(0, y1):min(frame.shape[0], y2),
                          max(0, x1):min(frame.shape[1], x2)]
            if patch.size:
                means[i] = patch.mean()
        brightest = np.nanmax(means) if np.isfinite(means).any() else 0.0
        bottom = xyxy[:, 3] / frame.shape[0]
        with np.errstate(invalid="ignore"):
            drop = ((means < FOREGROUND_MAX_REL_MEAN * brightest)
                    & (bottom > FOREGROUND_MIN_BOTTOM))
        keep = ~np.nan_to_num(drop, nan=False).astype(bool)
        # never let the filter take us below two candidates -- a wrong drop costs
        # a whole fencer, which is worse than keeping a silhouette
        if keep.sum() >= 2:
            xyxy, confs = xyxy[keep], confs[keep]

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
    """Crop out both fencers. A is the fencer on the left, B is on the right."""
    box_a, box_b = get_fencer_boxes(frame, model)
    return crop_box(frame, box_a), crop_box(frame, box_b)


def _self_test_foreground() -> None:
    """The silhouette filter needs BOTH cues. Synthetic, so it needs no video."""
    class _Arr:
        def __init__(self, v): self.v = v
        def cpu(self): return self
        def numpy(self): return self.v

    class _Boxes:
        def __init__(self, x, c):
            self.xyxy, self.conf = _Arr(np.array(x, np.float32)), _Arr(np.array(c, np.float32))
        def __len__(self): return len(self.xyxy.v)

    class _Res:
        def __init__(self, x, c): self.boxes = _Boxes(x, c)

    def _stub(x, c):
        return lambda *a, **k: [_Res(x, c)]

    H, W = 1080, 1920
    fencers = [[300, 430, 500, 730], [1300, 430, 1500, 730]]
    intruder = [100, 730, 400, 1080]          # bottom-anchored, in front of the piste

    def frame_with(bottom_val, fencer_val=235):
        f = np.full((H, W, 3), 120, np.uint8)
        for x in (400, 1400):
            f[430:730, x - 100:x + 100] = fencer_val
        f[730:1080, 100:400] = bottom_val
        return f

    def centres(frame, boxes, confs):
        a, b = get_fencer_boxes(frame, _stub(boxes, confs), min_h_frac=0.25)
        return tuple(None if v is None else round(float((v[0] + v[2]) / 2 / W), 3)
                     for v in (a, b))

    trio = [fencers[0], intruder, fencers[1]]
    cf = [0.87, 0.85, 0.65]      # intruder OUT-SCORES a fencer, as measured on bout 1

    assert centres(frame_with(20), trio, cf) == (0.208, 0.729), "dark silhouette kept"
    # bright box at the bottom is bout 2's real fencer -- must survive
    assert centres(frame_with(200), trio, cf) == (0.13, 0.208), "bright bottom box dropped"
    # dim venue: an absolute brightness cutoff would take the fencers too
    assert centres(frame_with(18, fencer_val=70), trio, cf) == (0.208, 0.729), \
        "relative brightness failed in a dim venue"
    print("self-test ok: silhouettes dropped, bottom-edge fencers kept")


if __name__ == "__main__":
    _self_test_foreground()

    print("loading YOLOv8n (downloads the first time)...")
    model = load_person_model()
    print("loaded")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    crop_a, crop_b = get_fencer_crops(blank, model)
    assert crop_a is None and crop_b is None
    print("smoke test ok: black frame gives (None, None)")
