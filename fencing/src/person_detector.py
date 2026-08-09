"""Find the two fencers in a frame with stock YOLOv8n and crop them out."""

from __future__ import annotations

import os

import numpy as np
from ultralytics import YOLO

PERSON_CLASS_ID = 0
MIN_CONFIDENCE = 0.4

# Foreground silhouettes -- a spectator's head and shoulders along the bottom of
# frame, and the referee -- stand between the camera and the piste. Being nearer
# the camera they are TALLER than the fencers (measured 0.32-0.41 of frame height
# against the fencers' 0.26-0.27), so min_h_frac cannot reach them, and being
# still and upright they out-score a lunging fencer on confidence.
#
# Both thresholds are calibrated, not tuned: over 1262 tall detections across both
# bouts, boxes running to the frame bottom have median brightness 51 in bout 1
# (against 101 for the rest) but 106 in bout 2, whose tighter framing puts real
# fencers' feet near the bottom edge. So NEITHER cue works alone -- a bottom-edge
# filter would delete bout 2's fencers. Together they flag 16.2% of bout 1's
# detections and 4.1% of bout 2's, which is the expected split: bout 1 is the wide
# shot with people in the foreground. Fencers wear WHITE, which is what keeps them
# clear of the brightness bound.
#
# Brightness is judged RELATIVE to the brightest box in the same frame, not
# against a fixed 60. An absolute cutoff also deletes a real fencer standing in a
# dark patch near the bottom edge, and lighting varies between venues and across a
# single bout. Within one frame the comparison is what matters: a white uniform is
# far brighter than a silhouette, whatever the exposure.
#
# Set FENCING_NO_SILHOUETTE_FILTER=1 to disable, for A/B measurement only. The
# filter's original -3 pt cost was measured while _assign_boxes was still swapping
# A/B slots, which corrupted direction on the same frames, so that number needs
# re-taking against the corrected pipeline before it means anything.
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

    # More than 2 people means a referee, coach or spectator got picked up too.
    #
    # Drop foreground silhouettes FIRST, then fall back to the old top-2-by-
    # confidence rule for whatever is left. Deliberately subtractive: three
    # attempts to replace the confidence rule outright (continuity, capped
    # continuity, widest-separation) all scored ~35% on bout 1 against its 45.9%,
    # because they changed which box lands in which SLOT. Slot A faces right and B
    # faces left, so a misassignment inverts net-forward and turns advances into
    # retreats -- measured, 63 of 122. This only removes boxes that are provably
    # not fencers and leaves the assignment path untouched.
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
    """Crop out both fencers. A is the fencer on the left, B is on the right.

    Same slot logic as get_fencer_boxes; a slot is also None if its box would
    crop down to nothing.
    """
    box_a, box_b = get_fencer_boxes(frame, model)
    return crop_box(frame, box_a), crop_box(frame, box_b)


def _self_test_foreground() -> None:
    """The silhouette filter needs BOTH cues. Synthetic, so it needs no video.

    Guards a rule that looks over-complicated and is not. Dropping the brightness
    test deletes bout 2's fencers, whose feet reach the frame bottom under tighter
    framing. Making brightness ABSOLUTE instead of relative deletes any fencer in a
    dark patch and breaks on a different venue's lighting.
    """
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
