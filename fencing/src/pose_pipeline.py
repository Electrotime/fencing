"""Pose estimation with MediaPipe (the newer Tasks API, since mp.solutions is gone)."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

N_LANDMARKS = 33
MIN_POSE_DETECTION_CONFIDENCE = 0.5
MIN_POSE_PRESENCE_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
VISIBILITY_THRESHOLD = 0.5

SHOULDER_LEFT, SHOULDER_RIGHT = 11, 12
HIP_LEFT, HIP_RIGHT = 23, 24
ANKLE_LEFT, ANKLE_RIGHT = 27, 28

# background-pan estimation (recovers the footwork direction the panning camera hides)
PAN_DOWNSCALE = (320, 180)  # small grayscale copies used for phase correlation
PAN_STRIP_FRAC = 0.22       # width of the left/right background strips
PAN_MIN_RESPONSE = 0.08     # phaseCorrelate confidence gate

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_MODEL_FILENAME = "pose_landmarker_full.task"

# which landmarks connect to which, for drawing the skeleton
BODY_CONNECTIONS: list[tuple[int, int]] = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31),
    (24, 26), (26, 28), (28, 30), (28, 32),
]

_landmarker_single: mp.tasks.vision.PoseLandmarker | None = None


def download_pose_model(models_dir: Path | None = None) -> Path:
    """Grab the pose model file (~27 MB) if it isn't sitting in models/ already."""
    if models_dir is None:
        models_dir = Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out = models_dir / _MODEL_FILENAME
    if not out.exists():
        print(f"Downloading {_MODEL_FILENAME} (~27 MB), only happens once...")
        urllib.request.urlretrieve(_MODEL_URL, out)
        print(f"saved to {out}")
    return out


def _model_path() -> Path:
    return download_pose_model()


def _make_landmarker(running_mode: mp.tasks.vision.RunningMode) -> mp.tasks.vision.PoseLandmarker:
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(_model_path())),
        running_mode=running_mode,
        num_poses=1,
        min_pose_detection_confidence=MIN_POSE_DETECTION_CONFIDENCE,
        min_pose_presence_confidence=MIN_POSE_PRESENCE_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
    )
    return mp.tasks.vision.PoseLandmarker.create_from_options(options)


def _landmarks_to_array(result: mp.tasks.vision.PoseLandmarkerResult) -> np.ndarray:
    """Turn a mediapipe result into a (33, 4) array, all zeros if it found nobody."""
    if not result.pose_landmarks:
        return np.zeros((N_LANDMARKS, 4), dtype=np.float32)
    lms = result.pose_landmarks[0]
    return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in lms], dtype=np.float32)


def _normalize(kp: np.ndarray) -> np.ndarray:
    """Center the keypoints on the hips and scale by torso length, so it doesn't"""
    kp = kp.copy()
    head_center = (kp)
    hip_mid = (kp[HIP_LEFT, :2] + kp[HIP_RIGHT, :2]) / 2.0
    shoulder_mid = (kp[SHOULDER_LEFT, :2] + kp[SHOULDER_RIGHT, :2]) / 2.0
    torso_len = float(np.linalg.norm(shoulder_mid - hip_mid))
    if torso_len < 1e-6:
        torso_len = 1.0
    kp[:, :2] = (kp[:, :2] - hip_mid) / torso_len
    return kp


def _median3(xy: np.ndarray) -> np.ndarray:
    """3-frame median filter on (n, 33, 2) coordinates. Kills the single-frame"""
    if len(xy) < 3:
        return xy
    out = xy.copy()
    out[1:-1] = np.median(np.stack([xy[:-2], xy[1:-1], xy[2:]]), axis=0)
    return out


def _normalize_sequence(kp_seq: np.ndarray) -> np.ndarray:
    """Clip-level cleanup: despike, center on the hips, scale by body height."""
    if len(kp_seq) == 0:
        return kp_seq
    kp = kp_seq.copy()
    kp[:, :, :2] = _median3(kp[:, :, :2])
    hip_mid = (kp[:, HIP_LEFT, :2] + kp[:, HIP_RIGHT, :2]) / 2.0
    sho_mid = (kp[:, SHOULDER_LEFT, :2] + kp[:, SHOULDER_RIGHT, :2]) / 2.0
    ank_mid = (kp[:, ANKLE_LEFT, :2] + kp[:, ANKLE_RIGHT, :2]) / 2.0
    heights = np.abs(sho_mid[:, 1] - ank_mid[:, 1])
    heights = heights[heights > 1e-6]
    scale = float(np.median(heights)) if len(heights) else 1.0
    kp[:, :, :2] = (kp[:, :, :2] - hip_mid[:, None, :]) / scale
    return kp


def _estimate_pan(grays: list[np.ndarray]) -> np.ndarray:
    """Per-frame horizontal background shift in px (at 320px width), + = scene moved right."""
    n = len(grays)
    if n < 2:
        return np.zeros(max(n, 1), dtype=np.float32)
    stack = np.stack(grays)
    col_activity = stack.std(axis=0).mean(axis=0)
    active = np.where(col_activity > 2.0)[0]
    x0, x1 = (int(active[0]), int(active[-1]) + 1) if len(active) > 10 else (0, stack.shape[2])
    strip_w = max(10, int(PAN_STRIP_FRAC * (x1 - x0)))
    strips = [(x0, x0 + strip_w), (x1 - strip_w, x1)]
    rows = slice(18, 135)  # skip the broadcast graphics up top and scoreboard below
    window = cv2.createHanningWindow((strip_w, rows.stop - rows.start), cv2.CV_32F)

    pan = [0.0]
    for i in range(1, n):
        shifts = []
        for a, b in strips:
            (dx, _), response = cv2.phaseCorrelate(grays[i - 1][rows, a:b], grays[i][rows, a:b], window)
            if response > PAN_MIN_RESPONSE:
                shifts.append(dx)
        pan.append(float(np.median(shifts)) if shifts else pan[-1])
    return np.array(pan, dtype=np.float32)


def extract_keypoints_from_frame(frame: np.ndarray) -> np.ndarray:
    """One BGR frame -> (33, 4) normalized keypoints. All zeros if nobody's there."""
    global _landmarker_single
    if _landmarker_single is None:
        _landmarker_single = _make_landmarker(mp.tasks.vision.RunningMode.IMAGE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker_single.detect(mp_image)
    return _normalize(_landmarks_to_array(result))


def extract_keypoints_and_pan_from_video(video_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Whole video -> ((n, 33, 4) cleaned keypoints, (n, 2) motion track)."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        # some files report 0 or NaN fps. mediapipe just wants increasing
        # timestamps, so 30 works fine as a stand-in
        fps = 30.0

    raw: list[np.ndarray] = []
    grays: list[np.ndarray] = []
    prev = np.zeros((N_LANDMARKS, 4), dtype=np.float32)

    with _make_landmarker(mp.tasks.vision.RunningMode.VIDEO) as landmarker:
        with tqdm(total=total, desc=video_path.stem[:40], unit="frame") as bar:
            frame_idx = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_ms = int(frame_idx * 1000 / fps)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                kp = _landmarks_to_array(result)
                # if a joint basically disappeared (mask, motion blur...), keep
                # wherever it was last frame
                low_vis = kp[:, 3] < VISIBILITY_THRESHOLD
                kp[low_vis, :3] = prev[low_vis, :3]
                prev = kp.copy()
                raw.append(kp)

                small = cv2.resize(frame, PAN_DOWNSCALE)
                grays.append(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32))

                frame_idx += 1
                bar.update(1)

    cap.release()
    if not raw:
        return (np.zeros((0, N_LANDMARKS, 4), dtype=np.float32),
                np.zeros((0, 2), dtype=np.float32))
    raw_stack = np.stack(raw)
    hip_x = (raw_stack[:, HIP_LEFT, 0] + raw_stack[:, HIP_RIGHT, 0]) / 2.0  # pre-normalization
    motion = np.stack([_estimate_pan(grays), hip_x], axis=1).astype(np.float32)  # (n, 2)
    return _normalize_sequence(raw_stack), motion


def extract_keypoints_from_video(video_path: str | Path) -> np.ndarray:
    """Whole video -> (n, 33, 4) cleaned keypoints. Same as the function above,"""
    return extract_keypoints_and_pan_from_video(video_path)[0]


if __name__ == "__main__":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    kp_blank = extract_keypoints_from_frame(blank)
    assert kp_blank.shape == (N_LANDMARKS, 4)
    assert np.allclose(kp_blank, 0.0)
    print("test 1 ok: blank frame gives zeros with the right shape")
    
    sample_dir = PROJECT_ROOT / "data" / "sample_frames"
    # leave pose_viz.jpg out, that's our own output. otherwise reruns detect on
    # the drawn-over image and stack stale skeletons on top of each other
    samples = sorted(p for p in sample_dir.glob("*.jpg") if p.stem != "pose_viz")
    if not samples:
        print("no sample frames around, skipping tests 2 and 3")  
        sys.exit(0)
     
    sample_img = cv2.imread(str(samples[0]))
    kp = extract_keypoints_from_frame(sample_img)
    assert kp.shape == (N_LANDMARKS, 4)
    
    if np.any(kp[:, 3] > 0):
        hip_mid = (kp[HIP_LEFT, :2] + kp[HIP_RIGHT, :2]) / 2.0
        assert np.allclose(hip_mid, 0.0, atol=1e-4), f"hips should be at (0,0), got {hip_mid}"
        print("test 2 ok: found a person and the hips ended up at (0, 0)")
    else:
        print("test 2 skipped: no person in the sample frame")
    
    viz_img = sample_img.copy()
    h, w = viz_img.shape[:2]

    with _make_landmarker(mp.tasks.vision.RunningMode.IMAGE) as viz_lm:
        rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        viz_result = viz_lm.detect(mp_img)

    if viz_result.pose_landmarks:
        lms = viz_result.pose_landmarks[0]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for a, b in BODY_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(viz_img, pts[a], pts[b], (0, 255, 0), 2)
        for pt in pts:
            cv2.circle(viz_img, pt, 4, (0, 0, 255), -1)
        viz_out = sample_dir / "pose_viz.jpg"
        cv2.imwrite(str(viz_out), viz_img)
        print(f"test 3 ok: drew the skeleton, check {viz_out}")
    else:
        print("test 3 skipped: nothing to draw")

    print("\nall good")
