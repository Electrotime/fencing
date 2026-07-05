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
    """Center the keypoints on the hips and scale by shoulder width, so it doesn't
    matter where the fencer is standing or how big they are in frame.
    z and visibility stay untouched."""
    kp = kp.copy()
    hip_mid = (kp[HIP_LEFT, :2] + kp[HIP_RIGHT, :2]) / 2.0
    shoulder_w = float(np.linalg.norm(kp[SHOULDER_LEFT, :2] - kp[SHOULDER_RIGHT, :2]))
    if shoulder_w < 1e-6:
        shoulder_w = 1.0
    kp[:, :2] = (kp[:, :2] - hip_mid) / shoulder_w
    return kp


def extract_keypoints_from_frame(frame: np.ndarray) -> np.ndarray:
    """One BGR frame -> (33, 4) normalized keypoints. All zeros if nobody's there."""
    global _landmarker_single
    if _landmarker_single is None:
        _landmarker_single = _make_landmarker(mp.tasks.vision.RunningMode.IMAGE)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _landmarker_single.detect(mp_image)
    return _normalize(_landmarks_to_array(result))


def extract_keypoints_from_video(video_path: str | Path) -> np.ndarray:
    """Whole video -> (n_frames, 33, 4). Joints that drop out get carried over
    from the previous frame instead of jumping to garbage."""
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

    out: list[np.ndarray] = []
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
                out.append(_normalize(kp))

                frame_idx += 1
                bar.update(1)

    cap.release()
    return np.stack(out) if out else np.zeros((0, N_LANDMARKS, 4), dtype=np.float32)


if __name__ == "__main__":
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    kp_blank = extract_keypoints_from_frame(blank)
    assert kp_blank.shape == (N_LANDMARKS, 4)
    assert np.allclose(kp_blank, 0.0)
    print("test 1 ok: blank frame gives zeros with the right shape")

    sample_dir = PROJECT_ROOT / "data" / "sample_frames"
    samples = sorted(sample_dir.glob("*.jpg"))
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
