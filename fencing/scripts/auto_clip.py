"""Experimental auto-detection and clip cutting for action classes.

Runs MediaPipe on a raw video, detects advance/retreat/lunge/parry windows from
keypoint velocities, and saves clips to data/clips/auto_test/<action>/.

Review the output and move good clips to data/clips/<action>/.

Run from project root:  python scripts/auto_clip.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_pipeline import _landmarks_to_array, _make_landmarker, download_pose_model

VIDEO = (
    PROJECT_ROOT / "data" / "raw_video"
    / "2025 122 SWF Coupe du Monde, Vancouver ERRIGO Arianna vs FAVARETTO Martina_720p_deinterlaced.mp4"
)
AUTO_TEST_DIR = PROJECT_ROOT / "data" / "clips" / "auto_test"

MAX_FRAMES = 3000
CLIPS_PER_CLASS = 2
CLIP_PAD = 8

HIP_VEL_THRESH = 0.0025
LUNGE_WRIST_THRESH = 0.012
PARRY_WRIST_THRESH = 0.006
HIP_STILL_THRESH = 0.004

SMOOTH_W = 7
MIN_ADV_FRAMES = 14
MIN_LUNGE_FRAMES = 5
MIN_PARRY_FRAMES = 4
MERGE_GAP = 20

HIP_L, HIP_R = 23, 24
WRIST_L, WRIST_R = 15, 16
ACTIONS = ["advance", "retreat", "lunge", "parry"]


def moving_avg(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr.copy()
    kernel = np.ones(w) / w
    if arr.ndim == 1:
        return np.convolve(arr, kernel, mode="same")
    return np.column_stack([np.convolve(arr[:, i], kernel, mode="same") for i in range(arr.shape[1])])


def first_diff(arr: np.ndarray) -> np.ndarray:
    d = np.diff(arr, axis=0)
    return np.concatenate([np.zeros_like(arr[:1]), d], axis=0)


def find_windows(mask: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Find contiguous True regions, merge nearby ones, filter by min length."""
    if not np.any(mask):
        return []
    padded = np.concatenate([[False], mask, [False]])
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends = np.where(padded[:-1] & ~padded[1:])[0] - 1
    windows = list(zip(starts.tolist(), ends.tolist()))

    merged = [list(windows[0])]
    for s, e in windows[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    return [(s, e) for s, e in merged if e - s + 1 >= min_len]


def top_windows(windows: list[tuple[int, int]], score: np.ndarray, n: int) -> list[tuple[int, int]]:
    """Return up to n highest-scoring non-overlapping windows."""
    scored = sorted(
        [(float(np.mean(np.abs(score[s:e + 1]))), s, e) for s, e in windows],
        reverse=True,
    )
    picked: list[tuple[int, int]] = []
    for _, s, e in scored:
        if len(picked) >= n:
            break
        if not any(not (e < ps or s > pe) for ps, pe in picked):
            picked.append((s, e))
    return picked


def extract_raw_kp(video_path: Path, max_frames: int) -> tuple[np.ndarray, float]:
    """Extract raw (un-normalized) MediaPipe keypoints from the first max_frames frames."""
    download_pose_model()
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), max_frames)
    rows: list[np.ndarray] = []

    with _make_landmarker(mp.tasks.vision.RunningMode.VIDEO) as landmarker:
        with tqdm(total=total, desc="Pose extraction", unit="frame") as bar:
            for idx in range(total):
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_img, int(idx * 1000 / fps))
                rows.append(_landmarks_to_array(result))
                bar.update(1)

    cap.release()
    return np.stack(rows), fps


def detect_actions(kp: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    detected = kp[:, HIP_L, 3] > 0.3

    hip_x = (kp[:, HIP_L, 0] + kp[:, HIP_R, 0]) / 2.0
    wl_s = moving_avg(kp[:, WRIST_L, :2], SMOOTH_W)
    wr_s = moving_avg(kp[:, WRIST_R, :2], SMOOTH_W)
    hip_x_s = moving_avg(hip_x, SMOOTH_W)

    hip_vel_x = first_diff(hip_x_s)
    wrist_speed = np.maximum(
        np.linalg.norm(first_diff(wl_s), axis=1),
        np.linalg.norm(first_diff(wr_s), axis=1),
    )
    hip_vel_x[~detected] = 0.0
    wrist_speed[~detected] = 0.0

    facing_right = moving_avg(hip_x_s, 30) < 0.5
    lunge_suppression = np.clip(1.0 - wrist_speed / LUNGE_WRIST_THRESH, 0.0, 1.0)
    advance_raw = np.where(facing_right, hip_vel_x, -hip_vel_x)

    advance_score = advance_raw * lunge_suppression
    retreat_score = -advance_raw * lunge_suppression
    lunge_score = np.where(wrist_speed >= LUNGE_WRIST_THRESH, wrist_speed, 0.0)
    stillness = np.clip(1.0 - np.abs(hip_vel_x) / HIP_STILL_THRESH, 0.0, 1.0)
    parry_score = np.where(wrist_speed >= PARRY_WRIST_THRESH, wrist_speed * stillness, 0.0)

    all_windows = {
        "advance": find_windows(advance_score > HIP_VEL_THRESH, MIN_ADV_FRAMES, MERGE_GAP),
        "retreat": find_windows(retreat_score > HIP_VEL_THRESH, MIN_ADV_FRAMES, MERGE_GAP),
        "lunge":   find_windows(lunge_score > 0, MIN_LUNGE_FRAMES, MERGE_GAP),
        "parry":   find_windows(parry_score > 0, MIN_PARRY_FRAMES, MERGE_GAP),
    }
    score_map = {
        "advance": advance_score, "retreat": retreat_score,
        "lunge": lunge_score, "parry": parry_score,
    }
    return {action: top_windows(wins, score_map[action], CLIPS_PER_CLASS) for action, wins in all_windows.items()}


def save_clip(video_path: Path, start_frame: int, end_frame: int, fps: float, output_path: Path) -> None:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_frame / fps:.4f}",
        "-i", str(video_path),
        "-t", f"{(end_frame - start_frame + 1) / fps:.4f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
        str(output_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> None:
    if not VIDEO.exists():
        print(f"Video not found: {VIDEO}")
        sys.exit(1)

    print(f"Video: {VIDEO.name}")
    print(f"Analysing first {MAX_FRAMES} frames (~{MAX_FRAMES / 30:.0f}s)\n")

    kp, fps = extract_raw_kp(VIDEO, MAX_FRAMES)
    print(f"\nPerson detected in {100 * np.mean(kp[:, HIP_L, 3] > 0.3):.0f}% of frames")

    windows = detect_actions(kp)
    print("\nDetected candidates:")
    for action in ACTIONS:
        for s, e in windows[action]:
            print(f"  {action:<10} frames {s:4d}-{e:4d}  ({(e - s + 1) / fps:.1f}s)")
        if not windows[action]:
            print(f"  {action:<10} -- none found")

    total_frames = len(kp)
    for action in ACTIONS:
        wins = windows[action]
        if not wins:
            continue
        out_dir = AUTO_TEST_DIR / action
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, (s, e) in enumerate(wins):
            start = max(0, s - CLIP_PAD)
            end = min(total_frames - 1, e + CLIP_PAD)
            out_path = out_dir / f"auto_{i:02d}.mp4"
            save_clip(VIDEO, start, end, fps, out_path)
            print(f"  Saved {out_path.relative_to(PROJECT_ROOT)}  ({(end - start + 1) / fps:.1f}s)")

    print(f"\nDone. Review clips in data/clips/auto_test/")


if __name__ == "__main__":
    main()
