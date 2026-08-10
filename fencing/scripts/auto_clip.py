"""Experiment: auto-find action clips in a match video.

Runs pose estimation over a raw video, looks for advance/retreat/lunge/parry
patterns in how the hips and wrists move, and cuts candidate clips into
data/clips/auto_test/<action>/. The clips are guesses. Watch them and move
the good ones into data/clips/<action>/.

Run from project root:  python scripts/auto_clip.py [path/to/video.mp4]
(no argument = the Errigo vs Favaretto video)
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

DEFAULT_VIDEO = (
    PROJECT_ROOT / "data" / "raw_video"
    / "2025 122 SWF Coupe du Monde, Vancouver ERRIGO Arianna vs FAVARETTO Martina_720p_deinterlaced.mp4"
)
AUTO_TEST_DIR = PROJECT_ROOT / "data" / "clips" / "auto_test"

MAX_FRAMES = 3000    # only scan the start of the video, keeps the run short
CLIPS_PER_CLASS = 2  # keep the best few candidates per action
CLIP_PAD = 8         # extra frames on both ends so clips don't start mid-motion

# thresholds tuned by eyeballing velocity plots from the Errigo/Favaretto video
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
    """Simple moving average, works on 1D or (n, k) arrays."""
    if w <= 1:
        return arr.copy()
    kernel = np.ones(w) / w
    if arr.ndim == 1:
        return np.convolve(arr, kernel, mode="same")
    return np.column_stack([np.convolve(arr[:, i], kernel, mode="same") for i in range(arr.shape[1])])


def first_diff(arr: np.ndarray) -> np.ndarray:
    """Frame-to-frame change, front-padded with zeros so the length stays the same."""
    d = np.diff(arr, axis=0)
    return np.concatenate([np.zeros_like(arr[:1]), d], axis=0)


def find_windows(mask: np.ndarray, min_len: int, merge_gap: int) -> list[tuple[int, int]]:
    """Contiguous stretches of True, with nearby ones merged and short ones dropped."""
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
    """Up to n best-scoring windows that don't overlap each other."""
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
    """Raw (un-normalized) keypoints for the first max_frames frames of the video."""
    download_pose_model()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"couldn't open {video_path}, is the path right?")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
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
    if not rows:
        sys.exit(f"read zero frames from {video_path.name}, something's wrong with the file")
    return np.stack(rows), fps


def detect_actions(kp: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    detected = kp[:, HIP_L, 3] > 0.3

    # track where the hips are and how fast each wrist is moving
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

    # which side of the frame the fencer lives on decides what "forward" means
    facing_right = moving_avg(hip_x_s, 30) < 0.5
    advance_raw = np.where(facing_right, hip_vel_x, -hip_vel_x)

    # a flying wrist usually means a lunge, so damp advance/retreat scores there
    not_lunging = np.clip(1.0 - wrist_speed / LUNGE_WRIST_THRESH, 0.0, 1.0)
    advance_score = advance_raw * not_lunging
    retreat_score = -advance_raw * not_lunging

    lunge_score = np.where(wrist_speed >= LUNGE_WRIST_THRESH, wrist_speed, 0.0)

    # parry = wrist moving while the hips stay put
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
    """Cut [start, end] out of the video with ffmpeg (re-encoded so the cut lands on the frame)."""
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start_frame / fps:.4f}",
        "-i", str(video_path),
        "-t", f"{(end_frame - start_frame + 1) / fps:.4f}",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-an",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError:
        sys.exit("ffmpeg isn't installed or isn't on PATH. Get it and rerun "
                 "(on Windows: winget install Gyan.FFmpeg)")
    except subprocess.CalledProcessError as e:
        tail = e.stderr.decode(errors="replace")[-500:]
        sys.exit(f"ffmpeg died while cutting {output_path.name}:\n{tail}")
    except subprocess.TimeoutExpired:
        sys


def main() -> None:
    video = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_VIDEO
    if not video.exists():
        print(f"video not found: {video}")
        print("usage: python scripts/auto_clip.py [path/to/video.mp4]")
        sys.exit(1)

    print(f"Video: {video.name}")
    print(f"Looking at the first {MAX_FRAMES} frames\n")

    kp, fps = extract_raw_kp(video, MAX_FRAMES)
    print(f"\nGot a pose in {100 * np.mean(kp[:, HIP_L, 3] > 0.3):.0f}% of frames")

    windows = detect_actions(kp)
    print("\nCandidates:")
    for action in ACTIONS:
        for s, e in windows[action]:
            print(f"  {action:<10} frames {s:4d}-{e:4d}  ({(e - s + 1) / fps:.1f}s)")
        if not windows[action]:
            print(f"  {action:<10} nothing found")

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
            # keep the source name in the clip name so two videos don't clobber each other
            out_path = out_dir / f"{video.stem[:40]}_auto_{i:02d}.mp4"
            save_clip(video, start, end, fps, out_path)
            print(f"  saved {out_path.relative_to(PROJECT_ROOT)}  ({(end - start + 1) / fps:.1f}s)")

    print("\nDone. Clips are in data/clips/auto_test/, watch them and keep the good ones.")


if __name__ == "__main__":
    main()
