"""Sample frames from raw videos for Roboflow blade labeling.

Pulls a fixed budget of ~400 frames total (evenly spaced, balanced across videos)
and saves them as JPEGs to data/blade_frames/.

Run from project root:  python scripts/extract_blade_frames.py
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_video"
BLADE_FRAMES_DIR = PROJECT_ROOT / "data" / "blade_frames"

TARGET_TOTAL_FRAMES = 400
JPEG_QUALITY = 95
DEINTERLACED_SUFFIX = "_deinterlaced"


def find_videos(raw_dir: Path) -> list[Path]:
    """Return videos to process, skipping interlaced originals that have a deinterlaced twin."""
    all_videos = sorted(raw_dir.glob("*.mp4"))
    deinterlaced_bases = {
        v.stem[: -len(DEINTERLACED_SUFFIX)]
        for v in all_videos
        if v.stem.endswith(DEINTERLACED_SUFFIX)
    }

    keep = []
    for video in all_videos:
        if not video.stem.endswith(DEINTERLACED_SUFFIX) and video.stem in deinterlaced_bases:
            print(f"  (skipping interlaced original: {video.name})")
            continue
        keep.append(video)
    return keep


def pick_frame_indices(total_frames: int, count: int) -> set[int]:
    """Return evenly spaced frame indices across the video."""
    if total_frames <= 0 or count <= 0:
        return set()
    count = min(count, total_frames)
    return set(np.linspace(0, total_frames - 1, count, dtype=int).tolist())


def extract_frames(video_path: Path, output_dir: Path, count: int) -> int:
    """Save evenly-spaced frames as JPEGs. Returns number of frames written."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path.name}, skipping")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = pick_frame_indices(total_frames, count)

    saved = 0
    frame_index = 0
    with tqdm(total=total_frames, desc=video_path.stem[:40], unit="frame") as bar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index in target_indices:
                out_name = f"{video_path.stem}_f{frame_index:06d}.jpg"
                cv2.imwrite(str(output_dir / out_name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
                saved += 1
            frame_index += 1
            bar.update(1)

    cap.release()
    return saved


def main() -> None:
    BLADE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    videos = find_videos(RAW_VIDEO_DIR)
    if not videos:
        print(f"No .mp4 files found in {RAW_VIDEO_DIR}")
        return

    per_video = max(1, TARGET_TOTAL_FRAMES // len(videos))
    print(f"Found {len(videos)} video(s), targeting ~{per_video} frames each\n")

    total_saved = sum(extract_frames(v, BLADE_FRAMES_DIR, per_video) for v in videos)
    print(f"\nDone. Saved {total_saved} frame(s) to {BLADE_FRAMES_DIR}")


if __name__ == "__main__":
    main()
