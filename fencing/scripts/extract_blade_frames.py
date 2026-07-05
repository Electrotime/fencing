"""Grab frames from the raw match videos so I can label blades in Roboflow.

Saves ~400 frames total (TARGET_TOTAL_FRAMES), spread evenly across all the
videos, as jpgs in data/blade_frames/.

Run from project root:  python scripts/extract_blade_frames.py
"""

from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_VIDEO_DIR = PROJECT_ROOT / "data" / "raw_video"
BLADE_FRAMES_DIR = PROJECT_ROOT / "data" / "blade_frames"

TARGET_TOTAL_FRAMES = 400  # roughly how many labeled images I want in total
JPEG_QUALITY = 95
DEINTERLACED_SUFFIX = "_deinterlaced"


def find_videos(raw_dir: Path) -> list[Path]:
    """Pick which videos to sample. If a match has a deinterlaced copy, use that
    and skip the interlaced original so the same match isn't sampled twice."""
    all_videos = sorted(raw_dir.glob("*.mp4"))
    have_deinterlaced = {
        v.stem[: -len(DEINTERLACED_SUFFIX)]
        for v in all_videos
        if v.stem.endswith(DEINTERLACED_SUFFIX)
    }

    keep = []
    for video in all_videos:
        if not video.stem.endswith(DEINTERLACED_SUFFIX) and video.stem in have_deinterlaced:
            print(f"  skipping {video.name} (deinterlaced copy exists)")
            continue
        keep.append(video)
    return keep


def pick_frame_indices(total_frames: int, count: int) -> set[int]:
    """Evenly spaced frame numbers across the whole video."""
    if total_frames <= 0 or count <= 0:
        return set()
    count = min(count, total_frames)
    return set(np.linspace(0, total_frames - 1, count, dtype=int).tolist())


def extract_frames(video_path: Path, output_dir: Path, count: int) -> int:
    """Save evenly spaced frames from one video as jpgs. Returns how many got written.

    Jumps straight to each frame we want instead of decoding the whole video,
    which is way faster since we only keep ~50 frames out of thousands. Seeking
    isn't perfectly frame-accurate on every codec, but for labeling frames it
    really doesn't matter.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  couldn't open {video_path.name}, skipping it")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"  {video_path.name} has no frame count in its metadata, skipping it")
        cap.release()
        return 0

    saved = 0
    failed = 0
    # sorted so every seek moves forward (backwards seeks are slower)
    for idx in tqdm(sorted(pick_frame_indices(total_frames, count)),
                    desc=video_path.stem[:40], unit="frame"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            failed += 1
            continue
        name = f"{video_path.stem}_f{idx:06d}.jpg"
        if cv2.imwrite(str(output_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
            saved += 1
        else:
            failed += 1

    if failed:
        print(f"  heads up: {failed} frames from {video_path.name} couldn't be read/written")
    cap.release()
    return saved


def main() -> None:
    BLADE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    videos = find_videos(RAW_VIDEO_DIR)
    if not videos:
        print(f"No .mp4 files in {RAW_VIDEO_DIR}")
        return

    # split the budget evenly, first few videos pick up the leftover
    base, extra = divmod(TARGET_TOTAL_FRAMES, len(videos))
    quotas = [max(1, base + (1 if i < extra else 0)) for i in range(len(videos))]
    print(f"Found {len(videos)} videos, taking about {max(quotas)} frames from each\n")

    total_saved = sum(extract_frames(v, BLADE_FRAMES_DIR, q) for v, q in zip(videos, quotas))
    print(f"\nDone, saved {total_saved} frames to {BLADE_FRAMES_DIR}")


if __name__ == "__main__":
    main()
