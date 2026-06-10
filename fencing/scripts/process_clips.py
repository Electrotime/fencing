"""Batch pose extraction — runs the pose pipeline on every action clip and saves .npy files.

Run from project root:  python scripts/process_clips.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pose_pipeline import extract_keypoints_from_video

CLIPS_BASE = PROJECT_ROOT / "data" / "clips"
KEYPOINTS_BASE = PROJECT_ROOT / "data" / "keypoints"
CLASS_NAMES = ["advance", "lunge", "parry", "retreat"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


def process_class(action: str) -> int:
    clips_dir = CLIPS_BASE / action
    kp_dir = KEYPOINTS_BASE / action
    kp_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not clips:
        print(f"  [{action}] no clips found — add videos to {clips_dir}")
        return 0

    print(f"\n--- {action} ({len(clips)} clip{'s' if len(clips) != 1 else ''}) ---")
    for clip_path in clips:
        kp = extract_keypoints_from_video(clip_path)
        out_path = kp_dir / f"{clip_path.stem}.npy"
        np.save(out_path, kp)
        print(f"  {clip_path.name} -> {kp.shape} -> {out_path.name}")

    return len(clips)


def main() -> None:
    total = sum(process_class(action) for action in CLASS_NAMES)
    print(f"\nDone. {total} clip(s) processed across {len(CLASS_NAMES)} classes.")
    if total == 0:
        print("Populate data/clips/<action>/ with short video clips and re-run.")


if __name__ == "__main__":
    main()
