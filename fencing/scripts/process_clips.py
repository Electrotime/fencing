"""Run the pose pipeline on every action clip and save the keypoints as .npy files.

Clips that already have a keypoint file get skipped, so this is safe to rerun
whenever new clips get added. Use --force to redo everything (say, if a clip
got re-cut under the same name).

Run from project root:  python scripts/process_clips.py [--force]
"""

import argparse
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


def process_class(action: str, force: bool) -> tuple[int, int]:
    """Extract keypoints for every clip of one action. Returns (done, skipped)."""
    clips_dir = CLIPS_BASE / action
    kp_dir = KEYPOINTS_BASE / action
    clips_dir.mkdir(parents=True, exist_ok=True)
    kp_dir.mkdir(parents=True, exist_ok=True)

    clips = sorted(p for p in clips_dir.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS)
    if not clips:
        print(f"  [{action}] no clips here yet -> {clips_dir}")
        return 0, 0

    done = 0
    skipped = 0
    print(f"\n--- {action} ({len(clips)} clips) ---")
    for clip in clips:
        out_path = kp_dir / f"{clip.stem}.npy"
        if out_path.exists() and not force:
            print(f"  {clip.name} already done, skipping (--force to redo)")
            skipped += 1
            continue
        kp = extract_keypoints_from_video(clip)
        np.save(out_path, kp)
        print(f"  {clip.name} -> {kp.shape} -> {out_path.name}")
        done += 1

    return done, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pose keypoints from all the action clips.")
    parser.add_argument("--force", action="store_true",
                        help="redo clips even if their .npy already exists")
    args = parser.parse_args()

    done = 0
    skipped = 0
    for action in CLASS_NAMES:
        d, s = process_class(action, args.force)
        done += d
        skipped += s

    print(f"\nFinished: {done} clips processed, {skipped} skipped.")
    if done + skipped == 0:
        print("Drop some clips into data/clips/<action>/ and run this again.")


if __name__ == "__main__":
    main()
