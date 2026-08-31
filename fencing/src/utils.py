"""Overlay drawing helpers: skeleton, blade tip, action label."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from src.pose_pipeline import BODY_CONNECTIONS
except ImportError:  # running this file directly from src/
    from pose_pipeline import BODY_CONNECTIONS

BONE_COLOR = (0, 255, 0)
JOINT_COLOR = (0, 0, 255)
TIP_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)


def draw_skeleton(frame: np.ndarray, points: np.ndarray,
                  color: tuple[int, int, int] = BONE_COLOR) -> np.ndarray:
    """Draw the pose skeleton onto frame (in place) and return it."""
    pts = points.astype(int)
    ok = ~np.all(pts == 0, axis=1)
    for a, b in BODY_CONNECTIONS:
        if ok[a] and ok[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 2)
    for i in np.where(ok)[0]:
        cv2.circle(frame, tuple(pts[i]), 3, JOINT_COLOR, -1)
    return frame


def draw_blade_tip(frame: np.ndarray, tip: tuple[float, float] | None) -> np.ndarray:
    """Green dot on the blade point (no-op if it is None). Returns the frame."""
    if tip is not None:
        x, y = int(tip[0]), int(tip[1])
        cv2.circle(frame, (x, y), 8, TIP_COLOR, 2)
        cv2.circle(frame, (x, y), 3, TIP_COLOR, -1)
    return frame


def draw_action_label(frame: np.ndarray, action: str, confidence: float | None,
                      org: tuple[int, int] = (10, 40),
                      color: tuple[int, int, int] = (0, 200, 255)) -> np.ndarray:
    """Action name (+ confidence, if given) on a dark box so it reads over any background."""
    text = action if confidence is None else f"{action}  {confidence:.0%}"
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
    x, y = org
    cv2.rectangle(frame, (x - 6, y - th - 8), (x + tw + 6, y + base + 4), (0, 0, 0), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    return frame


if __name__ == "__main__":
    canvas = np.zeros((300, 400, 3), dtype=np.uint8)

    fake = np.zeros((33, 2))
    fake[11], fake[12] = (150, 80), (250, 80)   # shoulders
    fake[23], fake[24] = (160, 180), (240, 180)  # hips
    out = draw_skeleton(canvas, fake)
    assert out.shape == canvas.shape and out.sum() > 0
    print("test 1 ok: skeleton drew something, missing joints skipped")

    draw_blade_tip(canvas, (330.0, 40.0))
    draw_blade_tip(canvas, None)  # must not crash
    print("test 2 ok: blade tip drawn, None tolerated")

    draw_action_label(canvas, "lunge", 0.87)
    draw_action_label(canvas, "parry", 0.55, org=(200, 280))
    draw_action_label(canvas, "A: ready", None, org=(10, 120))  # ready tag, no %
    print("test 3 ok: labels drawn (with and without confidence)")

    print("\nall good")
