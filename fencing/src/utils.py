"""Overlay drawing helpers: skeleton, blade tip, action label."""

from __future__ import annotations

import cv2
import numpy as np

try:
    from src.pose_pipeline import BODY_CONNECTIONS
except ImportError:  # running this file directly from src/
    from pose_pipeline import BODY_CONNECTIONS

# neutral so the red/green blade trails carry the left/right coding on their own
BONE_COLOR = (205, 205, 205)
JOINT_COLOR = (150, 150, 150)
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
        cv2.circle(frame, tuple(pts[i]), 2, JOINT_COLOR, -1)
    return frame


def draw_blade_tip(frame: np.ndarray, tip: tuple[float, float] | None) -> np.ndarray:
    """Green dot on the blade point (no-op if it is None). Returns the frame."""
    if tip is not None:
        x, y = int(tip[0]), int(tip[1])
        cv2.circle(frame, (x, y), 8, TIP_COLOR, 2)
        cv2.circle(frame, (x, y), 3, TIP_COLOR, -1)
    return frame


def draw_blade_trail(frame, trails, colors, glow=True):
    """Luminous ribbon per blade: saturated glow under a hot near-white core."""
    glow_m = np.zeros_like(frame)
    core_m = np.zeros_like(frame)
    for slot, pts in trails.items():
        if len(pts) < 2:
            continue
        base = colors.get(slot, (255, 255, 255))
        n = len(pts)
        for i in range(1, n):
            age = i / (n - 1)                       # 0 oldest, 1 newest
            p, q = pts[i - 1], pts[i]
            a = (int(p[0]), int(p[1])); b = (int(q[0]), int(q[1]))
            cv2.line(glow_m, a, b, tuple(int(c * age ** 0.6) for c in base),
                     max(2, int(3 + 11 * age)), cv2.LINE_AA)
            if age > 0.35:
                k = (age - 0.35) / 0.65
                hot = tuple(int(c + (255 - c) * 0.75 * k) for c in base)
                cv2.line(core_m, a, b, tuple(int(c * k) for c in hot),
                         max(1, int(1 + 3 * k)), cv2.LINE_AA)
    if glow:
        glow_m = cv2.GaussianBlur(glow_m, (0, 0), 9)
        core_m = cv2.GaussianBlur(core_m, (0, 0), 2)
    # cv2.add SATURATES; np.add on uint8 wraps, which turned the red trail green
    cv2.add(frame, glow_m, dst=frame)
    cv2.add(frame, core_m, dst=frame)
    for slot, pts in trails.items():
        if pts:
            base = colors.get(slot, (255, 255, 255))
            hd = (int(pts[-1][0]), int(pts[-1][1]))
            cv2.circle(frame, hd, 9, tuple(int(c * 0.5) for c in base), -1, cv2.LINE_AA)
            cv2.circle(frame, hd, 4, (255, 255, 255), -1, cv2.LINE_AA)
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
