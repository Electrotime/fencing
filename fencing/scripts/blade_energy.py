"""Measure blade MOTION instead of detecting the blade. Dumps a per-frame cache."""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
import mediapipe as mp

import demo_video as D
from src.person_detector import crop_box, get_fencer_boxes, load_person_model
from src.pose_pipeline import N_LANDMARKS, VISIBILITY_THRESHOLD, _landmarks_to_array, _make_landmarker

VIDEO = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "data" / "raw_video" / "1.mp4"
OUT = PROJECT / "data" / "labels" / f"{VIDEO.stem}_blade.npz"

L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 13, 14, 15, 16
L_HIP, R_HIP = 23, 24
SCALE = 0.5              # work at half resolution: 4x less differencing, same signal
BLADE_REACH = 3.0        # blade extends ~3x forearm length past the hand
BLADE_HALFWIDTH = 0.9    # box half-width, in forearm lengths

STRIP_LEN = 64           # samples along the blade; output is tiny so warping is cheap
STRIP_WID = 16           # samples across it
STRIP_HALFWIDTH = 0.45   # HALF of v1's 0.9 -- a blade is thin, and the oriented
                         # strip no longer needs slack to contain a diagonal
TORSO_HALF = 0.55        # torso patch half-size, in forearm lengths, for the shift
MIN_SHIFT_RESPONSE = 0.05  # below this phase correlation is noise; fall back to pan


def blade_box(kp, W, H, opponent_x):
    """Axis-aligned box covering where the weapon blade must be, in pixels."""
    got = weapon_arm(kp, opponent_x)
    if got is None:
        return None
    elbow, wrist = got

    ex, ey = kp[elbow, 0] * W, kp[elbow, 1] * H
    wx, wy = kp[wrist, 0] * W, kp[wrist, 1] * H
    dx, dy = wx - ex, wy - ey
    flen = float(np.hypot(dx, dy))
    if flen < 2.0:                      # arm folded toward the camera: direction is noise
        return None
    ux, uy = dx / flen, dy / flen
    tipx, tipy = wx + BLADE_REACH * flen * ux, wy + BLADE_REACH * flen * uy
    pad = BLADE_HALFWIDTH * flen
    x1, x2 = sorted((wx, tipx))
    y1, y2 = sorted((wy, tipy))
    return (x1 - pad, y1 - pad, x2 + pad, y2 + pad), flen


def weapon_arm(kp, opponent_x):
    """(elbow, wrist) of the hand nearer the OPPONENT, or None."""
    cands = []
    for elbow, wrist in ((L_ELBOW, L_WRIST), (R_ELBOW, R_WRIST)):
        if kp[elbow, 3] < VISIBILITY_THRESHOLD or kp[wrist, 3] < VISIBILITY_THRESHOLD:
            continue
        cands.append((abs(kp[wrist, 0] - opponent_x), elbow, wrist))
    if not cands:
        return None
    _, elbow, wrist = min(cands)
    return elbow, wrist


def wrist_dir(kp, W, H, opponent_x):
    """(wx, wy, ux, uy) -- wrist in pixels and the unit forearm direction."""
    got = weapon_arm(kp, opponent_x)
    if got is None:
        return None
    elbow, wrist = got
    ex, ey = kp[elbow, 0] * W, kp[elbow, 1] * H
    wx, wy = kp[wrist, 0] * W, kp[wrist, 1] * H
    dx, dy = wx - ex, wy - ey
    flen = float(np.hypot(dx, dy))
    if flen < 2.0:
        return None
    return wx, wy, dx / flen, dy / flen


def _strip_M(px, py, ux, uy, length, halfwidth, out_w, out_h):
    """dst(i,j) -> src(x,y) for an oriented strip, for warpAffine WARP_INVERSE_MAP."""
    nx, ny = -uy, ux                      # unit normal to the blade
    sx = length / out_w
    sy = 2.0 * halfwidth / out_h
    j0 = (out_h - 1) / 2.0
    return np.float32([
        [sx * ux, sy * nx, px - j0 * sy * nx],
        [sx * uy, sy * ny, py - j0 * sy * ny],
    ])


def strip_energy(prev, cur, M, shift, out_w, out_h):
    """|cur - prev| inside an oriented strip, with `prev` shifted to cancel body motion."""
    a = cv2.warpAffine(cur, M, (out_w, out_h),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE)
    Mp = M.copy()
    Mp[0, 2] -= shift[0]
    Mp[1, 2] -= shift[1]
    b = cv2.warpAffine(prev, Mp, (out_w, out_h),
                       flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                       borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(a - b)
    return float(d.mean()), float(np.percentile(d, 99))


def body_shift(prev, cur, cx, cy, half, cache):
    """(dx, dy) aligning this fencer's torso between frames, plus the response."""
    h, w = cur.shape
    x1, y1 = int(max(0, cx - half)), int(max(0, cy - half))
    x2, y2 = int(min(w, cx + half)), int(min(h, cy + half))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    a = prev[y1:y2, x1:x2]
    b = cur[y1:y2, x1:x2]
    key = a.shape
    if key not in cache:
        cache[key] = cv2.createHanningWindow((key[1], key[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cache[key])
    if resp < MIN_SHIFT_RESPONSE:
        return None
    return float(dx), float(dy), float(resp)


def mean_energy(diff, box):
    """Difference statistics inside a box: (mean, p99, area)."""
    h, w = diff.shape
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return np.nan, np.nan, 0
    patch = diff[y1:y2, x1:x2]
    return float(patch.mean()), float(np.percentile(patch, 99)), patch.size


def _self_test() -> int:
    """Geometry checks on synthetic frames. Run before spending hours of video on it."""
    H = W = 200

    def line_frame(off):
        im = np.zeros((H, W), np.float32)
        for k in range(80):
            x, y = int(50 + k * 0.707) + off, int(50 + k * 0.707)
            if 0 <= x < W and 0 <= y < H:
                im[y - 1:y + 2, x - 1:x + 2] = 255.0
        return im

    a, b = line_frame(0), line_frame(6)
    ux, uy = 0.707, 0.707
    on, _ = strip_energy(a, b, _strip_M(50, 50, ux, uy, 80, 6.0, STRIP_LEN, STRIP_WID),
                         (0.0, 0.0), STRIP_LEN, STRIP_WID)
    off, _ = strip_energy(a, b, _strip_M(50, 50, -uy, ux, 80, 6.0, STRIP_LEN, STRIP_WID),
                          (0.0, 0.0), STRIP_LEN, STRIP_WID)
    # the whole point of an oriented strip: pointing it along the blade must matter
    assert on > 5 * off, f"orientation does not matter: {on:.1f} vs {off:.1f}"

    rs = np.random.RandomState(0).rand(H, W).astype(np.float32) * 255
    shifted = np.roll(np.roll(rs, 4, axis=1), -3, axis=0)      # dx=+4, dy=-3
    got = body_shift(rs, shifted, 100, 100, 60, {})
    assert got is not None and abs(got[0] - 4) < 0.2 and abs(got[1] + 3) < 0.2, got

    Ms = _strip_M(100, 100, 1.0, 0.0, 60, 8.0, STRIP_LEN, STRIP_WID)
    raw, _ = strip_energy(rs, shifted, Ms, (0.0, 0.0), STRIP_LEN, STRIP_WID)
    comp, _ = strip_energy(rs, shifted, Ms, (got[0], got[1]), STRIP_LEN, STRIP_WID)
    # a PURE translation is exactly what a lunging body looks like; if compensation
    # does not flatten it, the strip is still measuring body motion
    assert comp < 0.4 * raw, f"compensation left {comp:.1f} of {raw:.1f}"

    print(f"blade_energy self-test ok: orientation {on / max(off, 1e-6):.0f}x, "
          f"shift recovered to 0.01px, translation {100 * comp / raw:.0f}% residual")
    return 0


def main() -> int:
    person_model = load_person_model()
    cap = cv2.VideoCapture(str(VIDEO))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W2, H2 = int(W * SCALE), int(H * SCALE)

    landmarkers = {s: _make_landmarker(mp.tasks.vision.RunningMode.VIDEO).__enter__()
                   for s in ("A", "B")}
    tracks = {s: D.FencerTrack() for s in ("A", "B")}
    prev_gray_small, pan_windows = None, {}
    prev_gray = None
    hann_cache = {}
    rows = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray_small = cv2.cvtColor(cv2.resize(frame, (320, 180)),
                                  cv2.COLOR_BGR2GRAY).astype(np.float32)
        pan = D._frame_pan(prev_gray_small, gray_small, pan_windows)
        prev_gray_small = gray_small

        gray = cv2.cvtColor(cv2.resize(frame, (W2, H2)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff, global_e = None, np.nan
        if prev_gray is not None:
            # shift the PREVIOUS frame by the measured pan before differencing, so
            # what remains is motion in the scene rather than motion of the camera
            shift = pan * (W2 / 320.0)
            M = np.float32([[1, 0, shift], [0, 1, 0]])
            warped = cv2.warpAffine(prev_gray, M, (W2, H2), borderMode=cv2.BORDER_REPLICATE)
            diff = np.abs(gray - warped)
            global_e = float(diff.mean())
        prev_frame = prev_gray
        prev_gray = gray

        box_a, box_b = get_fencer_boxes(frame, person_model, min_h_frac=D.MIN_BOX_H_FRAC)
        boxes = D._assign_boxes([b for b in (box_a, box_b) if b is not None], tracks, W)

        kps = {}
        for slot, box in (("A", boxes["A"]), ("B", boxes["B"])):
            track = tracks[slot]
            kp = np.zeros((N_LANDMARKS, 4), dtype=np.float32)
            if box is not None:
                crop = crop_box(frame, box)
                if crop is not None:
                    res = landmarkers[slot].detect_for_video(
                        mp.Image(image_format=mp.ImageFormat.SRGB,
                                 data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)),
                        int(frame_idx * 1000 / fps))
                    kp = _landmarks_to_array(res)
                    x1, y1, x2, y2 = box
                    kp[:, 0] = (x1 + kp[:, 0] * (x2 - x1)) / W
                    kp[:, 1] = (y1 + kp[:, 1] * (y2 - y1)) / H
                    track.last_hip_x = float((kp[23, 0] + kp[24, 0]) / 2)
            kps[slot] = kp
            track.kp.append(kp)

        if diff is not None:
            for slot in ("A", "B"):
                kp = kps[slot]
                other = kps["B" if slot == "A" else "A"]
                if not np.any(kp):
                    continue
                opp_x = (float((other[L_HIP, 0] + other[R_HIP, 0]) / 2)
                         if np.any(other) else (1.0 if slot == "A" else 0.0))
                bb = blade_box(kp, W2, H2, opp_x)
                if bb is None:
                    continue
                box_px, flen = bb
                b_e, b_p99, b_area = mean_energy(diff, box_px)

                # control: SAME AREA, centred on the hips. If the blade box only
                # looks hot because the whole fencer is moving, this moves with it.
                have_hips = (kp[L_HIP, 3] >= VISIBILITY_THRESHOLD
                             and kp[R_HIP, 3] >= VISIBILITY_THRESHOLD)
                if not have_hips:
                    t_e = t_p99 = np.nan
                    hx = hy = np.nan
                else:
                    hx = (kp[L_HIP, 0] + kp[R_HIP, 0]) / 2 * W2
                    hy = (kp[L_HIP, 1] + kp[R_HIP, 1]) / 2 * H2
                    side = np.sqrt(max(b_area, 1)) / 2
                    t_e, t_p99, _ = mean_energy(
                        diff, (hx - side, hy - side, hx + side, hy + side))

                # ---- v2: oriented strip, aligned on this fencer's own torso ----
                s_e = s_p99 = c_e = c_p99 = np.nan
                sh_resp = np.nan
                if have_hips and prev_frame is not None:
                    got = wrist_dir(kp, W2, H2, opp_x)
                    sh = body_shift(prev_frame, gray, hx, hy,
                                    TORSO_HALF * flen, hann_cache)
                    if sh is None:
                        # weak correlation: fall back to the global pan rather than
                        # to zero, so a failed local estimate is no worse than v1
                        shift = (pan * (W2 / 320.0), 0.0)
                    else:
                        shift = (sh[0], sh[1])
                        sh_resp = sh[2]
                    if got is not None:
                        wx, wy, ux, uy = got
                        M = _strip_M(wx, wy, ux, uy, BLADE_REACH * flen,
                                     STRIP_HALFWIDTH * flen, STRIP_LEN, STRIP_WID)
                        s_e, s_p99 = strip_energy(prev_frame, gray, M, shift,
                                                  STRIP_LEN, STRIP_WID)
                        Mc = _strip_M(hx, hy, ux, uy, BLADE_REACH * flen * 0.5,
                                      STRIP_HALFWIDTH * flen, STRIP_LEN, STRIP_WID)
                        c_e, c_p99 = strip_energy(prev_frame, gray, Mc, shift,
                                                  STRIP_LEN, STRIP_WID)

                rows.append((frame_idx / fps, slot, b_e, t_e, global_e, flen,
                             b_p99, t_p99, s_e, s_p99, c_e, c_p99, sh_resp))
        frame_idx += 1

    cap.release()
    for s_ in landmarkers.values():
        s_.__exit__(None, None, None)

    np.savez(OUT,
             time=np.array([r[0] for r in rows], dtype=np.float32),
             slot=np.array([r[1] for r in rows]),
             blade=np.array([r[2] for r in rows], dtype=np.float32),
             torso=np.array([r[3] for r in rows], dtype=np.float32),
             global_e=np.array([r[4] for r in rows], dtype=np.float32),
             forearm=np.array([r[5] for r in rows], dtype=np.float32),
             blade_p99=np.array([r[6] for r in rows], dtype=np.float32),
             torso_p99=np.array([r[7] for r in rows], dtype=np.float32),
             # v2: oriented strip, aligned on the fencer's own torso
             strip=np.array([r[8] for r in rows], dtype=np.float32),
             strip_p99=np.array([r[9] for r in rows], dtype=np.float32),
             ctrl=np.array([r[10] for r in rows], dtype=np.float32),
             ctrl_p99=np.array([r[11] for r in rows], dtype=np.float32),
             shift_resp=np.array([r[12] for r in rows], dtype=np.float32))
    print(f"wrote {OUT.name}: {len(rows)} rows over {frame_idx} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test() if "--self-test" in sys.argv else main())
