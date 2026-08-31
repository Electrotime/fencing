"""Scoreboard panel for the demo: lamp states, referee call, model call."""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import exp_pooled as EP
import exp_touch_probe as PR
import read_scoreboard as RS

LAB = PROJECT / "data" / "labels"
HOLD = (-0.3, 2.5)          # how long a halt banner stays up, seconds either side
OFF = (70, 70, 70)
WHITE = (235, 235, 235)
BAR_H = 130


def _instant_states(series, thr):
    """Per-sample colour/white/off for each side, without the halt window."""
    out = {}
    for s in ("left", "right"):
        col = series[s]["red" if s == "left" else "green"]
        wht = series.get(f"{s}_white", series[s])["wmean"]
        st = np.where(col > thr["colour"][s], 2, np.where(wht > thr["white"][s], 1, 0))
        out[s] = st
    return out


def load(stem):
    """Lamp timeline plus one row per halt: referee priority and the model's call."""
    t, ser = RS.lamp_series("", RS.LAYOUT[stem]["lamp"], 0.1, LAB / f"{stem}_lamp.npz")
    thr = RS.lamp_all_thresholds(ser)
    rows = [(u, p, k) for u, p, k in EP.rows_for(stem) if p]
    calls = {}
    if rows:
        pr = np.load(LAB / PR.CACHE[stem])
        X, names, ok = PR.build([(stem, u, p) for u, p, _ in rows], [pr] * len(rows), 0.3)
        x = X[names.index(PR.PREREG)]
        z = (x - x.mean()) / (x.std() or 1.0)
        for (u, p, _), zz in zip([r for r, k in zip(rows, ok) if k], z):
            calls[round(u, 2)] = ("left" if zz > 0 else "right", p)
    return {"t": t, "state": _instant_states(ser, thr),
            "halts": [(u, p) for u, p, _ in rows], "calls": calls}


def draw(frame, data, now_s):
    """Panel at top-centre, clear of the broadcast's own scoreboard along the bottom."""
    H, W = frame.shape[:2]
    x0, x1, y1 = int(W * 0.31), int(W * 0.69), 150
    box = frame[0:y1, x0:x1].copy()
    cv2.rectangle(box, (0, 0), (x1 - x0, y1), (18, 18, 18), -1)
    frame[0:y1, x0:x1] = cv2.addWeighted(box, 0.88, frame[0:y1, x0:x1], 0.12, 0)
    cv2.rectangle(frame, (x0, 0), (x1 - 1, y1), (90, 90, 90), 2)

    i = int(np.searchsorted(data["t"], now_s))
    i = min(max(i, 0), len(data["t"]) - 1)
    for s, cx in (("left", x0 + int((x1 - x0) * 0.16)),
                  ("right", x0 + int((x1 - x0) * 0.84))):
        v = int(data["state"][s][i])
        col = OFF if v == 0 else (WHITE if v == 1 else
                                  ((60, 60, 235) if s == "left" else (60, 210, 60)))
        cv2.circle(frame, (cx, 52), 27, col, -1)
        cv2.circle(frame, (cx, 52), 27, (200, 200, 200), 2)
        cv2.putText(frame, s.upper(), (cx - 30, 96), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (205, 205, 205), 1, cv2.LINE_AA)

    live = [(u, p) for u, p in data["halts"] if u + HOLD[0] <= now_s <= u + HOLD[1]]
    if not live:
        cv2.putText(frame, "scoreboard reader", (x0 + 20, y1 - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (140, 140, 140), 1, cv2.LINE_AA)
        return frame
    u, ref = live[0]
    got = data["calls"].get(round(u, 2))
    cx = (x0 + x1) // 2
    cv2.putText(frame, "HALT", (cx - 38, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"referee: {ref}", (cx - 76, 72), cv2.FONT_HERSHEY_SIMPLEX,
                0.62, (235, 235, 235), 2, cv2.LINE_AA)
    if got:
        call, truth = got
        ok = call == truth
        cv2.putText(frame, f"model: {call}  {'OK' if ok else 'MISS'}", (cx - 82, 108),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (60, 210, 60) if ok else (70, 70, 245), 2, cv2.LINE_AA)
    return frame


def _self_test():
    d = load("6")
    assert len(d["t"]) > 100
    assert set(d["state"]) == {"left", "right"}
    assert d["halts"], "bout 6 should have priority halts"
    f = np.zeros((1080, 1920, 3), np.uint8)
    assert draw(f.copy(), d, d["halts"][0][0]).shape == f.shape
    hit = sum(1 for u, p in d["halts"] if d["calls"].get(round(u, 2), (None,))[0] == p)
    print(f"self-test ok: {len(d['halts'])} halts, model agrees on {hit}")


if __name__ == "__main__":
    _self_test()
