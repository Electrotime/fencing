"""Draw EVERY tall person detection on sampled frames, so the >2-people case can"""
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))
from ultralytics import YOLO

import demo_video as D
from src.person_detector import MIN_CONFIDENCE, PERSON_CLASS_ID

VIDEO = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT / "data" / "raw_video" / "1.mp4"
WANT = int(sys.argv[2]) if len(sys.argv) > 2 else 12
OUTDIR = PROJECT / "data" / "diagnostics" / f"{VIDEO.stem}_detections"
STEP = 7

GREEN, RED, YELLOW, WHITE = (0, 220, 0), (0, 0, 255), (0, 220, 220), (255, 255, 255)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(str(VIDEO))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    saved, idx = 0, 0
    while saved < WANT:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % STEP:
            idx += 1
            continue
        idx += 1
        r = model(frame, classes=[PERSON_CLASS_ID], conf=MIN_CONFIDENCE, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        tall = (xyxy[:, 3] - xyxy[:, 1]) >= D.MIN_BOX_H_FRAC * H
        xyxy, confs = xyxy[tall], confs[tall]
        if len(xyxy) <= 2:
            continue

        cx = (xyxy[:, 0] + xyxy[:, 2]) / 2 / W
        conf_pair = set(np.argsort(confs)[::-1][:2].tolist())
        order = np.argsort(cx)
        sep_pair = {int(order[0]), int(order[-1])}
        if conf_pair == sep_pair:
            continue          # nothing to learn from frames where they agree

        vis = frame.copy()
        rank = {int(k): i for i, k in enumerate(np.argsort(confs)[::-1])}
        for i, (b, c) in enumerate(zip(xyxy, confs)):
            x1, y1, x2, y2 = b.astype(int)
            in_conf, in_sep = i in conf_pair, i in sep_pair
            col = (GREEN if in_conf and in_sep else
                   RED if in_conf else YELLOW if in_sep else WHITE)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            tag = ("CONF+SEP" if in_conf and in_sep else
                   "CONF" if in_conf else "SEP" if in_sep else "-")
            cv2.putText(vis, f"#{rank[i]} {c:.2f} {tag}", (x1, max(14, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            cv2.putText(vis, f"x={cx[i]:.2f} h={(y2 - y1) / H:.2f}", (x1, min(H - 4, y2 + 16)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
        cv2.putText(vis, f"t={idx / fps:.2f}s  {len(xyxy)} tall people  "
                         f"RED=conf-only  YELLOW=sep-only  GREEN=both",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        out = OUTDIR / f"{idx:06d}.jpg"
        cv2.imwrite(str(out), vis)
        saved += 1
    cap.release()
    print(f"wrote {saved} annotated frames to {OUTDIR}")
    print("RED boxes are what the CURRENT rule keeps; YELLOW is what separation "
          "would keep instead.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
