"""Emit only the halts a human still has to judge: the ones where both lamps fired."""
import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "scripts"))

import read_scoreboard as RS

LAB = PROJECT / "data" / "labels"
RAW = PROJECT / "data" / "raw_video"
REPLAY = 30.0


def header(stem):
    return [
        f"# Two-light halts in {stem}.mp4. ONE ROW PER HALT WHERE BOTH LAMPS FIRED,",
        "# whether that is two colours, or one colour and one white.",
        "# SKIP halts where BOTH lights are white (two off-targets): nothing can be",
        "# awarded whoever had priority, so the outcome carries no information.",
        "#   awarded    left | right | none",
        "#   off_side   the side whose light was OFF TARGET, if one was; blank if both",
        "#              were valid. Needed because `none` on a mixed halt means the",
        "#              OFF-TARGET fencer held priority, and which side that was cannot",
        "#              be recovered from `none` alone.",
        "# Single-lamp halts are omitted: the lamp already names the scorer.",
        "time\tawarded\toff_side",
    ]


def dedupe(halts, window=REPLAY):
    """Drop re-fires of the same lights within `window` -- the broadcast replay."""
    keep = []
    for h in halts:
        if any(0 < h["t"] - k["t"] < window and k["lights"] == h["lights"] for k in keep):
            continue
        keep.append(h)
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bout", required=True)
    ap.add_argument("--stride", type=float, default=0.1)
    ap.add_argument("--out")
    ap.add_argument("--template", action="store_true",
                    help="blank sheet, for a bout with no calibrated lamp box yet")
    a = ap.parse_args()

    out = Path(a.out) if a.out else LAB / f"bout{a.bout}_contested.tsv"
    calibrated = a.bout in RS.LAYOUT and "lamp" in RS.LAYOUT[a.bout]

    if a.template or not calibrated:
        out.write_text("\n".join(header(a.bout)) + "\n", encoding="utf-8")
        why = "asked for a template" if a.template else "no calibrated lamp box"
        print(f"bout {a.bout}: {why}; wrote a blank sheet -> {out}")
        return 0

    cache = LAB / f"{a.bout}_lamp.npz"
    t, ser = RS.lamp_series(RAW / f"{a.bout}.mp4", RS.LAYOUT[a.bout]["lamp"], a.stride, cache)
    halts = dedupe(RS.detect_halts(t, ser, RS.lamp_all_thresholds(ser)))
    contested = [h for h in halts if h["lights"] == "both"]

    print(f"bout {a.bout}: {len(halts)} halts, {len(halts) - len(contested)} settled by a "
          f"single lamp, {len(contested)} contested")
    out.write_text("\n".join(header(a.bout) + [f"{h['t']:.3f}\t\t" for h in contested]) + "\n",
                   encoding="utf-8")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
