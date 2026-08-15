"""Transcribe Aaron's bout 5 table -> two-track CSV. Anomalies FLAGGED, not fixed."""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "labels" / "bout5_intervals_2track.csv"

# see anomaly 1
DROP_DUPLICATE = "first" if "--keep-first-duplicate" not in sys.argv else "second"
# see anomaly 2
FIXED = {"04:04.067": "04:06.067"}

R = [
 ("00:04.350 - 00:05.371", "neutral", "", "neutral", ""),
 ("00:05.371 - 00:06.750", "advance", "", "advance", ""),
 ("00:06.750 - 00:10.100", "advance", "", "retreat", ""),
 ("00:10.100 - 00:10.500", "retreat", "", "advance", ""),
 ("00:10.500 - 00:10.967", "lunge", "arm ext", "advance", ""),
 ("00:11.800 - 00:13.400", "walk", "", "walk", ""),
 ("00:16.766 - 00:19.300", "neutral", "", "neutral", ""),
 ("00:19.300 - 00:19.867", "advance", "", "advance", ""),
 ("00:19.867 - 00:20.601", "lunge", "arm ext", "retreat", "parry"),
 ("00:20.601 - 00:21.101", "lunge", "arm ext", "neutral", "arm ext"),
 ("00:21.101 - 00:25.000", "walk", "", "walk", ""),
 ("00:32.450 - 00:35.883", "neutral", "", "neutral", ""),
 ("00:35.883 - 00:36.950", "advance", "", "advance", ""),
 ("00:36.950 - 00:40.720", "advance", "", "retreat", ""),
 ("00:40.720 - 00:41.320", "neutral", "", "advance", "arm ext"),
 ("00:41.320 - 00:46.383", "walk", "", "walk", ""),
 ("00:53.500 - 00:56.600", "neutral", "", "neutral", ""),
 ("00:56.600 - 00:57.801", "advance", "", "advance", ""),
 ("00:57.801 - 01:07.300", "retreat", "", "advance", ""),
 ("01:07.300 - 01:08.367", "neutral", "parry", "lunge", "arm ext"),
 ("01:08.367 - 01:09.034", "neutral", "arm ext", "retreat", "parry"),
 ("01:09.467 - 01:10.334", "retreat", "parry", "advance", "arm ext"),
 ("01:10.334 - 01:12.499", "walk", "", "walk", ""),
 ("01:19.750 - 01:23.233", "walk", "", "walk", ""),
 ("01:25.883 - 01:26.483", "advance", "", "advance", ""),
 ("01:26.483 - 01:27.050", "lunge", "arm ext", "lunge", "arm ext"),
 ("01:27.050 - 01:28.816", "walk", "", "walk", ""),
 ("01:55.900 - 01:58.816", "walk", "", "walk", ""),
 ("02:04.850 - 02:10.500", "neutral", "", "neutral", ""),
 ("02:10.500 - 02:11.667", "advance", "", "advance", ""),
 ("02:11.667 - 02:17.250", "retreat", "", "advance", ""),
 ("02:17.250 - 02:18.017", "retreat", "", "lunge", "arm ext"),
 ("02:18.017 - 02:19.018", "advance", "", "retreat", ""),
 ("02:19.018 - 02:19.985", "advance", "", "retreat", "arm ext"),
 ("02:19.985 - 02:22.450", "advance", "", "retreat", "arm ext"),
 ("02:22.450 - 02:23.317", "neutral", "", "neutral", "arm ext"),
 ("02:23.317 - 02:29.566", "retreat", "", "advance", ""),
 ("02:29.566 - 02:30.065", "lunge", "arm ext", "neutral", ""),
 ("02:30.065 - 02:31.750", "walk", "", "walk", ""),
 ("02:40.550 - 02:43.850", "neutral", "", "neutral", ""),
 ("02:43.850 - 02:45.151", "advance", "", "advance", ""),
 ("02:45.151 - 02:46.451", "retreat", "", "advance", ""),
 ("02:46.451 - 02:47.618", "retreat", "parry", "advance", "arm ext"),
 ("02:47.618 - 02:49.316", "walk", "", "walk", ""),
 ("03:00.450 - 03:01.517", "advance", "", "advance", ""),
 ("03:01.517 - 03:03.700", "retreat", "", "advance", ""),
 ("03:03.700 - 03:04.767", "neutral", "arm ext", "advance", "arm ext"),
 ("03:04.767 - 03:06.183", "advance", "", "retreat", ""),
 ("03:06.183 - 03:06.783", "advance", "", "advance", "arm ext"),
 ("03:06.783 - 03:09.150", "walk", "", "walk", ""),
 # ---- duplicate block, copy 1 (see anomaly 1) ----
 ("03:21.249 - 03:23.700", "neutral", "", "neutral", "", "dup1"),
 ("03:23.700 - 03:24.767", "advance", "", "advance", "", "dup1"),
 ("03:24.767 - 03:27.169", "advance", "", "retreat", "", "dup1"),
 ("03:27.169 - 03:27.869", "advance", "", "advance", "arm ext", "dup1"),
 ("03:27.869 - 03:30.100", "walk", "", "walk", "", "dup1"),
 # ---- duplicate block, copy 2 ----
 ("03:21.200 - 03:23.700", "neutral", "", "neutral", "", "dup2"),
 ("03:23.700 - 03:24.767", "advance", "", "advance", "", "dup2"),
 ("03:24.767 - 03:27.083", "advance", "", "retreat", "", "dup2"),
 ("03:27.083 - 03:27.950", "advance", "", "advance", "arm ext", "dup2"),
 ("03:27.950 - 03:30.100", "walk", "", "walk", "", "dup2"),
 ("03:39.600 - 03:41.383", "neutral", "", "neutral", ""),
 ("03:41.383 - 03:42.117", "advance", "", "advance", ""),
 ("03:42.117 - 03:44.233", "advance", "", "retreat", ""),
 ("03:44.233 - 03:44.666", "lunge", "arm ext", "neutral", ""),
 ("03:44.666 - 03:49.649", "walk", "", "walk", ""),
 ("03:58.300 - 04:04.199", "neutral", "", "neutral", ""),
 ("04:04.199 - 04:05.066", "advance", "", "advance", ""),
 ("04:05.066 - 04:06.067", "advance", "", "advance", "arm ext"),
 ("04:04.067 - 04:10.183", "walk", "", "walk", ""),          # anomaly 2
 ("04:21.600 - 04:22.900", "neutral", "", "neutral", ""),
 ("04:22.900 - 04:23.734", "advance", "", "advance", ""),
 ("04:25.168 - 04:26.068", "advance", "arm ext", "advance", "arm ext"),
 ("04:26.068 - 04:27.600", "walk", "", "walk", ""),
 ("04:37.100 - 04:39.400", "neutral", "", "neutral", ""),
 ("04:39.400 - 04:40.434", "advance", "", "advance", ""),
 ("04:40.434 - 04:42.751", "retreat", "arm ext", "advance", ""),
 ("04:42.751 - 04:43.785", "advance", "arm ext", "advance", "arm ext"),
 ("04:43.785 - 04:48.116", "walk", "", "walk", ""),
 ("04:51.450 - 04:53.349", "neutral", "", "neutral", ""),
 ("04:53.349 - 04:54.616", "advance", "", "advance", ""),
 ("04:54.616 - 04:56.333", "retreat", "", "advance", ""),
 ("04:56.333 - 04:58.435", "advance", "", "retreat", ""),
 ("04:58.435 - 04:59.402", "lunge", "arm ext", "retreat", "parry"),
 ("05:00.200 - 05:04.800", "retreat", "arm ext", "advance", ""),
 ("05:06.134 - 05:07.116", "retreat", "arm ext", "advance", ""),
 ("05:09.800 - 05:10.600", "retreat", "parry", "lunge", "arm ext"),
 ("05:11.367 - 05:15.150", "advance", "", "retreat", ""),
 ("05:15.150 - 05:16.117", "advance", "parry", "advance", "arm ext"),
 ("05:16.951 - 05:18.833", "walk", "", "walk", ""),
 ("05:27.466 - 05:33.200", "walk", "", "walk", ""),
 ("05:33.200 - 05:34.850", "neutral", "", "neutral", ""),
 ("05:34.850 - 05:35.917", "advance", "", "advance", ""),
 ("05:35.917 - 05:37.150", "advance", "", "retreat", ""),
 ("05:37.150 - 05:37.917", "lunge", "arm ext", "retreat", "parry"),
 ("05:37.917 - 05:45.600", "retreat", "", "advance", ""),
 ("05:45.600 - 05:46.267", "advance", "arm ext", "advance", "arm ext"),
 ("05:46.267 - 05:48.300", "walk", "", "walk", ""),
 ("05:57.000 - 06:02.050", "walk", "", "walk", ""),
 ("06:15.550 - 06:21.550", "neutral", "", "neutral", ""),
 ("06:21.550 - 06:22.784", "advance", "", "advance", ""),
 ("06:22.784 - 06:23.017", "advance", "", "retreat", ""),     # anomaly 3: 0.233 s
 ("06:23.017 - 06:23.884", "lunge", "arm ext", "retreat", ""),
 ("06:23.884 - 06:28.700", "retreat", "", "advance", ""),
 ("06:28.700 - 06:29.223", "retreat", "arm ext", "neutral", ""),
 ("06:29.223 - 06:30.950", "walk", "", "walk", ""),
 ("06:39.800 - 06:41.300", "neutral", "", "neutral", ""),
 ("06:41.300 - 06:42.367", "advance", "", "advance", ""),
 ("06:42.367 - 06:45.200", "retreat", "", "advance", ""),
 ("06:45.200 - 06:45.900", "retreat", "parry", "advance", "arm ext"),
 ("06:45.900 - 06:48.433", "walk", "", "walk", ""),
 ("06:51.833 - 06:54.466", "neutral", "", "neutral", ""),
 ("06:54.466 - 06:55.166", "advance", "", "advance", ""),
 ("06:55.166 - 06:55.699", "advance", "arm ext", "advance", "arm ext"),
 ("06:55.699 - 07:00.250", "walk", "", "walk", ""),
 ("07:08.300 - 07:11.100", "neutral", "", "neutral", ""),
 ("07:11.100 - 07:12.034", "advance", "", "advance", ""),
 ("07:12.034 - 07:12.968", "lunge", "arm ext", "neutral", ""),
 ("07:21.083 - 07:23.183", "neutral", "", "neutral", ""),
 ("07:23.183 - 07:24.350", "advance", "", "advance", ""),
 ("07:24.350 - 07:27.500", "retreat", "", "advance", ""),
 ("07:27.500 - 07:28.567", "retreat", "parry", "advance", "arm ext"),
 ("07:28.567 - 07:30.566", "walk", "", "walk", ""),
 ("07:38.250 - 07:41.150", "neutral", "", "neutral", ""),
 ("07:41.150 - 07:42.117", "advance", "", "advance", ""),
 ("07:42.117 - 07:42.717", "advance", "arm ext", "advance", ""),
 ("07:42.717 - 07:43.417", "neutral", "arm ext", "retreat", "arm ext"),
 ("07:43.417 - 07:45.383", "walk", "", "walk", ""),
 ("07:49.750 - 07:53.850", "neutral", "", "walk", ""),
 ("07:53.850 - 07:55.800", "neutral", "", "neutral", ""),
 ("07:55.800 - 07:56.033", "advance", "", "advance", ""),     # anomaly 3: 0.233 s
 ("07:56.033 - 07:56.633", "lunge", "arm ext", "retreat", "parry"),
 ("07:56.633 - 08:04.733", "retreat", "", "advance", ""),
 ("08:04.733 - 08:05.633", "lunge", "arm ext", "lunge", "arm ext"),
 ("08:05.633 - 08:09.350", "walk", "", "walk", ""),
 ("08:23.350 - 08:28.350", "walk", "", "walk", ""),
 ("08:29.900 - 08:38.100", "walk", "", "walk", ""),
 ("08:38.100 - 08:42.550", "neutral", "", "neutral", ""),
 ("08:42.550 - 08:43.489", "advance", "", "advance", ""),
 ("08:43.489 - 08:44.223", "lunge", "arm ext", "retreat", "arm ext"),
 ("08:44.223 - 08:45.833", "walk", "", "walk", ""),
 ("08:53.800 - 08:58.100", "neutral", "", "neutral", ""),
 ("08:58.100 - 08:59.534", "advance", "", "advance", ""),
 ("08:59.534 - 09:02.300", "retreat", "", "advance", ""),
 ("09:02.300 - 09:03.234", "advance", "parry", "advance", "arm ext"),
 ("09:03.234 - 09:06.200", "walk", "", "walk", ""),
 ("09:10.200 - 09:13.500", "neutral", "", "neutral", ""),
 ("09:13.500 - 09:14.434", "advance", "", "advance", ""),
 ("09:14.434 - 09:15.134", "advance", "", "advance", ""),
 ("09:15.134 - 09:17.050", "walk", "", "walk", ""),
]

FOOTWORK = {"walk": "walking", "neutral": "neutral", "advance": "advance",
            "retreat": "retreat", "lunge": "lunge"}
BLADE = {"": "none", "arm ext": "extension", "parry": "parry"}

HEADER = """\
# Ground-truth interval labels for data/raw_video/5.mp4. Aaron, 2026-08-12.
# NEW VENUE: 1906x1080 @ 30.000 fps, vs bouts 1-4 at 1920x1080 @ 29.97. This is the
# first footage that tests the hand-calibrated framing constants rather than the model.
#
# TWO-TRACK schema (footwork + blade). 9.6 min source, densely labelled: every phrase
# from en-garde through the touch, with the between-phrase `walk` marked too.
#
# `arm ext` -> `extension`, `walk` -> `walking`. `extension` is not one of the six
# model classes, so evaluate_labels.py collapses those rows to the FOOTWORK label
# rather than dropping them as unscorable.
#
# Transcribed by scripts/transcribe_bout5.py -- see its docstring for the three
# source anomalies (one duplicated phrase, one backwards timestamp, two 0.233 s
# intervals) and how each was resolved.
fencer,start,end,footwork,blade
"""


def secs(ts: str) -> float:
    ts = FIXED.get(ts.strip(), ts.strip())
    m, s = ts.split(":")
    return int(m) * 60 + float(s)


def main() -> int:
    rows, dropped = [], 0
    for r in R:
        span, af, ab, bf, bb = r[0], r[1], r[2], r[3], r[4]
        tag = r[5] if len(r) > 5 else None
        if tag == ("dup1" if DROP_DUPLICATE == "first" else "dup2"):
            dropped += 1
            continue
        a, b = (x.strip() for x in span.split(" - "))
        s, e = secs(a), secs(b)
        rows.append(("left", s, e, FOOTWORK[af], BLADE[ab]))
        rows.append(("right", s, e, FOOTWORK[bf], BLADE[bb]))

    # overlap check per fencer -- truth_at() returns the FIRST match, so an overlap
    # silently mis-scores every window inside it
    problems = []
    for who in ("left", "right"):
        mine = sorted([r for r in rows if r[0] == who], key=lambda r: r[1])
        for x, y in zip(mine, mine[1:]):
            if y[1] < x[2] - 1e-9:
                problems.append(f"{who}: {x[1]:.3f}-{x[2]:.3f} overlaps {y[1]:.3f}-{y[2]:.3f}")
        for r in mine:
            if r[2] <= r[1]:
                problems.append(f"{who}: non-positive interval {r[1]:.3f}-{r[2]:.3f}")

    OUT.write_text(HEADER + "".join(
        f"{w},{s:.3f},{e:.3f},{f},{b}\n" for w, s, e, f, b in
        sorted(rows, key=lambda r: (r[1], r[0]))), encoding="utf-8")

    n_par = sum(1 for r in rows if r[4] == "parry")
    n_ext = sum(1 for r in rows if r[4] == "extension")
    dur = sum(e - s for _, s, e, _, _ in rows)
    print(f"wrote {OUT.name}: {len(rows)} rows ({len(rows)//2} intervals x 2 fencers)")
    print(f"  dropped {dropped} rows as the duplicated 03:21-03:30 phrase "
          f"(kept the {'second' if DROP_DUPLICATE=='first' else 'first'} copy)")
    print(f"  labelled fencer-time {dur:.0f} s over a 578 s source "
          f"({dur / 2 / 578:.0%} coverage per fencer)")
    print(f"  parries {n_par}, arm extensions {n_ext}")
    short = [(s, e) for _, s, e, _, _ in rows if e - s < 0.30]
    if short:
        print(f"  {len(short)} intervals under 0.30 s (shorter than anything in bouts "
              f"1-4): {', '.join(f'{s:.3f}-{e:.3f}' for s, e in sorted(set(short)))}")
    if problems:
        print("\n!! OVERLAPS / BAD INTERVALS -- these mis-score silently:")
        for p in problems:
            print(f"   {p}")
        return 1
    print("\nno overlaps. Validate with: py -3 scripts/check_labels.py "
          f"data/labels/{OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
