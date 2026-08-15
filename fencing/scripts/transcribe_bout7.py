"""Aaron's bout 7 table -> data/labels/bout7_intervals_2track.csv."""
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "labels" / "bout7_intervals_2track.csv"

FOOTWORK = {"neutral", "advance", "retreat", "lunge", "walk"}
BLADE = {"parry", "arm ext"}
FW_MAP = {"walk": "walking"}
BL_MAP = {"arm ext": "extension"}
# footwork and blade vocabularies are disjoint, so tokens can be assigned by kind
# rather than by column -- immune to the ragged tabs in the pasted table
TOKEN = re.compile(r"\b(arm ext|neutral|advance|retreat|lunge|walk|parry)\b")
TIMES = re.compile(r"(\d+):(\d+\.\d+)\s*-\s*(\d+):(\d+\.\d+)")

TABLE = """
00:00.000 - 00:01.956 	neutral		neutral
00:01.956 - 00:02.790	advance		advance
00:03.390 - 00:04.457	retreat	parry	lunge	arm ext
00:04.870 - 00:05.437	neutral	parry	lunge	arm ext
00:05.723 - 00:07.024	retreat		advance
00:07.391 - 00:08.024	advance	parry	advance
00:08.824 - 00:17.766	advance		retreat
00:17.766 - 00:18.166	lunge	arm ext	retreat
00:18.166 - 00:24.080	walk		walk
00:34.370 - 00:35.770	neutral		neutral
00:36.603 -00:36.936	retreat		advance
00:36.936 - 00:37.770	retreat	parry	lunge	arm ext
00:37.770 - 00:40.539	advance		retreat	arm ext
00:40.539 - 00:42.707	advance		retreat
00:42.707 - 00:43.474	lunge	arm ext	retreat	parry
00:43.474 - 00:46.597	walk		walk
00:50.783 - 00:54.023	neutral		neutral
00:54.656 - 00:55.523	advance	parry	lunge
00:55.523 - 00:58.626	walk		walk
01:04.540 - 01:06.220	neutral		neutral
01:07.943 - 01:08.877	retreat	arm ext	advance
01:08.877 - 01:09.043	lunge	arm ext	advance
01:09.043 - 01:09.676	retreat	parry	lunge	arm ext
01:09.676 - 01:12.906 	walk		walk
01:18.470 - 01:19.236	neutral		neutral
01:19.236 - 01:20.036	advance		advance
01:20.036 - 01:20.803	retreat	parry	lunge	arm ext
01:20.803 - 01:22.563	walk		walk
01:27.173 - 01:29.909	walk		neutral
01:31.577 - 01:32.144	retreat	parry	lunge	arm ext
01:32.644 - 01:35.583	walk		walk
01:42.270 - 01:42.856	advance		retreat
01:42.856 - 01:45.158	advance		retreat	arm ext
01:45.758 - 01:46.659	advance		retreat	arm ext
01:46.859 - 01:47.726	lunge	arm ext	retreat	arm ext
01:47.726 - 01:49.790	walk		walk
02:00.229 - 02:01.530	retreat	parry	lunge	arm ext
02:01.795 - 02:02.295	retreat	parry	lunge	arm ext
02:02.295 - 02:02.762	neutral	arm ext	neutral	parry
02:02.762 - 02:03.529	retreat	arm ext	lunge	arm ext
02:03.529 - 02:05.336	walk		walk
02:12.990 - 02:14.491	neutral		neutral
02:15.525 - 02:17.593	retreat		advance
02:17.593 - 02:18.594	lunge	arm ext	advance
02:18.594 - 02:20.070	walk		walk
02:45.556 - 02:46.490	retreat	parry	lunge	arm ext
02:46.490 - 02:47.820	walk		walk
02:54.090 - 03:00.210	neutral		neutral
03:00.576 - 03:01.410	lunge	arm ext	advance	parry
03:02.044 - 03:02.878	advance	arm ext	retreat	parry
03:02.878 - 03:05.116	walk		walk
03:12.570 - 03:31.040	walk		neutral
03:37.490 - 03:38.224	lunge	arm ext	lunge	arm ext
03:38.224 - 03:42.000	walk		walk
03:49.390 - 03:51.316	neutral		neutral
03:52.150 - 04:02.426	advance		retreat
04:03.794 - 04:05.028	lunge	arm ext	lunge
04:05.028 - 04:08.500	walk		walk
04:29.220 - 04:30.020	advance		advance
04:30.020 - 04:33.590	walk		walk
04:42.057 - 04:46.229	retreat		advance
04:46.229 - 04:46.829	advance	arm ext	advance
04:48.140 - 04:49.574	advance		retreat	arm ext
04:53.805 - 04:57.396	retreat		advance
04:57.396 - 04:58.463	retreat		lunge	arm ext
04:59.597 - 05:04.406	advance		retreat
05:04.406 - 05:05.573	lunge	arm ext	retreat	parry
05:05.573 - 05:08.910	walk		walk
05:16.333 - 05:18.080	neutral		neutral
05:18.580 - 05:19.380	lunge	arm ext	retreat	parry
05:20.180 - 05:22.953	retreat		advance
05:22.953 - 05:23.987	retreat	parry	lunge	arm ext
05:23.987 - 05:26.280	advance		retreat
05:26.280 - 05:26.680	neutral	parry	lunge	arm ext
05:26.680 - 05:27.547	lunge	arm ext	retreat	parry
05:28.214 - 05:32.943	retreat		advance
05:32.943 - 05:33.543	retreat	parry	neutral	arm ext
05:33.776 - 05:34.810	lunge	arm ext	retreat	parry
05:34.810 - 05:36.478	advance		retreat
05:36.478 - 05:37.411	advance	arm ext	advance	arm ext
05:37.411 - 05:39.040	walk		walk
06:14.150 - 06:17.826	neutral		neutral
06:18.459 - 06:19.259	neutral	parry	lunge	arm ext
06:19.259 - 06:19.859	neutral	arm ext	retreat	parry
06:20.693 - 06:21.427	retreat	arm ext	lunge	arm ext
06:21.427 - 06:25.520	walk		walk
06:31.619 - 06:33.216	neutral		neutral
06:33.749 -06:34.584	retreat	parry	lunge	arm ext
06:34.584 - 06:36.218	advance		retreat	arm ext
06:36.218 - 06:37.052	lunge	arm ext	retreat	arm ext
06:37.052 - 06:40.929	walk		walk
06:46.910 - 06:55.323	walk		walk
06:58.063 - 07:02.060	neutral		neutral
07:02.822 - 07:03.689	lunge		lunge
07:03.689 - 07:06.183	walk		walk
07:13.560 - 07:14.294	lunge		lunge	arm ext
07:14.294 - 07:17.076	walk		walk
07:50.260 - 07:57.330	neutral		neutral
07:58.443 - 08:03.386	advance		retreat
08:03.386 - 08:03.819	lunge	arm ext	retreat	arm ext
08:03.819 - 08:06.706	walk		walk
08:18.220 -08:20.222	advance		retreat	arm ext
08:20.222 - 08:23.956	advance		retreat
08:23.956 - 08:24.589	advance	parry	lunge	arm ext
08:39.470 - 08:41.010	neutral		neutral
08:43.379 - 08:46.263	retreat		advance
08:46.263 - 08:47.330	retreat	parry	lunge	arm ext
08:48.780 - 08:53.190	advance		retreat	arm ext
08:54.090 - 08:54.757	lunge	arm ext	retreat	parry
08:54.757 - 08:58.433	walk		walk
09:03.376 - 09:04.210	lunge	arm ext	lunge	arm ext
09:04.210 - 09:07.470	walk		walk
09:41.350 - 09:45.550	walk		walk
09:50.963 - 09:53.565	neutral	parry	lunge
10:03.540 - 10:04.741	retreat	parry	lunge	arm ext
10:04.741 - 10:08.273 	advance		retreat
10:08.273 - 10:09.307	lunge	arm ext	retreat
10:26.719 - 10:27.419	lunge	arm ext	advance	parry
10:35.950 - 10:37.718	neutral		neutral
10:39.252 - 10:42.460	advance		retreat	arm ext
10:43.527 - 10:44.528	neutral	arm ext	lunge	arm ext
10:44.528 - 10:48.672	walk		walk
10:53.287 - 10:56.250	neutral		neutral
10:57.017 - 11:02.660	retreat		advance
11:02.660 - 11:08.010	neutral		neutral
11:08.010 - 11:10.603	walk		walk
12:09.610 - 12:16.190	walk		walk
12:21.580 - 12:24.650	neutral		neutral
12:25.751 - 12:29.788	advance		retreat	arm ext
12:29.788 - 12:32.323	advance		retreat
12:32.323 - 12:33.191	lunge	arm ext	neutral	parry
12:34.192 -12:36.416	walk		walk
13:21.570 - 13:28.920	walk		neutral
13:28.920 -13:32.350	neutral		neutral
13:33.617 - 13:39.149	advance		retreat
13:39.149 - 13:40.183	lunge	arm ext	retreat	parry
13:40.183 - 13:44.573	retreat		advance
13:44.573 - 13:45.540	lunge	arm ext	advance
13:45.540 - 13:51.530	walk		walk
14:03.640 - 14:07.563	retreat		advance
14:07.563 - 14:08.530	retreat	parry	lunge	arm ext
14:14.093 - 14:15.394	neutral	parry	lunge	arm ext
14:15.394 - 14:23.240	walk		walk
14:25.177 - 14:25.844	neutral		lunge
14:35.364 - 14:36.231	advance	parry	lunge	arm ext
14:36.231 - 14:42.187	advance		retreat
14:42.187 - 14:42.921	advance	parry	neutral	arm ext
14:44.422 - 14:46.936	retreat		advance
14:46.936 - 14:47.870	retreat	parry	advance	arm ext
14:47.870 - 14:49.506	advance		retreat
14:49.506 - 14:50.540	lunge	arm ext	retreat	arm ext
14:50.540 - 14:52.603	walk		walk
15:01.233 - 15:04.640	retreat		advance
15:04.640 - 15:05.607	retreat	parry	lunge
15:12.310 -15:17.000	neutral		neutral
15:17.703 -15:18.604	lunge	arm ext	advance
15:18.604 - 15:28.060	walk		walk
15:42.900- 15:44.650	neutral		neutral
15:45.017 - 15:46.017	lunge		lunge	arm ext
15:46.017 - 15:50.600	walk		walk
15:57.460 - 16:04.016	walk		walk
16:18.216 - 16:21.453	retreat		advance
16:21.453 - 16:22.353	retreat	parry	lunge	arm ext
16:38.830 -16:45.340 	walk		walk
"""


def parse():
    rows, problems = [], []
    for lineno, line in enumerate(TABLE.strip().splitlines(), 1):
        if not line.strip():
            continue
        m = TIMES.search(line)
        if not m:
            problems.append((lineno, line.strip(), "no time range"))
            continue
        start = int(m.group(1)) * 60 + float(m.group(2))
        end = int(m.group(3)) * 60 + float(m.group(4))
        toks = TOKEN.findall(line[m.end():])
        fw = [t for t in toks if t in FOOTWORK]
        if len(fw) != 2:
            problems.append((lineno, line.strip(), f"{len(fw)} footwork tokens, expected 2"))
            continue
        # walk the token stream: footwork claims a fencer, a blade token attaches
        # to whichever fencer was named last
        per = [["", ""], ["", ""]]
        who = -1
        for t in toks:
            if t in FOOTWORK:
                who += 1
                per[who][0] = FW_MAP.get(t, t)
            else:
                per[who][1] = BL_MAP.get(t, t)
        if end <= start:
            problems.append((lineno, line.strip(), f"end {end:.3f} <= start {start:.3f}"))
            continue
        for side, (f, b) in zip(("left", "right"), per):
            rows.append((side, start, end, f, b or "none"))
    return rows, problems


def main() -> int:
    rows, problems = parse()
    for lineno, text, why in problems:
        print(f"  !! line {lineno}: {why}\n     {text}")
    if problems:
        print(f"\n{len(problems)} unparsed rows -- refusing to write a partial file")
        return 1

    for side in ("left", "right"):
        s = sorted([r for r in rows if r[0] == side], key=lambda r: r[1])
        for a, b in zip(s, s[1:]):
            if b[1] < a[2] - 1e-6:
                print(f"  !! {side} overlap: [{a[1]:.3f},{a[2]:.3f}] and [{b[1]:.3f},{b[2]:.3f}]")
                return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Ground-truth interval labels for data/raw_video/7.mp4. Aaron, 2026-08-15.\n")
        f.write("# THIRD VENUE: 1280x718 @ 60 fps, vs 1920x1080 @ ~30 for bouts 1-5.\n")
        f.write("# 60 fps is a pipeline hazard: WINDOW_LONG=60 frames is 2 s at 30 fps but\n")
        f.write("# 1 s here. Resample to 30 fps before extraction.\n")
        f.write("# Two-track schema; `walk` -> walking, `arm ext` -> extension.\n")
        f.write("fencer,start,end,footwork,blade\n")
        for side, s, e, fw, bl in sorted(rows, key=lambda r: (r[1], r[0])):
            f.write(f"{side},{s:.3f},{e:.3f},{fw},{bl}\n")

    from collections import Counter
    fw = Counter(r[3] for r in rows)
    bl = Counter(r[4] for r in rows)
    secs = sum(r[2] - r[1] for r in rows) / 2
    print(f"wrote {OUT.name}: {len(rows)} rows ({len(rows)//2} intervals x 2 fencers), "
          f"{secs:.0f}s labelled")
    print("  footwork:", dict(fw.most_common()))
    print("  blade:   ", dict(bl.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
