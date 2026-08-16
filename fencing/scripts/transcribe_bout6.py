"""Aaron's bout 6 table -> data/labels/bout6_intervals_2track.csv."""
import re
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "data" / "labels" / "bout6_intervals_2track.csv"

FOOTWORK = {"neutral", "advance", "retreat", "lunge", "walk"}
BLADE = {"parry", "arm ext"}
FW_MAP = {"walk": "walking"}
BL_MAP = {"arm ext": "extension"}
TOKEN = re.compile(r"\b(arm ext|neutral|advance|retreat|lunge|walk|parry)\b")
TIMES = re.compile(r"(\d+):(\d+\.\d+)\s*-\s*(\d+):(\d+\.\d+)")

TABLE = """
00:00.000 - 00:10.233	walk		walk
00:22.701 - 00:26.271	advance		retreat
00:26.271 - 00:27.866	walk		walk
00:29.801 - 00:30.602	retreat	parry	lunge 	arm ext
00:30.602 - 00:35.266	advance		retreat
00:35.266 - 00:44.833	retreat		advance
00:44.833 - 00:46.134	retreat	parry	lunge 	arm ext
00:46.934 - 00:47.935	advance	parry	advance	arm ext
00:47.935 - 00:56.200	advance		retreat
00:56.200 - 00:56.833	advance	arm ext	retreat	parry
00:56.833 - 00:57.367	retreat	parry	lunge 	arm ext
00:58.068 - 00:58.935	advance	parry	advance	arm ext
00:58.935 - 00:59.769	lunge	arm ext	retreat
01:02.437 - 01:03.638	lunge	arm ext	advance
01:03.638 - 01:06.200	walk		walk
01:10.166 - 01:11.333	advance		advance
01:11.666 - 01:15.670	retreat		advance
01:15.670 -01:16.937	advance	arm ext	neutral
01:16.937 - 01:20.733	walk		walk
01:23.000 - 01:24.900	neutral		neutral
01:24.900 - 01:25.967	advance		advance
01:25.967 - 01:26.801	lunge	arm ext	advance
01:26.801 - 01:33.366	walk		walk
01:35.800 - 01:36.834	advance		advance
01:36.834 - 01:37.634	retreat	parry	lunge 	arm ext
01:37.634 - 01:41.433	advance		retreat
01:41.433 - 01:42.033	lunge	arm ext	advance	parry
01:42.967 - 01:46.133	walk		walk
01:53.300 - 01:55.035	neutral		neutral
01:55.035 - 01:56.236	advance		advance
01:56.236 - 01:58.037	retreat		advance
01:58.037 - 01:58.871	retreat	parry	lunge 	arm ext
01:58.871 - 02:04.500	walk		walk
02:04.500 - 02:05.934	neutral		neutral
02:05.934 - 02:06.368	advance		advance
02:06.368 - 02:07.035	retreat	parry	lunge 	arm ext
02:07.035 - 02:09.800	advance		retreat
02:09.800 - 02:10.400	advance	arm ext	retreat	arm ext
02:22.500 - 02:28.200	walk		walk
02:28.200 - 02:30.368	neutral		neutral
02:30.368 - 02:31.601	advance		advance
02:31.601 - 02:32.736	lunge	arm ext	advance
02:32.736 - 02:37.480	walk		walk
02:45.934 - 02:46.668	lunge	arm ext	advance
02:46.668 - 02:50.500	walk		walk
02:58.500 - 03:00.268	neutral		neutral
03:00.268 - 03:01.269	advance		advance
03:01.269 - 03:04.772	advance		retreat
03:04.772 - 03:10.956	walk		walk
03:10.956 - 03:11.756	advance		advance
03:11.756 - 03:12.156	lunge	arm ext	advance
03:12.156 - 03:12.556	neutral	parry	lunge 	arm ext
03:18.900 - 03:21.766	neutral		neutral
03:21.766 - 03:23.000 	advance		advance
03:23.000 - 03:24.234	retreat	arm ext	advance	arm ext
03:24.234 - 03:29.099	walk		walk
03:34.633 - 03:36.900	neutral		neutral
03:36.900 - 03:38.134	advance		advance
03:38.667 -03:39.901	lunge	arm ext	retreat
03:41.669 - 03:42.269	lunge	arm ext	retreat
03:43.403 - 03:45.003	retreat		advance
03:45.003 - 03:45.636	retreat	parry	lunge 	arm ext
03:45.636 - 03:46.136	advance	arm ext	neutral 	parry
03:46.369 - 03:49.866 	walk		walk
03:56.700 - 03:57.967	advance		advance
03:59.635 - 04:03.977	walk		walk
04:03.977 - 04:06.480	neutral		neutral
04:06.480 - 04:07.447	advance		advance
04:07.447 - 04:08.115	retreat		advance
04:08.115 - 04:10.250	advance		retreat
04:10.250 - 04:10.984	lunge	arm ext	neutral 	parry
04:13.700 - 04:14.767	neutral		neutral
04:14.767 - 04:15.935	advance		advance
04:15.935 - 04:17.803	retreat		advance
04:17.803 - 04:18.704	retreat	parry	lunge 	arm ext
04:25.367 - 04:25.833	retreat	parry	lunge 	arm ext
04:25.833 - 04:30.499	advance		retreat
04:30.499 - 04:30.999	advance	parry	advance	arm ext
04:30.999 - 04:32.033	advance	arm ext	retreat
04:37.400 - 04:38.200	retreat	parry	advance	arm ext
04:38.200 - 04:39.600	advance		retreat
04:39.600 - 04:40.634	lunge	arm ext	retreat
04:40.634 - 04:44.167	retreat		advance
04:46.100 - 04:47.933	advance		advance
04:47.933 - 04:48.934	lunge	arm ext	retreat
04:50.835 - 04:51.602	lunge	arm ext	advance
05:08.233 - 05:09.435	advance		advance
05:09.435 - 05:10.902	retreat		advance
05:10.902 - 05:11.602	advance		advance
05:11.602 - 05:16.366	walk		walk
05:22.500 - 05:23.367	advance		advance
05:23.367 - 05:28.899	advance		retreat
05:28.899 - 05:31.533	retreat		advance
05:31.533 - 05:32.133	retreat	parry	lunge 	arm ext
05:32.133 - 05:32.700	neutral	arm ext	neutral 	parry
05:32.700 - 05:35.669	retreat		advance
05:35.669 - 05:36.469	advance		lunge 	arm ext
05:46.399 - 05:47.834	advance		advance
05:48.034 - 05:48.734	retreat	parry	lunge 	arm ext
05:48.734 - 05:53.366	advance		retreat
05:53.366 - 05:54.668	advance	parry	advance
05:54.668 - 05:55.702	retreat	arm ext	advance
05:59.667 - 06:00.534	advance		advance
06:00.534 - 06:03.700	advance		retreat
06:03.700 - 06:04.500	lunge	arm ext	retreat	parry
06:05.567 - 06:10.833	retreat		advance
06:11.366 - 06:11.934	neutral	arm ext	advance	arm ext
06:15.567 - 06:16.635	advance		advance
06:16.635 - 06:22.333	retreat		advance
06:22.333 - 06:23.301	advance	parry	advance	arm ext
06:28.401 - 06:29.135	lunge	arm ext	advance
06:42.600 - 06:44.034	neutral		neutral
06:44.034 - 06:45.502	advance		advance
06:45.502 - 06:49.166	advance		retreat
06:49.166 - 06:50.067	lunge	arm ext	retreat
06:50.834 - 06:52.102	lunge	arm ext	advance
06:52.102 - 06:54.800	walk		walk
07:03.600 - 07:04.667	advance		advance
07:05.034 - 07:05.735	retreat	parry	lunge 	arm ext
07:05.735 - 07:10.567	advance		retreat
07:18.100 - 07:18.767	advance		advance
07:18.767 - 07:19.367	advance	arm ext	retreat	parry
07:22.303 - 07:23.437	lunge	arm ext	advance	arm ext
07:46.833 - 07:47.567	lunge		lunge
08:02.733 - 08:05.833	advance		retreat
08:05.833 - 08:06.533	lunge		advance	arm ext
08:11.100 - 08:13.033	neutral		neutral
08:13.033 - 08:13.933	advance		advance
08:13.933 - 08:19.333	retreat		advance
08:19.333 - 08:20.233	retreat	parry	lunge 	arm ext
08:20.600 - 08:24.133	advance		retreat
08:24.133 - 08:24.600	advance		advance	arm ext
08:24.600 - 08:25.067	advance	arm ext	retreat	parry
08:42.768 - 08:43.335	retreat		lunge 	arm ext
08:46.633 - 08:47.301	retreat	parry	lunge 	arm ext
08:47.301 - 08:47.834	lunge	arm ext	retreat
08:49.402 - 08:50.799	retreat		advance
08:50.799 - 08:51.500	retreat	parry	lunge 	arm ext
08:51.500 - 08:57.572	advance		retreat
08:57.572 - 08:58.506	advance	arm ext	advance	parry
09:09.866 - 09:10.900	retreat		advance
09:10.900 - 09:15.733	advance		retreat
09:25.766 - 09:26.466	neutral	arm ext	advance	arm ext
09:32.666 - 09:33.400	advance	parry	lunge 	arm ext
09:33.400 - 09:33.767	advance		retreat	arm ext
09:42.899 - 09:43.734	retreat	parry	lunge 	arm ext
09:46.303 - 09:46.970	advance	arm ext	retreat	parry
09:54.200 - 09:54.900	lunge	arm ext	advance	parry
10:04.934 - 10:05.534	retreat	parry	lunge 	arm ext
10:12.733 - 10:13.301	retreat	parry	lunge 	arm ext
10:13.301 - 10:18.400	advance		retreat
10:18.400 - 10:19.467	lunge	arm ext	advance	arm ext
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

    bad = False
    for side in ("left", "right"):
        s = sorted([r for r in rows if r[0] == side], key=lambda r: r[1])
        for a, b in zip(s, s[1:]):
            if b[1] < a[2] - 1e-6:
                print(f"  !! {side} overlap: [{a[1]:.3f},{a[2]:.3f}] {a[3]}/{a[4]} "
                      f"and [{b[1]:.3f},{b[2]:.3f}] {b[3]}/{b[4]}")
                bad = True
    if bad:
        return 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("# Ground-truth interval labels for data/raw_video/6.mp4. Aaron, 2026-08-16.\n")
        f.write("# 1920x1080 @ 29.97 fps, 10.7 min -- same broadcast format as bouts 1-5.\n")
        f.write("# Two-track schema; `walk` -> walking, `arm ext` -> extension.\n")
        f.write("# Two source slips, corrected here: end `0629.135` read as `06:29.135`,\n")
        f.write("# and the riposte at 00:56.200 read as starting 00:56.833 where the\n")
        f.write("# attack it answers ends -- the only reading that leaves no overlap.\n")
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
