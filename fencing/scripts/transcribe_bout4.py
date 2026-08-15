"""Transcribe Aaron's bout 4 table -> two-track CSV. Anomalies FLAGGED, not fixed."""
import sys
from pathlib import Path

PROJECT = Path(r"c:\Users\aaron\OneDrive\Documents\GitHub\fencing\fencing")
OUT = PROJECT / "data" / "labels" / "bout4_intervals_2track.csv"

FIXED = {"20.53.520": "20:53.520", "22:118.967": "22:18.967"}

R = [
 ("00:0.0 - 00:00.767","neutral","","neutral",""),
 ("00:02.135 - 00:05.071","advance","","retreat",""),
 ("00:05.204 - 00:05.738","lunge","arm ext","retreat","parry"),
 ("00:05.772 - 00:06.706","retreat","parry","advance",""),
 ("00:07.374 - 00:08.341","walk","","walk",""),
 ("00:23.890 - 00:25.391","retreat","","advance",""),
 ("00:25.391 - 00:25.958","neutral","arm ext","advance",""),
 ("00:25.958 - 00:30.361","retreat","","advance",""),
 ("00:30.361 - 00:31.261","retreat","","lunge","arm ext"),
 ("00:31.191 - 00:32.932","walk","","walk",""),
 ("00:54.854 - 00:55.621","lunge","arm ext","retreat","parry"),
 ("00:58.290 - 01:05.760","walk","","walk",""),
 ("01:26.586 - 01:27.153","advance","","retreat",""),
 ("01:27.153 - 01:27.820","lunge","arm ext","retreat","parry"),
 ("01:32.792 - 01:35.760","retreat","","advance",""),
 ("01:48.975 - 01:50.509","advance","","retreat",""),
 ("01:50.509 - 01:51.209","lunge","arm ext","retreat","parry"),
 ("02:13.632 - 02:14.366","lunge","arm ext","retreat","parry"),
 ("02:21.900 - 02:25.110","retreat","","advance",""),
 ("02:31.710 - 02:32.939","retreat","","advance",""),
 ("02:32.929 - 02:33.472","lunge","arm ext","advance","parry"),
 ("03:19.803 - 03:20.570","lunge","arm ext","retreat","parry"),
 ("03:20.570 - 03:22.838","retreat","","advance",""),
 ("03:24.360 - 03:26.106","walk","","walk",""),
 ("04:42.170 - 04:43.004","lunge","arm ext","lunge","arm ext"),
 ("04:43.004 - 04:44.640","walk","","walk",""),
 ("05:20.100 - 05:21.568","neutral","","neutral",""),
 ("05:22.467 - 05:23.134","lunge","arm ext","lunge","arm ext"),
 ("05:23.134 - 05:29.310","walk","","walk",""),
 ("05:31.496 - 05:33.598","advance","","retreat",""),
 ("05:33.598 - 05:34.465","retreat","parry","lunge","arm ext"),
 ("05:34.500 - 05:38.310","walk","","walk",""),
 ("05:43.770 - 05:44.537","advance","parry","lunge","arm ext"),
 ("05:45.271 - 05:52.680","walk","","walk",""),
 ("05:56.760 - 06:00.120","advance","","retreat",""),
 ("06:00.120 - 06:00.453","advance","","lunge","arm ext"),
 ("06:01.140 - 06:03.570","walk","","walk",""),
 ("06:19.553 - 06:21.187","retreat","parry","advance","arm ext"),
 ("06:40.693 - 06:50.280","walk","","walk",""),
 ("06:52.200 - 06:52.900","advance","","lunge","arm ext"),
 ("06:53.580 - 07:01.980","walk","","walk",""),
 ("07:05.550 - 07:06.780","advance","","retreat",""),
 ("07:15.900 - 07:19.223","neutral","","neutral",""),
 ("07:20.490 - 07:23.025","advance","","retreat",""),
 ("07:27.929 - 07:28.596","lunge","arm ext","retreat","parry"),
 ("07:43.440 - 07:44.140","neutral","","lunge",""),
 ("08:02.370 - 08:03.070","neutral","","neutral","parry"),
 ("08:44.160 - 08:47.250","walk","","walk",""),
 ("08:47.250 - 08:50.580","neutral","","neutral",""),
 ("08:51.540 - 08:53.490","retreat","","advance",""),
 ("08:53.490 - 08:54.724","retreat","parry","advance",""),
 ("09:10.583 - 09:11.750","retreat","parry","lunge","arm ext"),
 ("09:14.220 - 09:15.254","lunge","arm ext","retreat","parry"),
 ("09:15.780 - 09:16.914","lunge","arm ext","advance","parry"),
 ("09:49.830 - 09:50.831","retreat","parry","lunge","arm ext"),
 ("09:50.831 - 09:55.050","advance","","retreat",""),
 ("09:55.050 - 09:55.917","lunge","arm ext","retreat","parry"),
 ("10:12.570 - 10:17.190","neutral","","neutral",""),
 ("10:31.529 - 10:32.663","retreat","parry","lunge","arm ext"),
 ("10:32.700 - 10:38.220","walk","","walk",""),
 ("11:02.400 - 11:07.080","advance","","retreat",""),
 ("11:07.080 - 11:08.214","lunge","arm ext","retreat","parry"),
 ("11:08.220 - 11:13.440","retreat","","advance",""),
 ("11:13.440 - 11:14.140","retreat","parry","lunge","arm ext"),
 ("11:38.580 - 11:45.360","walk","","walk",""),
 ("11:47.010 - 11:48.077","lunge","arm ext","lunge","arm ext"),
 ("12:53.190 - 12:54.057","retreat","parry","lunge","arm ext"),
 ("12:54.060 - 12:57.480","walk","","walk",""),
 ("13:00.810 - 13:01.510","retreat","parry","lunge","arm ext"),
 ("13:03.960 - 13:04.527","advance","","lunge","arm ext"),
 ("13:24.120 - 13:26.310","neutral","","neutral",""),
 ("13:27.060 - 13:27.827","lunge","arm ext","retreat",""),
 ("13:28.294 - 13:34.650","retreat","","advance",""),
 ("13:34.650 - 13:34.980","lunge","arm ext","neutral","parry"),
 ("13:34.980 - 13:35.280","neutral","parry","",""),
 ("13:35.280 - 13:35.613","neutral","","retreat","parry"),
 ("13:35.613 - 13:35.879","retreat","parry","retreat",""),
 ("13:43.393 - 13:44.193","lunge","arm ext","neutral","parry"),
 ("14:03.884 - 14:05.490","neutral","","neutral",""),
 ("14:06.300 - 14:07.034","lunge","arm ext","lunge",""),
 ("14:07.470 - 14:16.740","walk","","walk",""),
 ("14:21.810 - 14:27.030","walk","","walk",""),
 ("14:29.633 - 14:30.233","retreat","parry","lunge","arm ext"),
 ("14:37.196 - 14:38.866","neutral","","neutral",""),
 ("14:39.543 - 14:40.310","advance","","lunge","arm ext"),
 ("14:40.350 - 14:44.850","walk","","walk",""),
 ("14:44.850 - 14:46.350","neutral","","neutral",""),
 ("14:47.490 - 14:50.100","advance","","retreat",""),
 ("14:50.100 - 14:50.967","lunge","arm ext","retreat","parry"),
 ("14:57.533 - 14:58.300","lunge","arm ext","lunge","arm ext"),
 ("14:58.310 - 15:01.440","walk","","walk",""),
 ("15:39.270 - 15:43.500","neutral","","neutral",""),
 ("15:44.520 - 15:47.550","retreat","","advance",""),
 ("15:47.550 - 15:51.600","advance","","retreat",""),
 ("15:51.600 - 15:52.467","lunge","arm ext","retreat","parry"),
 ("15:52.467 - 15:56.550","retreat","","advance",""),
 ("15:56.550 - 15:57.150","lunge","arm ext","retreat","parry"),
 ("15:57.150 - 16:00.330","retreat","","advance",""),
 ("16:00.870 - 16:03.390","walk","","walk",""),
 ("16:08.190 - 16:13.290","walk","","walk",""),
 ("16:54.300 - 17:00.750","walk","","walk",""),
 ("17:00.752 - 17:03.600","neutral","","neutral",""),
 ("17:05.040 - 17:07.440","retreat","","advance",""),
 ("17:07.440 - 17:08.140","retreat","parry","lunge","arm ext"),
 ("17:08.880 - 17:13.650","advance","","retreat",""),
 ("17:23.730 - 17:27.330","advance","","retreat",""),
 ("18:11.453 - 18:12.491","retreat","arm ext","lunge","arm ext"),
 ("18:29.640 - 18:31.110","advance","","retreat","arm ext"),
 ("18:31.110 - 18:33.892","advance","","retreat",""),
 ("18:33.892 - 18:34.525","advance","arm ext","lunge","arm ext"),
 ("18:35.158 - 18:38.160","walk","","walk",""),
 ("18:45.330 - 18:47.553","advance","","retreat","arm ext"),
 ("19:05.460 - 19:07.620","neutral","","neutral",""),
 ("19:09.660 - 19:10.714","advance","","retreat","arm ext"),
 ("19:11.160 - 19:11.860","neutral","parry","lunge","arm ext"),
 ("19:11.860 - 19:14.010","walk","","walk",""),
 ("19:33.220 - 19:33.554","advance","","retreat","arm ext"),
 ("19:33.554 - 19:34.254","advance","","lunge","arm ext"),
 ("19:40.350 - 19:45.060","walk","","walk",""),
 ("20:00.060 - 20:04.890","retreat","arm ext","advance",""),
 ("20:04.890 - 20:05.857","retreat","","lunge","arm ext"),
 ("20:06.330 - 20:07.410","advance","","retreat","arm ext"),
 ("20:07.410 - 20:08.820","advance","","retreat",""),
 ("20:10.050 - 20:18.330","walk","","walk",""),
 ("20:20.730 - 20:21.810","neutral","parry","lunge","arm ext"),
 ("20:22.260 - 20:26.820","walk","","walk",""),
 ("20:29.269 - 20:30.470","retreat","arm ext","lunge","arm ext"),
 ("20:30.470 - 20:32.070","walk","","walk",""),
 ("20:50.496 - 20:51.463","lunge","arm ext","retreat","parry"),
 ("20:51.463 - 20.53.520","walk","","walk",""),
 ("21:20.920 - 21:26.040","walk","","walk",""),
 ("21:36.600 - 21:40.000","advance","","retreat",""),
 ("22:07.880 - 22:08.960","advance","","retreat",""),
 ("22:08.960 - 22:09.694","advance","parry","retreat","arm ext"),
 ("22:17.866 - 22:118.967","lunge","arm ext","retreat",""),
 ("22:29.400 - 22:31.299","advance","","retreat",""),
 ("22:31.299 - 22:32.400","lunge","arm ext","retreat","parry"),
 ("22:39.146 - 22:40.180","neutral","parry","lunge","arm ext"),
 ("22:40.045 - 22:41.012","advance","parry","","arm ext"),
 ("22:55.813 - 22:57.480","advance","","retreat",""),
 ("22:57.480 - 22:58.647","lunge","arm ext","retreat","parry"),
 ("23:28.114 - 23:29.800","advance","","retreat",""),
 ("23:29.800 - 23:37.240","retreat","","advance",""),
 ("23:39.320 - 23:45.040","walk","","walk",""),
 ("23:47.646 - 23:48.480","retreat","parry","lunge","arm ext"),
 ("24:05.600 - 24:06.367","advance","arm ext","lunge","arm ext"),
 ("24:39.927 - 24:41.228","lunge","arm ext","retreat","parry"),
 ("24:41.228 - 24:43.840","walk","","walk",""),
 ("24:51.280 - 24:52.247","retreat","parry","lunge","arm ext"),
 ("24:58.166 - 24:59.534","lunge","arm ext","retreat","parry"),
 ("25:06.280 - 25:11.240","advance","","retreat",""),
 ("25:34.680 - 25:40.040","walk","","walk",""),
 ("25:43.280 - 25:44.600","advance","","retreat","arm ext"),
]

FW = {"walk": "walking", "advance": "advance", "retreat": "retreat",
      "lunge": "lunge", "neutral": "neutral"}
BL = {"arm ext": "extension", "parry": "parry", "": "none"}


def secs(t):
    t = FIXED.get(t.strip(), t.strip())
    m, s = t.split(":")
    return int(m) * 60 + float(s)


def main():
    notes, rows = [], []
    for i, (tr, afw, abl, bfw, bbl) in enumerate(R, start=1):
        a_raw, b_raw = [x.strip() for x in tr.split(" - ")]
        for raw in (a_raw, b_raw):
            if raw in FIXED:
                notes.append(f"row {i}: timestamp {raw!r} malformed, read as {FIXED[raw]!r}")
        s, e = secs(a_raw), secs(b_raw)
        for who, fw, bl in (("left", afw, abl), ("right", bfw, bbl)):
            if not fw:
                if bl:
                    notes.append(f"row {i} ({s:.3f}s): {who} has blade {bl!r} but NO footwork "
                                 f"-- dropped per your rule; that blade action is lost")
                continue
            rows.append((who, s, e, FW[fw], BL[bl]))

    rows.sort(key=lambda r: (r[0], r[1]))
    MAX_CLAMP = 0.20
    fixed = []
    for who in ("left", "right"):
        iv = [r for r in rows if r[0] == who]
        for i, (x, y) in enumerate(zip(iv, iv[1:])):
            if y[1] < x[2] - 1e-9:
                gap = x[2] - y[1]
                if gap <= MAX_CLAMP:
                    notes.append(f"clamped {who} {x[3]} end {x[2]:.3f} -> {y[1]:.3f} "
                                 f"(overlapped next by {gap:.3f}s)")
                    iv[i] = (x[0], x[1], y[1], x[3], x[4])
                else:
                    notes.append(f"OVERLAP TOO LARGE TO CLAMP, left in place -- {who}: "
                                 f"{x[1]:.3f}-{x[2]:.3f} ({x[3]}) vs {y[1]:.3f}-{y[2]:.3f} "
                                 f"({y[3]})  [{gap:.3f}s]")
        fixed.extend(iv)
    rows = sorted(fixed, key=lambda r: (r[0], r[1]))

    with open(OUT, "w", encoding="utf-8", newline="") as f:
        f.write("# Ground-truth interval labels for data/raw_video/4.mp4. Aaron, 2026-08-09.\n"
                "# TWO-TRACK schema (footwork + blade), 26.1 min source, sparse by design:\n"
                "# a gap means no arm/blade action was happening there, not neutral.\n"
                "# Unlabelled time is EXCLUDED from scoring, so gaps are safe.\n"
                "#\n"
                "# `arm ext` -> `extension`, `walk` -> `walking`. `extension` is not one of\n"
                "# the six model classes, so evaluate_labels.py collapses those rows to the\n"
                "# FOOTWORK label rather than dropping them as unscorable.\n"
                "#\n"
                "# Rows where a fencer's footwork cell was blank are omitted for that fencer.\n"
                "# See the transcription notes printed by scratchpad/bout4_raw.py.\n"
                "fencer,start,end,footwork,blade\n")
        for who, s, e, fw, bl in rows:
            f.write(f"{who},{s:.3f},{e:.3f},{fw},{bl}\n")

    print(f"wrote {OUT.name}: {len(rows)} intervals from {len(R)} table rows\n")
    print("TRANSCRIPTION NOTES -- check these:")
    for n in notes:
        print(f"  {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
