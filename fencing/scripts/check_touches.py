"""Validate a touch-outcome CSV: format, ordering, and the running-score checksum."""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LAB = PROJECT / "data" / "labels"
SCORERS = {"left", "right", "none"}
TIME = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")
SCORE = re.compile(r"^(\d+)\s*-\s*(\d+)$")


def parse_time(s):
    m = TIME.match(s.strip())
    if not m:
        return None
    mins = int(m.group(1)) if m.group(1) else 0
    return mins * 60 + float(m.group(2))


def check(path, fps_span=None):
    problems, rows = [], []
    with open(path, encoding="utf-8") as f:
        rdr = csv.DictReader(r for r in f if not r.startswith("#"))
        cols = set(rdr.fieldnames or [])
        if not {"time", "scorer"} <= cols:
            return [(0, f"need columns time,scorer -- found {sorted(cols)}")], []
        has_score = "score" in cols
        for i, r in enumerate(rdr, 1):
            t = parse_time(r["time"])
            sc = r["scorer"].strip().lower()
            if t is None:
                problems.append((i, f"bad time {r['time']!r}; want 1:23.4 or 83.4"))
                continue
            if sc not in SCORERS:
                problems.append((i, f"scorer {sc!r} not one of {sorted(SCORERS)}"))
                continue
            s = None
            if has_score and r.get("score", "").strip():
                m = SCORE.match(r["score"].strip())
                if not m:
                    problems.append((i, f"bad score {r['score']!r}; want 3-2"))
                    continue
                s = (int(m.group(1)), int(m.group(2)))
            rows.append((i, t, sc, s))

    for (i, t, _, _), (j, u, _, _) in zip(rows, rows[1:]):
        if u <= t:
            problems.append((j, f"time {u:.2f} is not after the previous {t:.2f}"))

    # the checksum: a running score must advance by exactly the scorer, and only then
    scored = [r for r in rows if r[3] is not None]
    if len(scored) >= 2:
        for (i, _, _, a), (j, _, sc, b) in zip(scored, scored[1:]):
            want = (a[0] + (sc == "left"), a[1] + (sc == "right"))
            if b != want:
                problems.append(
                    (j, f"score {b[0]}-{b[1]} after a {sc!r} call, expected "
                        f"{want[0]}-{want[1]} (a row is probably missing)"))
    elif not scored:
        problems.append((0, "no score column filled -- the checksum cannot run "
                            "(allowed, but a missed touch will go unnoticed)"))

    if fps_span is not None and rows and rows[-1][1] > fps_span + 1.0:
        problems.append((rows[-1][0], f"time {rows[-1][1]:.1f}s is past the video "
                                      f"({fps_span:.1f}s)"))
    return problems, rows


def _self_test():
    import tempfile
    good = "time,scorer,score\n0:10.0,left,1-0\n0:20.0,none,1-0\n0:30.0,right,1-1\n"
    bad = "time,scorer,score\n0:10.0,left,1-0\n0:30.0,right,1-3\n"
    with tempfile.TemporaryDirectory() as d:
        g = Path(d) / "g.csv"; g.write_text(good)
        b = Path(d) / "b.csv"; b.write_text(bad)
        pg, rg = check(g)
        assert not pg, pg
        assert len(rg) == 3, rg
        pb, _ = check(b)
        assert any("missing" in m for _, m in pb), pb
        o = Path(d) / "o.csv"
        o.write_text("time,scorer\n0:30.0,left\n0:10.0,right\n")
        po, _ = check(o)
        assert any("not after" in m for _, m in po), po
    print("check_touches self-test: ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test or not a.files:
        _self_test()
        if not a.files:
            print("usage: py -3 scripts/check_touches.py data/labels/bout6_touches.csv")
        return 0
    _self_test()
    bad = 0
    for f in a.files:
        p = Path(f)
        if not p.exists():
            p = LAB / p.name
        problems, rows = check(p)
        print(f"\n{p.name}: {len(rows)} decisions")
        c = Counter(r[2] for r in rows)
        for k in ("left", "right", "none"):
            print(f"    {k:<7}{c.get(k, 0):>5}")
        if rows:
            print(f"    span {rows[0][1]:.1f}s to {rows[-1][1]:.1f}s")
        for i, m in problems:
            print(f"  !! row {i}: {m}")
        bad += sum(1 for i, m in problems if "checksum cannot run" not in m)
    print("\nno errors" if not bad else f"\n{bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
