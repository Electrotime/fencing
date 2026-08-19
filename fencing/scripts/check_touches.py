"""Validate a touch-outcome table and convert it to the canonical time,scorer,score,note CSV."""
import argparse
import csv
import re
import tempfile
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
LAB = PROJECT / "data" / "labels"
SCORERS = {"left", "right", "none"}
TIME = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")
SCORE = re.compile(r"^(\d+)\s*[-:]\s*(\d+)$")
BOTH = {"both", "simul", "simultaneous", "double"}
ALIAS = {"time": "time", "time stamp": "time", "timestamp": "time",
         "score": "score", "hit": "hit", "note": "note", "scorer": "scorer",
         "side": "side", "side conduct hit": "side", "side conducting hit": "side"}


def parse_time(s):
    m = TIME.match((s or "").strip())
    if not m:
        return None
    return (int(m.group(1)) if m.group(1) else 0) * 60 + float(m.group(2))


def read_rows(path):
    """Normalised dicts from a tab- or comma-separated table with aliased headers."""
    lines = [l for l in Path(path).read_text(encoding="utf-8-sig").splitlines()
             if l.strip() and not l.lstrip().startswith("#")]
    if not lines:
        return [], []
    rdr = csv.reader(lines, delimiter="\t" if "\t" in lines[0] else ",")
    head = [ALIAS.get(re.sub(r"\s+", " ", h.strip().lower()), "") for h in next(rdr)]
    rows = [{k: (r[i].strip() if i < len(r) else "")
             for i, k in enumerate(head) if k} for r in rdr]
    return rows, [h for h in head if h]


def check(path):
    """Returns (problems, rows) with scorer derived from score movement when absent."""
    problems = []
    raw, cols = read_rows(path)
    if "time" not in cols:
        return [(0, f"need a time column -- found {cols}")], []
    if "scorer" not in cols and "score" not in cols:
        return [(0, "need a scorer column, or a score column to derive it from")], []

    rows = []
    for i, d in enumerate(raw, 1):
        t = parse_time(d.get("time", ""))
        if t is None:
            problems.append((i, f"bad time {d.get('time', '')!r}; want 1:23.4 or 83.4"))
            continue
        s = None
        if d.get("score"):
            m = SCORE.match(d["score"])
            if not m:
                problems.append((i, f"bad score {d['score']!r}; want 3-2 or 3:2"))
                continue
            s = (int(m.group(1)), int(m.group(2)))
        side = d.get("side", "").lower()
        rows.append(dict(line=i, t=t, score=s, hit=d.get("hit", "").lower(),
                         side="both" if side in BOTH else side,
                         scorer=d.get("scorer", "").lower(), note=d.get("note", "")))

    for a, b in zip(rows, rows[1:]):
        if b["t"] <= a["t"]:
            problems.append((b["line"],
                             f"time {b['t']:.2f} is not after the previous {a['t']:.2f}"))

    prev, seeded = (0, 0), False
    for r in rows:
        if r["scorer"] and r["scorer"] not in SCORERS:
            problems.append((r["line"], f"scorer {r['scorer']!r} not one of {sorted(SCORERS)}"))
            r["scorer"] = ""
        s = r["score"]
        if s is None:
            r["scorer"] = r["scorer"] or "none"
            continue
        if not seeded:
            seeded = True
            if sum(s) > 1:
                prev = (s[0] - (r["side"] == "left"), s[1] - (r["side"] == "right"))
                problems.append((r["line"], f"first score {s[0]}-{s[1]} is not 0-0 or the "
                                            f"first touch; checksum seeded from here"))
        d = (s[0] - prev[0], s[1] - prev[1])
        won = {(0, 0): "none", (1, 0): "left", (0, 1): "right"}.get(d)
        if won is None:
            problems.append((r["line"], f"score {s[0]}-{s[1]} after {prev[0]}-{prev[1]} moves "
                                        f"by {d}; a row is probably missing"))
            won = "none"
        elif r["side"] and won in ("left", "right") and r["side"] != won:
            problems.append((r["line"], f"score advanced for {won} but the side column "
                                        f"says {r['side']!r}"))
        if r["scorer"] and r["scorer"] != won:
            problems.append((r["line"], f"scorer says {r['scorer']!r} but the score moved "
                                        f"{won!r}"))
        r["scorer"] = r["scorer"] or won
        prev = s

    for r in rows:
        h = r["hit"].replace("-", " ").replace("_", " ")
        if h.startswith("off") and r["scorer"] != "none":
            problems.append((r["line"], f"an off-target hit awarded a point to {r['scorer']}"))
        if h.startswith("valid") and r["scorer"] == "none" and r["side"] not in ("both", ""):
            problems.append((r["line"], f"a valid hit by {r['side']} scored nothing; either "
                                        f"the score is missing or it was a double"))
        if not r["note"]:
            r["note"] = ("off_target" if h.startswith("off")
                         else "simul" if r["scorer"] == "none" and r["side"] == "both" else "")

    if not any(r["score"] for r in rows):
        problems.append((0, "no score column filled -- the checksum cannot run "
                            "(allowed, but a missed touch will go unnoticed)"))
    return problems, rows


def write_canonical(rows, out):
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time", "scorer", "score", "note"])
        for r in rows:
            sc = f"{r['score'][0]}-{r['score'][1]}" if r["score"] else ""
            w.writerow([f"{r['t']:.3f}", r["scorer"], sc, r["note"]])


def _self_test():
    good = "time,scorer,score\n0:10.0,left,1-0\n0:20.0,none,1-0\n0:30.0,right,1-1\n"
    derived = ("Score\tTime stamp\tHit \tSide conduct hit\n"
               "\t00:10.0\toff target\tBoth\n"
               "1:0\t00:20.0\tvalid\tleft\n"
               "\t00:25.0\tvalid\tBoth\n"
               "1:1\t00:30.0\tvalid\tright\n")
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "g.csv").write_text(good)
        p, r = check(d / "g.csv")
        assert not p and len(r) == 3, (p, r)

        (d / "t.tsv").write_text(derived)
        p, r = check(d / "t.tsv")
        assert not p, p
        assert [x["scorer"] for x in r] == ["none", "left", "none", "right"], r
        assert [x["note"] for x in r] == ["off_target", "", "simul", ""], r

        (d / "b.csv").write_text("time,scorer,score\n0:10.0,left,1-0\n0:30.0,right,1-3\n")
        assert any("missing" in m for _, m in check(d / "b.csv")[0])

        (d / "o.csv").write_text("time,scorer\n0:30.0,left\n0:10.0,right\n")
        assert any("not after" in m for _, m in check(d / "o.csv")[0])

        # the side column must contradict a mistyped score
        (d / "x.tsv").write_text("Score\tTime stamp\tHit\tSide conduct hit\n"
                                 "1:0\t00:20.0\tvalid\tright\n")
        assert any("side column" in m for _, m in check(d / "x.tsv")[0])

        # off-target may never move the score
        (d / "y.tsv").write_text("Score\tTime stamp\tHit\tSide conduct hit\n"
                                 "1:0\t00:20.0\toff target\tleft\n")
        assert any("off-target" in m for _, m in check(d / "y.tsv")[0])

        # a lone valid light that scores nothing is a missed score entry
        (d / "z.tsv").write_text("Score\tTime stamp\tHit\tSide conduct hit\n"
                                 "\t00:20.0\tvalid\tleft\n1:0\t00:30.0\tvalid\tleft\n")
        assert any("scored nothing" in m for _, m in check(d / "z.tsv")[0])
    print("check_touches self-test: ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--out", help="write the canonical CSV here (one input file only)")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    _self_test()
    if a.self_test or not a.files:
        if not a.files:
            print("usage: py -3 scripts/check_touches.py data/labels/bout7_touches.tsv "
                  "--out data/labels/bout7_touches.csv")
        return 0

    bad = 0
    for f in a.files:
        p = Path(f)
        if not p.exists():
            p = LAB / p.name
        problems, rows = check(p)
        print(f"\n{p.name}: {len(rows)} decisions")
        c = Counter(r["scorer"] for r in rows)
        for k in ("left", "right", "none"):
            print(f"    {k:<11}{c.get(k, 0):>5}")
        n_off = sum(1 for r in rows if r["note"] == "off_target")
        n_sim = sum(1 for r in rows if r["note"] == "simul")
        print(f"    (off_target{n_off:>5}, simul{n_sim:>4})")
        if rows:
            print(f"    span {rows[0]['t']:.1f}s to {rows[-1]['t']:.1f}s, "
                  f"final {rows[-1]['score'] or '?'}")
            gaps = sorted(((b["t"] - a2["t"], a2["t"], b["t"])
                           for a2, b in zip(rows, rows[1:])), reverse=True)[:3]
            print("    longest gaps: " + ", ".join(f"{g:.0f}s at {s:.0f}-{e:.0f}s"
                                                   for g, s, e in gaps))
        for i, m in problems:
            print(f"  !! row {i}: {m}")
        bad += sum(1 for _, m in problems if "checksum cannot run" not in m)
        if a.out and len(a.files) == 1:
            write_canonical(rows, a.out)
            print(f"    -> wrote {a.out}")
    print("\nno errors" if not bad else f"\n{bad} problems")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
