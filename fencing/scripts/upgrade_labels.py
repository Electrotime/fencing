"""Lift a single-track label file to the two-track (footwork+blade) schema.

Most of the conversion is mechanical, but NOT all of it, and the rows it cannot
decide are exactly the interesting ones. Those get written as TODO rather than
guessed -- guessing is how the last three metrics ended up measuring an artifact
of the data instead of fencing.

  neutral, walking   -> footwork=<same>, blade=none      (confident: fencer is idle)
  advance, retreat   -> footwork=<same>, blade=TODO      (usually none, but a parry
                                                          may have gone unrecorded --
                                                          the old schema forced ONE
                                                          label, so a parry during a
                                                          retreat had to be dropped)
  lunge              -> footwork=lunge, blade=TODO       (a lunge nearly always
                                                          carries an extension, but
                                                          "nearly always" is not data)
  parry              -> footwork=TODO,  blade=parry      (the footwork underneath was
                                                          never recorded -- this is the
                                                          whole reason for the schema)
  extension          -> footwork=TODO,  blade=extension

Run check_labels.py on the result once the TODOs are filled in.

usage: py -3 scripts/upgrade_labels.py data/labels/bout1_intervals.csv
       (writes bout1_intervals_2track.csv beside it; never overwrites the input)
"""
import csv
import sys
from collections import Counter
from pathlib import Path

# (footwork, blade); None means "we genuinely don't know -- ask the labeller"
MAP = {
    "neutral":   ("neutral", "none"),
    "walking":   ("walking", "none"),
    "advance":   ("advance", None),
    "retreat":   ("retreat", None),
    "lunge":     ("lunge", None),
    "parry":     (None, "parry"),
    "extension": (None, "extension"),
}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    src = Path(sys.argv[1])
    dst = src.with_name(f"{src.stem}_2track.csv")
    if dst.exists():
        print(f"refusing to overwrite {dst.name} -- delete it first if you meant to")
        return 1

    header, rows = [], []
    with open(src, encoding="utf-8") as f:
        lines = f.readlines()
    comments = [ln for ln in lines if ln.startswith("#")]
    rdr = csv.DictReader(ln for ln in lines if not ln.startswith("#"))
    if "label" not in (rdr.fieldnames or []):
        print(f"{src.name} has no `label` column -- already two-track?")
        return 1
    rows = [dict(r) for r in rdr]

    todo = Counter()
    out = []
    for r in rows:
        lab = r["label"].strip()
        if lab not in MAP:
            print(f"unknown label {lab!r} -- add it to MAP first")
            return 1
        fw, bl = MAP[lab]
        if fw is None:
            todo["footwork"] += 1
        if bl is None:
            todo["blade"] += 1
        out.append({"fencer": r["fencer"], "start": r["start"], "end": r["end"],
                    "footwork": fw or "TODO", "blade": bl or "TODO"})

    with open(dst, "w", encoding="utf-8", newline="") as f:
        f.writelines(comments)
        f.write("# UPGRADED to two-track by scripts/upgrade_labels.py.\n"
                "# TODO cells could not be derived from the single-track label and need\n"
                "# a human -- see that script's docstring for why each one is unknown.\n")
        w = csv.DictWriter(f, fieldnames=["fencer", "start", "end", "footwork", "blade"])
        w.writeheader()
        w.writerows(out)

    n_todo = sum(1 for r in out if "TODO" in (r["footwork"], r["blade"]))
    print(f"wrote {dst.name}: {len(out)} rows, {n_todo} need review "
          f"({todo['footwork']} footwork, {todo['blade']} blade)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
