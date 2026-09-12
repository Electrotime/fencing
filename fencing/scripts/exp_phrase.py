"""Right-of-way accuracy split by phrase type and by how ambiguous the call was."""
import argparse
import re
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

import exp_contested as C
import exp_touch_probe as PR

LAB = PROJECT / "data" / "labels"
WORKLIST = LAB / "phrase_worklist.tsv"
# bout 13 is the same match as bout 9, so it is not a second sample
BOUTS = ("8", "9", "10", "11", "12", "14")
LETTER = {"c": "clean", "e": "clean", "h": "close-blade",
          "a": "attack-in-prep", "o": "other", "s": "both-attack"}
ORDER = ("clean", "close-blade", "both-attack", "attack-in-prep", "other")


def parse_row(line):
    """(bout, time, phrase, difficulty, alternatives) from one worklist line."""
    f = (line.split("\t") + [""] * 8)[:8]
    if not f[1].strip():
        return None
    phrase_raw, diff_raw = f[5].strip().lower(), f[6].strip()
    bare = re.sub(r"\(.*?\)", " ", phrase_raw)
    m = re.match(r"\s*(\d+)", re.sub(r"\(.*?\)", "", diff_raw))
    if m is None:                      # difficulty sometimes typed after the letter
        m = re.search(r"(\d+)\s*$", bare)
        if m:
            bare = bare[:m.start()]
    letters = [w for w in re.findall(r"[a-z]+", bare) if w in LETTER]
    if not letters or m is None:
        return None
    return (f[0], round(float(f[1]), 2), LETTER[letters[0]], int(m.group(1)),
            [LETTER[w] for w in letters[1:]])


def load_labels():
    out = {}
    for line in WORKLIST.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or line.startswith("bout\t"):
            continue
        got = parse_row(line)
        if got:
            out[(got[0], got[1])] = got[2:]
    return out


def build(bouts):
    T, D, pri, src = [], [], [], []
    for stem in bouts:
        probs = np.load(LAB / PR.CACHE[stem])
        for t, p, _ in C.read_contested(stem):
            if not p:
                continue
            T.append((stem, t, p)); D.append(probs); pri.append(p)
            src.append((stem, round(t, 2)))
    X, names, ok = PR.build(T, D, 0.3)
    keep = [s for s, k in zip(src, ok) if k]
    return X[names.index(PR.PREREG)], np.array(pri)[ok], keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--perm", type=int, default=20000)
    a = ap.parse_args()
    lab = load_labels()
    x, po, src = build(BOUTS)
    ph = np.array([lab.get(s, ("",))[0] for s in src])
    df = np.array([lab[s][1] if s in lab else np.nan for s in src], float)
    bo = np.array([s[0] for s in src])

    z = np.zeros_like(x)
    for b in set(bo):                  # per-bout scale differs, so normalise within bout
        m = bo == b
        z[m] = (x[m] - x[m].mean()) / (x[m].std() or 1)
    mag = np.abs(z)

    print(f"{(ph != '').sum()} labelled halts, bouts {','.join(BOUTS)}\n")
    print(f"{'phrase':>16} {'n':>4} {'AUC':>6} {'p(AUC<0.5)':>11} {'mean |z|':>9} {'med diff':>9}")
    for c in ORDER:
        m = ph == c
        if m.sum() < 3:
            continue
        y = po[m] == "left"
        v = PR.auc(x[m], y)
        _, pf, _ = PR.maxstat_p(x[m][None, :], y, a.perm)
        below = 1 - pf[0] / 2 if v > 0.5 else pf[0] / 2
        print(f"{c:>16} {m.sum():>4} {v:>6.2f} {below:>11.3f} "
              f"{mag[m].mean():>9.2f} {np.nanmedian(df[m]):>9.0f}")

    print(f"\nregistered: {PR.PHRASE_PREREG}")
    m = ph == "attack-in-prep"
    print(f"  attack-in-prep n={m.sum()} AUC {PR.auc(x[m], po[m] == 'left'):.2f}")
    for c in ("both-attack", "close-blade"):
        k = ph == c
        print(f"  {c} |z| {mag[k].mean():.2f} vs clean {mag[ph=='clean'].mean():.2f}")
    return 0


def _self_test():
    assert parse_row("8\t33.199\t00:33.199\ttwo_colour\tleft\tc\t3\t")[2:4] == ("clean", 3)
    assert parse_row("14\t16.9\t00:16.9\ttwo_colour\tleft\th and s\t4\t")[4] == ["both-attack"]
    r = parse_row("14\t99.0\t01:39.0\ttwo_colour\tleft\tc (could be s) 3\t\t")
    assert r[2:4] == ("clean", 3), r          # difficulty typed after the letter
    assert parse_row("9\t1.0\t00:01.0\tmixed\tleft\t\t\t") is None
    print("exp_phrase self-test: ok")


if __name__ == "__main__":
    raise SystemExit(_self_test() if "--self-test" in sys.argv else main())
