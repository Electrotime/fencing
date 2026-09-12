"""Halts found from body motion alone -- no scoreboard, no lamp box, no hand labels."""
import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES

LAB = PROJECT / "data" / "labels"

# Frozen at the F1 peak of a 180-point sweep over bouts 4-7 (lamp ground truth),
# then applied unchanged to 8-14. The surface is flat -- everything in the top
# dozen scores 0.70-0.71 -- so the exact pick carries little weight.
# Do not re-fit these on a bout you then report a number for.
DT = 0.2        # s, resampling grid
BEFORE = 2.0    # s of fencing that has to precede a halt
AFTER = 3.0     # s of walking-back that has to follow it
PCT = 85        # keep scores above this percentile
GAP = 6.0       # s, minimum spacing between two halts
TOL = 4.0       # s, match radius against ground truth

LAMP_BOUTS = ("4", "5", "6", "7")
HAND_BOUTS = ("8", "9", "10", "11", "12", "14")  # 13 is the same bout as 9


def activity(stem, dt=DT):
    """Both fencers' fencing-class probability mass on a uniform time grid.

    Walking is excluded, not just neutral: after a halt the pair walk back to en
    garde, so walking is part of the quiet side of the step, not the busy side.
    """
    d = np.load(LAB / f"{stem}_probs_mirror.npz", allow_pickle=True)
    t, p, slot = np.asarray(d["time"]), np.asarray(d["probs"]), np.asarray(d["slot"])
    fence = [CLASS_NAMES.index(c) for c in ("advance", "lunge", "parry", "retreat")]
    grid = np.arange(0.0, float(t.max()) + dt, dt)
    per = []
    for s in ("A", "B"):
        m = slot == s
        if m.sum() < 2:
            continue
        o = np.argsort(t[m])
        per.append(np.interp(grid, t[m][o], p[np.flatnonzero(m)[o]][:, fence].sum(1)))
    return grid, np.mean(per, axis=0)


def score(grid, a, before=BEFORE, after=AFTER):
    """Step-down matched filter: how busy the 4 s before, how quiet the 5 s after."""
    dt = float(grid[1] - grid[0])
    nb, na = int(round(before / dt)), int(round(after / dt))
    c = np.concatenate([[0.0], np.cumsum(a)])
    i = np.arange(len(a))

    def win(lo, hi):
        lo, hi = np.clip(lo, 0, len(a)), np.clip(hi, 0, len(a))
        return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)

    s = win(i - nb, i) * (1.0 - win(i, i + na))
    ok = np.zeros(len(a), bool)
    ok[nb:len(a) - na] = True
    return s, ok


def find(grid, a, before=BEFORE, after=AFTER, pct=PCT, gap=GAP):
    """Peaks of the matched filter, kept greedily so no two sit within `gap`."""
    s, ok = score(grid, a, before, after)
    if not ok.any():
        return []
    thr = np.percentile(s[ok], pct)
    out = []
    for j in np.flatnonzero(ok & (s >= thr))[np.argsort(-s[ok & (s >= thr)])]:
        if all(abs(grid[j] - u) >= gap for u in out):
            out.append(float(grid[j]))
    return sorted(out)


def chance(grid, pred, true, real, n=1000, seed=0, gap=GAP, tol=TOL):
    """Recall of the same number of detections scattered at random.

    Detections are dense enough that a +/-4 s match window catches a fair share of
    halts by luck, so raw recall means little without this.
    """
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        got, guard = [], 0
        while len(got) < len(pred) and guard < 200 * max(len(pred), 1):
            guard += 1
            u = rng.uniform(0, float(grid[-1]))
            if all(abs(u - v) >= gap for v in got):
                got.append(u)
        h, _ = match(sorted(got), true, tol)
        out.append(h / len(true))
    out = np.array(out)
    return float(out.mean()), float((out >= real).mean())


def truth(stem):
    """Halt times: lamp onsets where a lamp box exists, hand table otherwise."""
    if stem in HAND_BOUTS:
        import exp_contested as C
        return [r[0] for r in C.read_contested(stem)], False
    import read_scoreboard as RS
    t, ser = RS.lamp_series("", RS.LAYOUT[stem]["lamp"], 0.1, LAB / f"{stem}_lamp.npz")
    thr = RS.lamp_all_thresholds(ser)
    return [h["t"] for h in RS.detect_halts(t, ser, thr)], True


def match(pred, true, tol=TOL):
    """Greedy nearest-match; returns (hits, matched pairs)."""
    used, pairs = set(), []
    for u in true:
        near = [(abs(u - v), k) for k, v in enumerate(pred)
                if k not in used and abs(u - v) <= tol]
        if near:
            _, k = min(near)
            used.add(k)
            pairs.append((u, pred[k]))
    return len(pairs), pairs


def report(stem, sims=1000):
    grid, a = activity(stem)
    pred = find(grid, a)
    true, exhaustive = truth(stem)
    hits, _ = match(pred, true)
    rec = hits / len(true) if true else float("nan")
    prec = hits / len(pred) if (exhaustive and pred) else float("nan")
    ch, p = chance(grid, pred, true, rec, sims) if sims else (float("nan"),) * 2
    return dict(stem=stem, n_true=len(true), n_pred=len(pred), hits=hits,
                recall=rec, precision=prec, exhaustive=exhaustive, chance=ch, p=p,
                minutes=float(grid[-1]) / 60.0, pred=pred)


def _self_test():
    """A square-wave bout: 5 s of action, 10 s of stillness, repeated."""
    dt = DT
    grid = np.arange(0, 300, dt)
    halts = np.arange(20.0, 290.0, 15.0)
    a = np.zeros_like(grid)
    for h in halts:
        a[(grid >= h - 5) & (grid < h)] = 1.0
    got = find(grid, a, gap=10.0)
    hits, _ = match(got, list(halts), tol=1.0)
    assert hits / len(halts) > 0.9, (hits, len(halts))
    assert hits / len(got) > 0.9, (hits, len(got))
    flat, ok = score(grid, np.ones_like(grid))
    assert np.ptp(flat[ok]) < 1e-9, np.ptp(flat[ok])
    print(f"  self-test ok: {hits}/{len(halts)} square-wave halts, "
          f"flat score on constant motion")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bouts", default=",".join(LAMP_BOUTS + HAND_BOUTS))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sims", type=int, default=1000,
                    help="random-placement draws for the chance baseline; 0 skips")
    ap.add_argument("--worklist", metavar="STEM",
                    help="write a halt-by-halt labelling skeleton for one bout")
    a = ap.parse_args()

    if a.self_test:
        return _self_test()

    if a.worklist:
        r = report(a.worklist, sims=0)
        known = {}
        if a.worklist in HAND_BOUTS:
            import exp_contested as C
            _, pairs = match(r["pred"], [u for u, _, _ in C.read_contested(a.worklist)])
            call = {u: p for u, p, _ in C.read_contested(a.worklist)}
            known = {v: call[u] for u, v in pairs}
        out = LAB / f"bout{a.worklist}_halts_EMPTY.tsv"
        lines = ["# EVERY halt the motion detector found in "
                 f"data/raw_video/{a.worklist}.mp4 -- single-light ones included.",
                 "# Rows already in bout{}_contested.tsv come pre-filled; the blanks are"
                 .format(a.worklist),
                 "# the single-light halts, which need the referee's call too.",
                 "# Set `real` to n if the row is a false alarm, and add a row for any",
                 "# halt the detector missed.",
                 "# Lights: left / right / both / mixed / none.  Call: left / right / none.",
                 "real\tTime stamp\tLights\tCall"]
        for u in r["pred"]:
            lights, cl = ("both", known[u] or "") if u in known else ("", "")
            lines.append(f"y\t{int(u) // 60}:{int(u) % 60:02d}\t{lights}\t{cl}")
        out.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {out} ({len(r['pred'])} candidate halts, "
              f"{len(known)} pre-filled from the contested table, "
              f"{r['minutes']:.1f} min)")
        return 0

    print(f"{'bout':<6}{'min':>7}{'halts':>7}{'found':>7}{'hit':>6}"
          f"{'recall':>9}{'chance':>8}{'p':>7}{'prec':>8}  source")
    rows = []
    for stem in a.bouts.split(","):
        r = report(stem, a.sims)
        rows.append(r)
        p = "    --" if np.isnan(r["precision"]) else f"{r['precision']:6.0%}"
        print(f"{stem:<6}{r['minutes']:7.1f}{r['n_true']:7d}{r['n_pred']:7d}"
              f"{r['hits']:6d}{r['recall']:9.0%}{r['chance']:8.0%}"
              f"{r['p']:7.3f}{p}  "
              f"{'lamp (tuning)' if r['exhaustive'] else 'hand table (held out)'}")
    for tag, sel in (("tuning  ", [r for r in rows if r["exhaustive"]]),
                     ("held out", [r for r in rows if not r["exhaustive"]])):
        if sel:
            print(f"  mean recall, {tag}: "
                  f"{np.mean([r['recall'] for r in sel]):.0%}"
                  f"   vs {np.mean([r['chance'] for r in sel]):.0%} by chance"
                  f"   (n={len(sel)} bouts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
