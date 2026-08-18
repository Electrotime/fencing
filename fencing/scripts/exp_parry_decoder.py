"""Learned decision rule over both fencers' probability vectors vs the hand gate. See CLAUDE.md."""
import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import CLASS_NAMES
from src.labels import load_intervals, UNSCORABLE
import sweep_parry_promote as S

LAB = PROJECT / "data" / "labels"
PARRY_I = CLASS_NAMES.index("parry")


def _lags(slot, time, opp_lunge, own_parry):
    """Opponent lunge and own parry a few windows earlier, per slot, in time order."""
    n = len(time)
    out = np.zeros((n, 4), dtype=np.float32)
    for s in ("A", "B"):
        idx = np.flatnonzero(slot == s)
        idx = idx[np.argsort(time[idx])]
        ol, op = opp_lunge[idx], own_parry[idx]
        for k, lag in enumerate((1, 2, 3)):
            sh = np.concatenate([np.zeros(lag, np.float32), ol[:-lag]]) if lag < len(ol) else np.zeros_like(ol)
            out[idx, k] = sh
        prev = np.concatenate([np.zeros(1, np.float32), op[:-1]]) if len(op) > 1 else np.zeros_like(op)
        out[idx, 3] = prev
    return out


def load_pairs(stem):
    """Own probs, opponent probs and truth for every scorable window of one bout."""
    cache, csv_name, prov = S.MIRRORED[stem]
    d = np.load(LAB / cache)
    slot, time, probs = d["slot"].astype(str), d["time"], d["probs"]
    index = {(s, float(t)): i for i, (s, t) in enumerate(zip(slot, time))}
    opp = np.zeros_like(probs)
    seen = np.zeros(len(time), dtype=bool)
    for i, (s, t) in enumerate(zip(slot, time)):
        j = index.get(("B" if s == "A" else "A", float(t)))
        if j is not None:
            opp[i] = probs[j]
            seen[i] = True
    lag = _lags(slot, time, opp[:, S.LUNGE_I], probs[:, PARRY_I])
    truth, _ = load_intervals(LAB / csv_name)

    def at(s, t):
        for st, en, lab in truth.get(s, []):
            if st <= t < en:
                return lab
        return None

    keep, y = [], []
    for i, (s, t) in enumerate(zip(slot, time)):
        lab = at(s, float(t))
        if lab is None or lab in UNSCORABLE:
            continue
        keep.append(i)
        y.append(lab)
    keep = np.array(keep, dtype=int)
    return probs[keep], opp[keep], seen[keep], np.array(y), lag[keep], prov


def features(own, opp, seen, lag, kind):
    if kind == "gate2":
        return np.stack([own[:, PARRY_I], opp[:, S.LUNGE_I]], axis=1)
    if kind == "own6":
        return own
    if kind == "temporal":
        return np.concatenate([own[:, [PARRY_I]], opp[:, [S.LUNGE_I]], lag], axis=1)
    if kind == "temporal13":
        return np.concatenate([own, opp, seen[:, None].astype(np.float32), lag], axis=1)
    return np.concatenate([own, opp, seen[:, None].astype(np.float32)], axis=1)


def best_threshold(score, y):
    """Threshold maximising F1. Call this on TRAINING data only."""
    best, bt = -1.0, 0.5
    for t in np.unique(np.round(np.quantile(score, np.linspace(0.5, 0.999, 120)), 4)):
        pred = score >= t
        tp = int((pred & y).sum())
        if not tp:
            continue
        p, r = tp / int(pred.sum()), tp / int(y.sum())
        f = 2 * p * r / (p + r)
        if f > best:
            best, bt = f, float(t)
    return bt


def at_threshold(score, y, t):
    pred = score >= t
    tp = int((pred & y).sum())
    if not tp:
        return dict(f1=0.0, p=0.0, r=0.0, n=int(pred.sum()))
    p, r = tp / int(pred.sum()), tp / int(y.sum())
    return dict(f1=2 * p * r / (p + r), p=p, r=r, n=int(pred.sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stems", default="4,7,5,1")
    a = ap.parse_args()
    stems = a.stems.split(",")
    data = {s: load_pairs(s) for s in stems}

    print("Leave-one-bout-out. Decoder AND its threshold come from the OTHER bouts;\n"
          "nothing is fitted or tuned on the bout being scored.\n")
    print(f"  {'holdout':<9}{'rule':<18}{'F1':>7}{'P':>7}{'R':>7}{'n_pred':>8}")
    agg = {}
    for h in stems:
        own, opp, seen, ylab, lag, prov = data[h]
        y = (ylab == "parry")
        if y.sum() < 20:
            print(f"  bout {h}: {int(y.sum())} parry windows, skipped")
            continue
        gate = S.score(S.decide(own, opp[:, S.LUNGE_I], 0.15, 0.60), ylab)
        print(f"  {h:<9}{'shipped gate':<18}{gate['f1']:>7.2f}{gate['prec']:>7.0%}"
              f"{gate['rec']:>7.0%}{gate['n_pred']:>8}")
        agg.setdefault("shipped gate", []).append(gate["f1"])

        tr = [s for s in stems if s != h]
        # control: the SAME hand rule, thresholds re-tuned on the training bouts only
        best, bp = -1.0, (0.15, 0.60)
        for pm in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
            for om in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80):
                f = np.mean([S.score(S.decide(data[s][0], data[s][1][:, S.LUNGE_I], pm, om),
                                     data[s][3])["f1"] for s in tr])
                if f > best:
                    best, bp = f, (pm, om)
        rt = S.score(S.decide(own, opp[:, S.LUNGE_I], *bp), ylab)
        print(f"  {'':<9}{'hand retuned':<18}{rt['f1']:>7.2f}{rt['prec']:>7.0%}"
              f"{rt['rec']:>7.0%}{rt['n_pred']:>8}   {bp}")
        agg.setdefault("hand retuned", []).append(rt["f1"])
        for kind in ("gate2", "own6", "full13", "temporal", "temporal13"):
            Xtr = np.concatenate([features(data[s][0], data[s][1], data[s][2], data[s][4], kind) for s in tr])
            ytr = np.concatenate([(data[s][3] == "parry") for s in tr])
            clf = LogisticRegression(max_iter=2000, class_weight="balanced")
            clf.fit(Xtr, ytr)
            t = best_threshold(clf.predict_proba(Xtr)[:, 1], ytr)   # chosen on TRAIN
            c = at_threshold(clf.predict_proba(features(own, opp, seen, lag, kind))[:, 1], y, t)
            print(f"  {'':<9}{'learned ' + kind:<18}{c['f1']:>7.2f}{c['p']:>7.0%}"
                  f"{c['r']:>7.0%}{c['n']:>8}")
            agg.setdefault("learned " + kind, []).append(c["f1"])
        print()

    print(f"  {'rule':<20}{'mean F1':>9}")
    for k, v in agg.items():
        print(f"  {k:<20}{np.mean(v):>9.2f}")
    print("\n'P at gate recall' is the honest comparison: same recall, whose precision wins.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
