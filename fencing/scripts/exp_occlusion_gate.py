"""Is arm visibility a usable abstention signal? Held-out only.

usage: py -3 scripts/exp_occlusion_gate.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from src.action_model import ActionLSTM, N_AGG_WIDE, load_action_model
from exp_opponent import with_opponent

CONT = PROJECT / "data" / "train_continuous"
MODELS = PROJECT / "models"
STEMS = ("1", "4", "5", "7")

ARM_L = (13, 15, 17, 19, 21)      # elbow, wrist, pinky, index, thumb
ARM_R = (14, 16, 18, 20, 22)



def visibility(stem):
    """Per-window mean visibility of each arm, over the window's non-blank frames."""
    X = np.load(CONT / f"{stem}.npz")["X"]
    vis = X.reshape(len(X), X.shape[1], 33, 4)[..., 3]
    real = vis.max(axis=2) > 0
    n = np.maximum(real.sum(axis=1), 1)
    out = {}
    for tag, idx in (("left", ARM_L), ("right", ARM_R)):
        out[tag] = (vis[:, :, idx].mean(axis=2) * real).sum(axis=1) / n
    out["worse"] = np.minimum(out["left"], out["right"])
    out["mean"] = (out["left"] + out["right"]) / 2
    return out


def predict(stem, device):
    path = MODELS / f"verify_mirror_h{stem}.pth"
    if not path.exists():
        return None
    model = load_action_model(path, device=device,
                              cls=lambda: ActionLSTM(pool="last", n_agg=N_AGG_WIDE))
    d = with_opponent(CONT / f"{stem}.npz")[0]
    out = []
    with torch.no_grad():
        for i in range(0, len(d["X"]), 256):
            out.append(model(torch.from_numpy(d["X"][i:i + 256]).to(device),
                             torch.from_numpy(d["wide"][i:i + 256]).to(device),
                             torch.from_numpy(d["lengths"][i:i + 256]).to(device)
                             ).argmax(1).cpu())
    return torch.cat(out).numpy() == d["y"]


def auc(score, ok):
    """P(score higher on a correct window than an incorrect one)."""
    r = np.argsort(np.argsort(score)) + 1
    n1, n0 = int(ok.sum()), int((~ok).sum())
    if not n1 or not n0:
        return float("nan")
    return (r[ok].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default=None, help="force the gating feature")
    a = ap.parse_args()

    device = torch.device("cpu")
    ok, feats, bout = [], {k: [] for k in ("left", "right", "worse", "mean")}, []
    for s in STEMS:
        got = predict(s, device)
        if got is None:
            print(f"missing verify_mirror_h{s}.pth -- train it first")
            return 1
        v = visibility(s)
        ok.append(got)
        bout.append(np.full(len(got), s))
        for k in feats:
            feats[k].append(v[k])
    ok = np.concatenate(ok)
    bout = np.concatenate(bout)
    feats = {k: np.concatenate(v) for k, v in feats.items()}

    print(f"held-out windows {len(ok)}, overall accuracy {ok.mean():.1%}\n")
    print("  which visibility signal predicts a correct call?")
    print(f"  {'feature':<10}{'AUC':>8}")
    for k, v in feats.items():
        print(f"  {k:<10}{auc(v, ok):>8.3f}")

    best = a.variant or max(feats, key=lambda k: auc(feats[k], ok))
    f = feats[best]
    print(f"\n  gating on {best!r} arm visibility")
    qs = np.quantile(f, [0.25, 0.5, 0.75])
    print(f"  {'band':<22}{'n':>7}{'accuracy':>11}")
    edges = [-np.inf, *qs, np.inf]
    names = ["lowest quartile", "second", "third", "highest quartile"]
    for lo, hi, nm in zip(edges, edges[1:], names):
        m = (f >= lo) & (f < hi)
        if m.sum():
            print(f"  {nm:<22}{m.sum():>7}{ok[m].mean():>11.1%}")

    print(f"\n  abstention sweep -- suppress the label below the threshold")
    print(f"  {'thresh':>8}{'quiet':>8}{'% of all':>10}{'wrong cut':>11}"
          f"{'right lost':>12}{'ratio':>8}{'acc shown':>11}")
    for th in np.quantile(f, [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]):
        quiet = f < th
        wrong_cut, right_lost = int((quiet & ~ok).sum()), int((quiet & ok).sum())
        shown = ok[~quiet]
        ratio = wrong_cut / right_lost if right_lost else float("inf")
        print(f"  {th:>8.3f}{quiet.sum():>8}{quiet.mean():>9.0%}{wrong_cut:>11}"
              f"{right_lost:>12}{ratio:>8.2f}{shown.mean():>11.1%}")

    print("\n  per bout, at the median threshold")
    th = float(np.median(f))
    print(f"  {'bout':<8}{'n':>7}{'acc all':>10}{'acc shown':>11}{'quiet':>8}")
    for s in STEMS:
        m = bout == s
        q = m & (f < th)
        print(f"  {s:<8}{m.sum():>7}{ok[m].mean():>10.1%}"
              f"{ok[m & ~(f < th)].mean():>11.1%}{q.sum() / m.sum():>8.0%}")
    print("\nA ratio above 1.0 means the gate removes more wrong calls than right ones.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
