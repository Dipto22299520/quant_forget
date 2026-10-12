r"""Label-free calibration tracking under recursive self-training.

    .venv\Scripts\python.exe scripts\label_free.py

The loop sharpens the model: confidence rises while accuracy and AUROC stay flat.
If the drift is a temperature shift, then

  fixed      a temperature T0 fitted ONCE, on labelled data at g0, goes stale;
  tracked    re-fitting the temperature WITHOUT labels -- by matching the mean
             predictive entropy on unlabelled inputs to its calibrated g0 value --
             follows the drift;
  oracle     re-fitting on labelled data every round is the upper bound.

Two-fold cross-fitting: temperatures are fitted on one half of the items and
ECE is measured on the other, then the halves swap. `tracked` uses only the
unlabelled inputs of round g (all items) plus T0.

Also reports P1: does the entropy of a round's self-generated training targets
forecast the next round's calibration change?

Writes paper/label_free.json. No GPU.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from collapse.utils import read_jsonl        # noqa: E402
from paper_assets import discover, ece_np    # noqa: E402

RUNS = ROOT / "runs"
LOG_T = np.linspace(np.log(0.05), np.log(50.0), 400)


def softmax_t(L: np.ndarray, T: float) -> np.ndarray:
    z = L / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(L, y, T):
    P = softmax_t(L, T)
    return float(-np.mean(np.log(P[np.arange(len(y)), y] + 1e-12)))


def fit_t(L, y) -> float:
    """Oracle temperature: grid in log-space, then local refinement."""
    vals = [nll(L, y, np.exp(t)) for t in LOG_T]
    i = int(np.argmin(vals))
    lo, hi = LOG_T[max(0, i - 1)], LOG_T[min(len(LOG_T) - 1, i + 1)]
    for _ in range(40):                      # golden-section on the bracket
        a = hi - (hi - lo) * 0.618
        b = lo + (hi - lo) * 0.618
        if nll(L, y, np.exp(a)) < nll(L, y, np.exp(b)):
            hi = b
        else:
            lo = a
    return float(np.exp((lo + hi) / 2))


def mean_entropy(L, T) -> float:
    P = softmax_t(L, T)
    return float(-(P * np.log(P + 1e-12)).sum(1).mean())


def match_t(L, target: float) -> float:
    """Temperature at which mean predictive entropy equals `target` (monotone in T)."""
    lo, hi = np.log(0.05), np.log(50.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        if mean_entropy(L, np.exp(mid)) < target:
            lo = mid
        else:
            hi = mid
    return float(np.exp((lo + hi) / 2))


def ece_at(L, y, T) -> float:
    P = softmax_t(L, T)
    return ece_np(P.max(1), (P.argmax(1) == y).astype(float))


def _clean(L: np.ndarray) -> np.ndarray:
    """Replace -inf (an option letter with no single-token encoding) by a very low score."""
    L = L.copy()
    for i in range(len(L)):
        fin = np.isfinite(L[i])
        if not fin.all():
            L[i, ~fin] = L[i, fin].min() - 50.0
    return L


def round_logits(arm, g):
    """Debiased option logits for round g, and whether they are EXACT.

    Preference: the run's own records (code >= 4cd0e20a5c4f stores them) >
    the endpoint re-score from saved weights (g0/g5, scripts/rescore_raw.py) >
    log of the stored float32 probabilities, which saturate at 1.0 and so cap
    every logit gap at 27.6 nats (the 1e-12 clip). Only the last is inexact.
    """
    recs = read_jsonl(RUNS / arm["name"] / f"gen{g}" / f"eval_{arm['eval_set']}.jsonl")
    y = np.array([r["answer_idx"] for r in recs], dtype=int)
    uids = [r["uid"] for r in recs]
    if all(r.get("choice_logits") for r in recs):
        return _clean(np.array([r["choice_logits"] for r in recs], float)), y, uids, True
    if g in (0, 5):
        f = ROOT / "paper" / "raw_scores" / f"{arm['name']}_g{g}.jsonl"
        if f.exists():
            rs = {x["uid"]: x for x in read_jsonl(f)}
            if all(rs.get(u, {}).get("choice_logits") for u in uids):
                return _clean(np.array([rs[u]["choice_logits"] for u in uids], float)), y, uids, True
    L = np.log(np.clip(np.array([r["choice_probs"] for r in recs], dtype=float), 1e-12, 1.0))
    return L, y, uids, False


def load_round(arm, g):
    L, y, uids, _ = round_logits(arm, g)
    return L, y, uids


def log_conf(L: np.ndarray) -> np.ndarray:
    """log p(chosen option): monotone in p_pred but never saturates -- rank on this."""
    m = L.max(1)
    return m - (m + np.log(np.exp(L - m[:, None]).sum(1)))


def analyse(arm, seed=0):
    rounds = [load_round(arm, g) for g in range(6)]
    uids0 = rounds[0][2]
    assert all(r[2] == uids0 for r in rounds), f"{arm['name']}: item order differs across rounds"
    n = len(uids0)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = [(perm[: n // 2], perm[n // 2:]), (perm[n // 2:], perm[: n // 2])]
    out = {k: np.zeros(6) for k in ("raw", "fixed", "tracked", "oracle")}
    temps = {k: np.zeros(6) for k in ("fixed", "tracked", "oracle")}
    for fit_idx, ev_idx in folds:
        L0, y0, _ = rounds[0]
        T0 = fit_t(L0[fit_idx], y0[fit_idx])
        target = mean_entropy(L0, T0)            # unlabelled: all g0 inputs at T0
        for g, (L, y, _) in enumerate(rounds):
            Tt = match_t(L, target)              # unlabelled: all round-g inputs
            To = fit_t(L[fit_idx], y[fit_idx])
            out["raw"][g] += ece_at(L[ev_idx], y[ev_idx], 1.0) / 2
            out["fixed"][g] += ece_at(L[ev_idx], y[ev_idx], T0) / 2
            out["tracked"][g] += ece_at(L[ev_idx], y[ev_idx], Tt) / 2
            out["oracle"][g] += ece_at(L[ev_idx], y[ev_idx], To) / 2
            temps["fixed"][g] += T0 / 2
            temps["tracked"][g] += Tt / 2
            temps["oracle"][g] += To / 2
    return {k: v.tolist() for k, v in out.items()}, {k: v.tolist() for k, v in temps.items()}


def forecast(arms_out):
    """P1: self-target entropy of round g vs raw-ECE change g -> g+1."""
    xs, ys, groups = [], [], []
    for a in arms_out:
        syn = a.get("syn_entropy")
        if not syn or len(syn) < 5:
            continue
        raw = a["ece"]["raw"]
        for g in range(5):
            if syn[g] is None:
                continue
            xs.append(np.log(syn[g] / syn[0]) if syn[0] else np.nan)   # entropy lost so far
            ys.append(raw[g + 1] - raw[0])                              # calibration drift so far
            groups.append(a["name"])
    xs, ys = np.array(xs), np.array(ys)
    ok = np.isfinite(xs) & np.isfinite(ys)
    from scipy.stats import spearmanr
    rho, p = spearmanr(xs[ok], ys[ok])
    return {"n_points": int(ok.sum()), "spearman_rho": float(rho), "p": float(p),
            "note": "x = log(H_g / H_0) of self-generated targets; y = raw ECE(g+1) - ECE(0)"}


def main():
    arms = [a for a in discover() if a["valid"] and a["lang"] == "en"]
    res = []
    print(f"{'arm':26} {'size':4}  {'raw  g0->g5':>14} {'fixed T0':>16} {'tracked (no labels)':>22} {'oracle':>14}  closed")
    for a in arms:
        ece, temps = analyse(a)
        syn = []
        for g in range(5):
            f = RUNS / a["name"] / f"gen{g}" / "synthetic.jsonl"
            if f.exists():
                R = read_jsonl(f)
                v = [r["mean_entropy"] for r in R if r.get("mean_entropy") is not None]
                syn.append(float(np.mean(v)) if v else None)
            else:
                syn.append(None)
        gap = ece["fixed"][5] - ece["oracle"][5]
        closed = (ece["fixed"][5] - ece["tracked"][5]) / gap if gap > 1e-4 else float("nan")
        res.append({"name": a["name"], "size": a["size"], "cond": a["cond"], "seed": a["seed"],
                    "ece": ece, "temps": temps, "syn_entropy": syn, "gap_closed_g5": closed})
        f = lambda k: f"{ece[k][0]:.3f}->{ece[k][5]:.3f}"
        print(f"{a['name']:26} {a['size']:4}  {f('raw'):>14} {f('fixed'):>16} {f('tracked'):>22} {f('oracle'):>14}  {closed:6.0%}")
    p1 = forecast([r for r in res if r["size"] == "1.7B"])
    p1_8 = forecast([r for r in res if r["size"] == "8B"])
    print(f"\nP1 (1.7B): spearman(log H_g/H_0, ECE drift) = {p1['spearman_rho']:+.2f}  p={p1['p']:.3g}  n={p1['n_points']}")
    print(f"P1 (8B):   spearman = {p1_8['spearman_rho']:+.2f}  p={p1_8['p']:.3g}  n={p1_8['n_points']}")
    (ROOT / "paper" / "label_free.json").write_text(
        json.dumps({"arms": res, "p1_1.7B": p1, "p1_8B": p1_8}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
