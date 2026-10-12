"""Internal-Verbal Divergence (IVD) -- the H3 measurement.

H3: the probe-decodable internal confidence direction survives longer than the
verbalised hedge, so internal-verbal divergence WIDENS across generations.

One number cannot carry that claim, so we report three, and they can disagree:

  ivd_l1        mean |p_internal - p_verbal| after aligning both to [0,1].
                Sensitive to level shifts (a model that just stops hedging).
  ivd_auroc_gap AUROC(internal) - AUROC(verbal). Level-invariant: measures the
                gap in how well each signal RANKS correctness. This is the
                cleanest form of the H3 claim.
  ivd_corr      Spearman correlation between the two signals. Falls toward 0 as
                verbalisation decouples from internals.

Report all three. If ivd_l1 widens but the AUROC gap does not, the model has
merely shifted its hedging rate, which is a weaker claim than decoupling.
"""
from __future__ import annotations

import numpy as np

from .calibration import auroc
from .probes import probe_scores_to_prob


def _rank_avg_ties(x: np.ndarray) -> np.ndarray:
    """Ranks with TIES AVERAGED.

    Naive ordinal ranking gives tied values distinct ranks, which manufactures
    correlation where none exists. That matters directly here: a collapsed model
    emits a near-constant marker confidence, so the verbal signal is almost all
    ties, and the uncorrected statistic reported spurious structure in exactly
    the regime H3 is about.
    """
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), dtype=float)
    r[order] = np.arange(1, len(x) + 1)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    return r


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    ra, rb = _rank_avg_ties(a), _rank_avg_ties(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    # A constant signal has zero rank variance -> correlation is undefined, not 0.
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def internal_verbal_divergence(records: list[dict], probe_res: dict | None = None,
                               internal_signal: str = "mean_token_prob",
                               verbal_signal: str = "marker_conf") -> dict:
    """Compute IVD for one generation.

    If probe_res carries out-of-fold scores, those are used as the internal
    signal (preferred -- it is the actual H3 quantity). Otherwise we fall back to
    token probability, which is a weaker proxy and is flagged as such.
    """
    uid_to_rec = {r["uid"]: r for r in records}

    internal, verbal, y, used = [], [], [], internal_signal
    if probe_res and probe_res.get("oof_scores") and probe_res.get("uids"):
        used = f"probe_layer_{probe_res.get('best_layer')}"
        probs = probe_scores_to_prob(np.array(probe_res["oof_scores"], dtype=float))
        for uid, p in zip(probe_res["uids"], probs):
            r = uid_to_rec.get(uid)
            if r is None or not np.isfinite(p):
                continue
            v = r.get(verbal_signal)
            if v is None:
                continue
            internal.append(float(p))
            verbal.append(float(v))
            y.append(int(r.get("correct_generated", r.get("correct", 0)) or 0))
    else:
        for r in records:
            iv, v = r.get(internal_signal), r.get(verbal_signal)
            if iv is None or v is None:
                continue
            internal.append(float(iv))
            verbal.append(float(v))
            y.append(int(r.get("correct_generated", r.get("correct", 0)) or 0))

    n = len(internal)
    if n < 20:
        return {"n": n, "error": "too few paired observations for IVD",
                "internal_source": used}

    ia, va, ya = np.array(internal), np.array(verbal), np.array(y)
    auc_i, auc_v = auroc(ia, ya), auroc(va, ya)
    return {
        "n": n,
        "internal_source": used,
        "verbal_source": verbal_signal,
        "probe_used": used.startswith("probe_"),
        "mean_internal": float(ia.mean()),
        "mean_verbal": float(va.mean()),
        "ivd_l1": float(np.mean(np.abs(ia - va))),
        "ivd_signed": float(np.mean(ia - va)),
        "ivd_corr": _spearman(ia, va),
        "auroc_internal": auc_i,
        "auroc_verbal": auc_v,
        "ivd_auroc_gap": (float(auc_i - auc_v)
                          if not (np.isnan(auc_i) or np.isnan(auc_v)) else float("nan")),
    }
