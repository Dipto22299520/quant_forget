"""Calibration metrics -- the paper's outcome variable.

Everything here takes (confidences, correctness) and returns scalars, plus
bootstrap CIs, because H1's onset test is "first generation where the metric
leaves generation-0's CI" and that is meaningless without the CI.
"""
from __future__ import annotations

import numpy as np

# The confidence signals we track separately across generations. Keeping them
# separate is the whole point: H3 is a claim about them coming apart.
CONF_SIGNALS = (
    "mean_token_prob",    # internal, length-normalised sequence probability
    "min_token_prob",     # internal, weakest step
    "marker_conf",        # verbalised, from the hedging lexicon
    "verbalized_conf",    # verbalised, explicit number
    "sc_agreement",       # sampling-based
    "p_pred",             # mcq: probability of the CHOSEN option -- proper confidence
    "p_correct",          # mcq: probability of the GOLD option -- CIRCULAR, diagnostic only
)

# Which correctness target each signal must be calibrated AGAINST.
#
# This pairing is not cosmetic. Marker confidence, token probabilities and
# self-consistency are properties of the GENERATED TEXT, so they must be scored
# against whether that text was right. p_correct comes from ranking the option
# letters in a separate forward pass, so it must be scored against the
# letter-scored answer. Calibrating a marker (a statement about the free-text
# answer) against letter-scored correctness would measure the disagreement
# between two different answers and report it as miscalibration.
SIGNAL_TARGET = {
    "mean_token_prob": "correct_generated",
    "min_token_prob": "correct_generated",
    "marker_conf": "correct_generated",
    "verbalized_conf": "correct_generated",
    "sc_agreement": "correct_generated",
    "p_pred": "correct_scored",
    "p_correct": "correct_scored",
}

# p_correct is the probability mass on the GOLD option, while correct_scored is
# whether the argmax IS the gold option. The former is maximal exactly when the
# latter is true, so calibrating one against the other is tautological: it yields
# AUROC 1.000 at every generation. Observed in the first MCQ pilot. Keep it for
# diagnostics, never report it as calibration.
CIRCULAR_SIGNALS = ("p_correct",)


def _target(records, field):
    """Correctness vector for `field`, or None if no record carries it."""
    if not any(field in r for r in records):
        return None
    return [int(r.get(field) or 0) for r in records]


def _clean(conf, correct) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(conf, dtype=float)
    y = np.asarray(correct, dtype=float)
    keep = ~(np.isnan(c) | np.isnan(y))
    return c[keep], y[keep]


def ece(conf, correct, n_bins: int = 15) -> float:
    """Expected Calibration Error, equal-WIDTH bins (the standard definition)."""
    c, y = _clean(conf, correct)
    if len(c) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / len(c)) * abs(y[m].mean() - c[m].mean())
    return float(total)


def adaptive_ece(conf, correct, n_bins: int = 15) -> float:
    """Equal-MASS bins. Robust when confidences pile up near 1.0, which is exactly
    what an overconfident collapsed model does -- so report this alongside ECE."""
    c, y = _clean(conf, correct)
    if len(c) == 0:
        return float("nan")
    n_bins = min(n_bins, max(1, len(c) // 10))
    order = np.argsort(c)
    c, y = c[order], y[order]
    total = 0.0
    for chunk_c, chunk_y in zip(np.array_split(c, n_bins), np.array_split(y, n_bins)):
        if len(chunk_c) == 0:
            continue
        total += (len(chunk_c) / len(c)) * abs(chunk_y.mean() - chunk_c.mean())
    return float(total)


def mce(conf, correct, n_bins: int = 15) -> float:
    """Maximum Calibration Error."""
    c, y = _clean(conf, correct)
    if len(c) == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    worst = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() < 5:          # tiny bins make MCE pure noise
            continue
        worst = max(worst, abs(y[m].mean() - c[m].mean()))
    return float(worst)


def brier(conf, correct) -> float:
    c, y = _clean(conf, correct)
    return float(np.mean((c - y) ** 2)) if len(c) else float("nan")


def overconfidence(conf, correct) -> float:
    """mean(confidence) - mean(accuracy). Signed: >0 means confidently wrong.

    This is the headline quantity for the paper's title. ECE is unsigned and
    would hide the direction of the failure.
    """
    c, y = _clean(conf, correct)
    return float(c.mean() - y.mean()) if len(c) else float("nan")


def auroc(conf, correct) -> float:
    """Discrimination: can this confidence signal rank correct above incorrect?

    Calibration and discrimination fail independently, and H3 predicts the probe
    keeps AUROC while the verbalised signal loses it. Tracking only ECE would
    miss that entirely.
    """
    c, y = _clean(conf, correct)
    if len(c) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    # Rank-based (Mann-Whitney), ties averaged.
    order = np.argsort(c)
    ranks = np.empty(len(c), dtype=float)
    ranks[order] = np.arange(1, len(c) + 1)
    # average ranks for ties
    sc = c[order]
    i = 0
    while i < len(sc):
        j = i
        while j + 1 < len(sc) and sc[j + 1] == sc[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    n_pos = float(y.sum())
    n_neg = float(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def reliability_bins(conf, correct, n_bins: int = 10) -> list[dict]:
    """Per-bin data for the reliability diagram."""
    c, y = _clean(conf, correct)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out = []
    if len(c) == 0:
        return out
    idx = np.clip(np.digitize(c, edges[1:-1], right=False), 0, n_bins - 1)
    for b in range(n_bins):
        m = idx == b
        out.append({
            "bin_lo": float(edges[b]), "bin_hi": float(edges[b + 1]),
            "n": int(m.sum()),
            "mean_conf": float(c[m].mean()) if m.any() else None,
            "accuracy": float(y[m].mean()) if m.any() else None,
        })
    return out


def bootstrap_ci(fn, conf, correct, n_boot: int = 1000, alpha: float = 0.05,
                 seed: int = 0) -> dict:
    """Percentile bootstrap CI for any of the metrics above."""
    c, y = _clean(conf, correct)
    if len(c) < 10:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "n": len(c)}
    rng = np.random.default_rng(seed)
    point = fn(c, y)
    stats = np.empty(n_boot)
    n = len(c)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = fn(c[idx], y[idx])
    stats = stats[~np.isnan(stats)]
    if len(stats) == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "n": n}
    return {"point": float(point),
            "lo": float(np.percentile(stats, 100 * alpha / 2)),
            "hi": float(np.percentile(stats, 100 * (1 - alpha / 2))),
            "n": n}


def calibration_report(records: list[dict], signals: tuple[str, ...] = CONF_SIGNALS,
                       n_bins: int = 15, n_boot: int = 0, seed: int = 0) -> dict:
    """Full calibration table for one model at one generation."""
    y = [r.get("correct") for r in records]
    y = [0 if v is None else int(v) for v in y]
    n = len(y)
    acc = float(np.mean(y)) if n else float("nan")

    y_gen = _target(records, "correct_generated") or y
    y_scored = _target(records, "correct_scored")

    report: dict = {
        "n": n,
        # Primary accuracy: format-independent letter scoring for MCQ, lenient
        # free-text matching for open QA.
        "accuracy": acc,
        # Both are reported so format acquisition is VISIBLE rather than being
        # mistaken for a knowledge change: a self-training loop that only
        # teaches output format moves accuracy_generated a long way while
        # accuracy_scored barely moves.
        "accuracy_generated": float(np.mean(y_gen)) if n else float("nan"),
        "accuracy_scored": (float(np.mean(y_scored))
                            if y_scored is not None else None),
        "format_gap": ((float(np.mean(y_scored)) - float(np.mean(y_gen)))
                       if (y_scored is not None and n) else None),
        "exact_match": float(np.mean([r.get("exact_match", 0) for r in records])) if n else float("nan"),
        "token_f1": float(np.mean([r.get("token_f1", 0.0) for r in records])) if n else float("nan"),
        "hedge_rate": float(np.mean([bool(r.get("hedged", False)) for r in records])) if n else float("nan"),
        # Tracked separately from hedging on purpose: collapse into abstention and
        # collapse into confident wrongness are different failures.
        "refusal_rate": float(np.mean([bool(r.get("refused", False)) for r in records])) if n else float("nan"),
        # Three different things, kept apart because they were once conflated:
        #   postprocessed_rate  post-processing changed the text (think block,
        #                       turn boundary, echoed cue, partial character)
        #   budget_hit_rate     the model was still going when max_new_tokens
        #                       ran out -- the response is a fragment
        #   context_trimmed_rate the passage was shortened to fit the window
        "postprocessed_rate": float(np.mean([bool(r.get("truncated", False)) for r in records])) if n else float("nan"),
        "budget_hit_rate": float(np.mean([bool(r.get("hit_budget", False)) for r in records])) if n else float("nan"),
        "context_trimmed_rate": float(np.mean([bool(r.get("context_trimmed", False)) for r in records])) if n else float("nan"),
        "mean_n_markers": float(np.mean([r.get("n_markers", 0) for r in records])) if n else float("nan"),
        "mean_gen_tokens": float(np.mean([r.get("n_gen_tokens", 0) for r in records])) if n else float("nan"),
        "signals": {},
    }

    for sig in signals:
        conf = [r.get(sig) for r in records]
        conf = [np.nan if v is None else float(v) for v in conf]
        tgt_name = SIGNAL_TARGET.get(sig, "correct")
        tgt = _target(records, tgt_name)
        if tgt is None:
            tgt = y_gen if tgt_name == "correct_generated" else y
        c, yy = _clean(conf, tgt)
        if len(c) < 10:
            continue
        entry = {
            "calibrated_against": tgt_name,
            "coverage": len(c) / n,
            "mean_conf": float(c.mean()),
            "ece": ece(c, yy, n_bins),
            "adaptive_ece": adaptive_ece(c, yy, n_bins),
            "mce": mce(c, yy, n_bins),
            "brier": brier(c, yy),
            "auroc": auroc(c, yy),
            "overconfidence": overconfidence(c, yy),
            "reliability": reliability_bins(c, yy, n_bins=10),
        }
        if n_boot:
            entry["ece_ci"] = bootstrap_ci(lambda a, b: ece(a, b, n_bins), c, yy,
                                           n_boot=n_boot, seed=seed)
            entry["auroc_ci"] = bootstrap_ci(auroc, c, yy, n_boot=n_boot, seed=seed)
            entry["overconf_ci"] = bootstrap_ci(overconfidence, c, yy,
                                                n_boot=n_boot, seed=seed)
        report["signals"][sig] = entry

    if n_boot:
        ones = np.ones(n)
        report["accuracy_ci"] = bootstrap_ci(lambda a, b: float(b.mean()), ones,
                                             np.array(y, dtype=float),
                                             n_boot=n_boot, seed=seed)
    return report


def accuracy_by_bucket(records: list[dict], key: str, bins: list[float]) -> list[dict]:
    """Accuracy split by a meta field -- used for PopQA popularity (tail knowledge)."""
    vals, ys = [], []
    for r in records:
        v = (r.get("meta") or {}).get(key)
        if v is None:
            continue
        vals.append(float(v))
        ys.append(int(r.get("correct", 0)))
    if not vals:
        return []
    vals, ys = np.array(vals), np.array(ys)
    out = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (vals >= lo) & (vals < hi)
        if not m.any():
            continue
        out.append({"lo": lo, "hi": hi, "n": int(m.sum()),
                    "accuracy": float(ys[m].mean())})
    return out
