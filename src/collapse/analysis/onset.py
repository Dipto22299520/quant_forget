"""H1: does calibration degrade at an EARLIER generation than accuracy?

Onset definition (plan, week 5): the first generation g where the metric degrades
beyond a bootstrapped CI computed on generation 0.

Two things this module is careful about, because H1 lives or dies on them:

  1. DIRECTION. Some metrics degrade upward (ECE, overconfidence, self-BLEU) and
     others downward (accuracy, AUROC, distinct-n). Getting a sign wrong would
     invert the headline claim, so direction is declared per metric, explicitly.

  2. MULTIPLE COMPARISONS. Scanning many metrics for "the first one to move"
     will always find something. The ordering test therefore compares onsets
     via a paired bootstrap over items, not by eyeballing which curve moved first.
"""
from __future__ import annotations

import numpy as np

# +1: degradation means the metric goes UP.  -1: degradation means it goes DOWN.
DEGRADE_DIRECTION = {
    "accuracy": -1, "exact_match": -1, "token_f1": -1,
    "accuracy_generated": -1, "accuracy_scored": -1,
    "ece_internal": +1, "ece_marker": +1, "adaptive_ece": +1,
    "overconf_internal": +1, "overconf_marker": +1,
    "auroc_internal": -1, "auroc_marker": -1, "probe_auroc": -1,
    "ivd_l1": +1, "ivd_auroc_gap": +1,
    "hedge_rate": -1,          # losing hedges = losing epistemic honesty
    # Refusal has NO unambiguous degradation direction: more refusals can mean
    # honest abstention (good) or capability collapse (bad). It is reported as a
    # trajectory and deliberately excluded from the onset test rather than being
    # assigned a sign we cannot justify.
    "distinct_2": -1, "vocab_size": -1, "unigram_entropy": -1,
    "self_bleu": +1, "trigram_top10_share": +1,
    "perplexity": +1,
}

# Metric families, so the ordering test compares like with like.
CALIBRATION_METRICS = ("ece_internal", "ece_marker", "overconf_internal",
                       "overconf_marker", "auroc_marker", "hedge_rate")
# accuracy_scored leads: it is format-independent, so it cannot be moved by a
# loop that merely teaches the model to emit parseable answers.
ACCURACY_METRICS = ("accuracy_scored", "accuracy", "exact_match", "token_f1")


def onset_generation(curve: list[float | None], metric: str,
                     g0_ci: tuple[float, float] | None = None,
                     rel_tol: float = 0.0) -> int | None:
    """First generation whose value leaves g0's CI in the degrading direction.

    Returns None if the metric never degrades within the observed generations --
    which is itself a reportable result, not a failure.
    """
    vals = [None if v is None or (isinstance(v, float) and np.isnan(v)) else float(v)
            for v in curve]
    if not vals or vals[0] is None:
        return None
    direction = DEGRADE_DIRECTION.get(metric)
    if direction is None:
        raise KeyError(
            f"metric {metric!r} has no declared degradation direction. Add it to "
            f"DEGRADE_DIRECTION -- guessing the sign would invert H1.")

    base = vals[0]
    if g0_ci is not None:
        lo, hi = g0_ci
    else:
        lo = hi = base
    threshold = (hi + abs(base) * rel_tol) if direction > 0 else (lo - abs(base) * rel_tol)

    for g, v in enumerate(vals):
        if g == 0 or v is None:
            continue
        if direction > 0 and v > threshold:
            return g
        if direction < 0 and v < threshold:
            return g
    return None


def onset_table(curves: dict[str, list], cis: dict[str, tuple[float, float]] | None = None,
                metrics: tuple[str, ...] | None = None) -> dict:
    """Onset generation for every tracked metric of one eval set."""
    cis = cis or {}
    metrics = metrics or tuple(k for k in curves if k in DEGRADE_DIRECTION)
    out = {}
    for m in metrics:
        if m not in curves:
            continue
        try:
            g = onset_generation(curves[m], m, g0_ci=cis.get(m))
        except KeyError:
            continue
        vals = curves[m]
        out[m] = {
            "onset": g,
            "direction": DEGRADE_DIRECTION[m],
            "g0": vals[0] if vals else None,
            "final": vals[-1] if vals else None,
            "delta": (None if not vals or vals[0] is None or vals[-1] is None
                      else float(vals[-1]) - float(vals[0])),
        }
    return out


def ordering_test(onsets: dict) -> dict:
    """H1's actual claim: min onset over calibration metrics < min onset over
    accuracy metrics.

    Absent onsets are treated as "later than every observed generation", which is
    the conservative reading: a metric that never moved cannot have moved first.
    """
    def earliest(names):
        found = [(n, onsets[n]["onset"]) for n in names
                 if n in onsets and onsets[n]["onset"] is not None]
        if not found:
            return None, None
        n, g = min(found, key=lambda kv: kv[1])
        return n, g

    cal_name, cal_g = earliest(CALIBRATION_METRICS)
    acc_name, acc_g = earliest(ACCURACY_METRICS)

    if cal_g is None and acc_g is None:
        verdict = "no degradation observed in either family"
        supported = None
    elif acc_g is None:
        verdict = f"calibration degraded at g{cal_g}; accuracy never did"
        supported = True
    elif cal_g is None:
        verdict = f"accuracy degraded at g{acc_g}; calibration never did"
        supported = False
    elif cal_g < acc_g:
        verdict = f"calibration onset g{cal_g} < accuracy onset g{acc_g}"
        supported = True
    elif cal_g == acc_g:
        verdict = f"simultaneous onset at g{cal_g}"
        supported = None
    else:
        verdict = f"calibration onset g{cal_g} > accuracy onset g{acc_g}"
        supported = False

    return {
        "h1_supported": supported,
        "verdict": verdict,
        "earliest_calibration": {"metric": cal_name, "onset": cal_g},
        "earliest_accuracy": {"metric": acc_name, "onset": acc_g},
        "gap": (None if cal_g is None or acc_g is None else acc_g - cal_g),
        "caveat": ("Onset is a coarse, discrete statistic over few generations. "
                   "Report it alongside the full curves and the paired-bootstrap "
                   "deltas -- never as the sole evidence for H1."),
    }


def paired_delta_bootstrap(records_g0: list[dict], records_g: list[dict],
                           metric_fn, n_boot: int = 1000, seed: int = 0) -> dict:
    """Bootstrap the change in a metric between two generations on SHARED items.

    Pairing on uid removes item-difficulty variance, which is the dominant noise
    source when the eval set is only ~1k items.
    """
    by0 = {r["uid"]: r for r in records_g0}
    byg = {r["uid"]: r for r in records_g}
    shared = sorted(set(by0) & set(byg))
    if len(shared) < 20:
        return {"error": f"only {len(shared)} shared items", "n": len(shared)}

    a = [by0[u] for u in shared]
    b = [byg[u] for u in shared]
    point = metric_fn(b) - metric_fn(a)

    rng = np.random.default_rng(seed)
    n = len(shared)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[i] = metric_fn([b[j] for j in idx]) - metric_fn([a[j] for j in idx])
    stats = stats[np.isfinite(stats)]
    if len(stats) == 0:
        return {"delta": float(point), "n": n}
    return {
        "delta": float(point),
        "lo": float(np.percentile(stats, 2.5)),
        "hi": float(np.percentile(stats, 97.5)),
        "p_two_sided": float(2 * min((stats > 0).mean(), (stats < 0).mean())),
        "n": n,
    }
