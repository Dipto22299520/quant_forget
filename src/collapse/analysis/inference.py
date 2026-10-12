"""Pre-declared statistical analysis (audit finding #4).

The previous onset procedure had four defects, each of which inflates the chance
of declaring an effect that is not there:

  1. No baseline interval. `onset_table` was called without CIs, so the threshold
     was the generation-0 POINT estimate and any adverse wiggle produced an onset.
  2. No multiple-testing control. `ordering_test` scans a family and takes the
     earliest mover, which is a minimum over many correlated tests.
  3. No paired item-level analysis. `paired_delta_bootstrap` existed but nothing
     called it, so item difficulty stayed in the noise term.
  4. An absolute 0.10 accuracy floor, which sits BELOW the 0.25 chance level of a
     four-option MCQ.

Everything here is declared BEFORE looking at a run, so the analysis cannot be
tuned to the result. Change these constants deliberately and record the change.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------- declarations

# One primary endpoint per task type. Secondary metrics may be reported, but only
# the primary decides the headline claim.
PRIMARY = {
    "mcq": {
        "confidence": "p_pred",            # probability of the CHOSEN option
        "correctness": "correct_scored",   # format-independent letter scoring
        "accuracy": "accuracy_scored",
    },
    "open_qa": {
        "confidence": "mean_token_prob",
        "correctness": "correct_generated",
        "accuracy": "accuracy",
    },
}

# The calibration family for H1's ordering test. Fixed in advance; Holm correction
# is applied across exactly this set.
CALIBRATION_FAMILY = ("ece", "overconfidence", "auroc")

ALPHA = 0.05          # two-sided
N_BOOT = 2000
# Minimum detectable effect: changes smaller than this are not claimed even when
# an interval excludes zero. Prevents "significant but meaningless" findings.
MDE = {"ece": 0.02, "overconfidence": 0.02, "auroc": 0.02, "accuracy": 0.02}
# Accuracy must clear chance by this margin at g0, or the ordering claim is
# untestable because accuracy has nowhere to fall.
FLOOR_MARGIN = 0.05


def chance_level(records: list[dict]) -> float:
    """1/n_options for MCQ, 0 for open QA. The floor must be relative to this."""
    ns = [len(r["choices"]) for r in records
          if r.get("task_type") == "mcq" and r.get("choices")]
    if not ns:
        return 0.0
    return 1.0 / float(np.median(ns))


def task_type(records: list[dict]) -> str:
    return "mcq" if any(r.get("task_type") == "mcq" for r in records) else "open_qa"


# ------------------------------------------------------------------ statistics

def holm_bonferroni(pvals: dict[str, float], alpha: float = ALPHA) -> dict[str, dict]:
    """Holm step-down. Controls family-wise error across the declared family."""
    items = sorted(((k, v) for k, v in pvals.items() if np.isfinite(v)),
                   key=lambda kv: kv[1])
    m = len(items)
    out, rejected_so_far = {}, True
    for i, (k, p) in enumerate(items):
        thresh = alpha / (m - i)
        reject = bool(rejected_so_far and p <= thresh)
        rejected_so_far = reject
        out[k] = {"p": float(p), "threshold": float(thresh), "significant": reject}
    for k, v in pvals.items():
        out.setdefault(k, {"p": float(v), "threshold": None, "significant": False})
    return out


def paired_delta(recs_a: list[dict], recs_b: list[dict], stat_fn,
                 n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """Bootstrap the change in `stat_fn` between two generations on SHARED items.

    Pairing on uid removes item difficulty, which dominates the noise at n<=1000.
    Returns the delta, its CI, and a two-sided bootstrap p-value.
    """
    by_a = {r["uid"]: r for r in recs_a}
    by_b = {r["uid"]: r for r in recs_b}
    shared = sorted(set(by_a) & set(by_b))
    if len(shared) < 30:
        return {"delta": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "p": float("nan"), "n": len(shared),
                "error": f"only {len(shared)} shared items"}

    a = [by_a[u] for u in shared]
    b = [by_b[u] for u in shared]
    point = stat_fn(b) - stat_fn(a)

    rng = np.random.default_rng(seed)
    n = len(shared)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot[i] = stat_fn([b[j] for j in idx]) - stat_fn([a[j] for j in idx])
    boot = boot[np.isfinite(boot)]
    if len(boot) == 0:
        return {"delta": float(point), "n": n, "p": float("nan"),
                "lo": float("nan"), "hi": float("nan")}

    # Two-sided bootstrap p: how often does the resampled delta cross zero?
    p = 2.0 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"delta": float(point),
            "lo": float(np.percentile(boot, 100 * ALPHA / 2)),
            "hi": float(np.percentile(boot, 100 * (1 - ALPHA / 2))),
            "p": float(min(1.0, p)), "n": n}


def onset_paired(records_by_gen: list[list[dict]], stat_fn, direction: int,
                 mde: float, seed: int = 0) -> dict:
    """First generation whose PAIRED change from g0 is significant and large enough.

    direction: +1 if degradation means the statistic increases, -1 if it decreases.
    Returns onset (or None = right-censored), plus the per-generation deltas so
    ties and near-misses are visible rather than collapsed into one number.
    """
    if len(records_by_gen) < 2:
        return {"onset": None, "censored": True, "per_generation": [],
                "reason": "fewer than two generations"}

    g0 = records_by_gen[0]
    per_gen, onset = [], None
    for g in range(1, len(records_by_gen)):
        d = paired_delta(g0, records_by_gen[g], stat_fn, seed=seed)
        signed = d["delta"] * direction
        excludes_zero = (np.isfinite(d.get("lo", np.nan))
                         and (d["lo"] > 0 if direction > 0 else d["hi"] < 0))
        big_enough = np.isfinite(signed) and signed >= mde
        d.update({"generation": g, "degrading": bool(signed > 0),
                  "excludes_zero": bool(excludes_zero),
                  "exceeds_mde": bool(big_enough),
                  "onset_here": bool(excludes_zero and big_enough)})
        per_gen.append(d)
        if onset is None and d["onset_here"]:
            onset = g

    return {"onset": onset,
            # Right-censoring is a real outcome: "never degraded within the
            # generations we ran" is different from "degraded late", and
            # conflating them biases any ordering comparison.
            "censored": onset is None,
            "per_generation": per_gen,
            "p_min": float(min((d["p"] for d in per_gen
                                if np.isfinite(d.get("p", np.nan))), default=np.nan))}


# -------------------------------------------------------------- stat functions

def stat_accuracy(field: str):
    def f(recs):
        v = [r.get(field) for r in recs]
        v = [0 if x is None else int(x) for x in v]
        return float(np.mean(v)) if v else float("nan")
    return f


def stat_ece(conf_field: str, corr_field: str, n_bins: int = 15):
    from ..eval.calibration import ece

    def f(recs):
        c = [r.get(conf_field) for r in recs]
        y = [r.get(corr_field) for r in recs]
        pairs = [(float(a), int(b)) for a, b in zip(c, y)
                 if a is not None and b is not None]
        if len(pairs) < 10:
            return float("nan")
        return ece([p[0] for p in pairs], [p[1] for p in pairs], n_bins)
    return f


def stat_overconfidence(conf_field: str, corr_field: str):
    def f(recs):
        c = [r.get(conf_field) for r in recs]
        y = [r.get(corr_field) for r in recs]
        pairs = [(float(a), int(b)) for a, b in zip(c, y)
                 if a is not None and b is not None]
        if len(pairs) < 10:
            return float("nan")
        return float(np.mean([p[0] for p in pairs]) - np.mean([p[1] for p in pairs]))
    return f


def stat_auroc(conf_field: str, corr_field: str):
    from ..eval.calibration import auroc

    def f(recs):
        c = [r.get(conf_field) for r in recs]
        y = [r.get(corr_field) for r in recs]
        pairs = [(float(a), int(b)) for a, b in zip(c, y)
                 if a is not None and b is not None]
        if len(pairs) < 10:
            return float("nan")
        return auroc([p[0] for p in pairs], [p[1] for p in pairs])
    return f


# ------------------------------------------------------------- primary analysis

def h1_ordering(records_by_gen: list[list[dict]], seed: int = 0) -> dict:
    """H1: does the calibration family degrade at an earlier generation than accuracy?

    Uses the pre-declared primary signal and correctness target, paired
    item-level bootstraps, Holm correction across the calibration family, and an
    explicit chance-relative accuracy floor.
    """
    if not records_by_gen or not records_by_gen[0]:
        return {"error": "no records"}

    g0 = records_by_gen[0]
    tt = task_type(g0)
    spec = PRIMARY[tt]
    chance = chance_level(g0)

    acc_fn = stat_accuracy(spec["correctness"])
    acc_g0 = acc_fn(g0)
    floor = chance + FLOOR_MARGIN
    if not np.isfinite(acc_g0) or acc_g0 < floor:
        return {"task_type": tt, "chance_level": chance, "accuracy_g0": acc_g0,
                "floor": floor, "testable": False,
                "reason": (f"generation-0 accuracy {acc_g0:.3f} does not clear "
                           f"chance ({chance:.3f}) by the required {FLOOR_MARGIN:.2f} "
                           f"margin. H1 is an ordering claim about accuracy FALLING, "
                           f"so it is untestable here regardless of what calibration does.")}

    cal_stats = {
        "ece": (stat_ece(spec["confidence"], spec["correctness"]), +1),
        "overconfidence": (stat_overconfidence(spec["confidence"], spec["correctness"]), +1),
        "auroc": (stat_auroc(spec["confidence"], spec["correctness"]), -1),
    }

    results, pvals = {}, {}
    for name, (fn, direction) in cal_stats.items():
        r = onset_paired(records_by_gen, fn, direction, MDE[name], seed=seed)
        results[name] = r
        pvals[name] = r.get("p_min", float("nan"))

    acc = onset_paired(records_by_gen, acc_fn, -1, MDE["accuracy"], seed=seed)
    pvals["accuracy"] = acc.get("p_min", float("nan"))

    corrected = holm_bonferroni({k: v for k, v in pvals.items()
                                 if k in CALIBRATION_FAMILY})

    # Only Holm-surviving calibration metrics may claim an onset.
    cal_onsets = {k: results[k]["onset"] for k in CALIBRATION_FAMILY
                  if results[k]["onset"] is not None
                  and corrected.get(k, {}).get("significant")}
    cal_onset = min(cal_onsets.values()) if cal_onsets else None
    acc_onset = acc["onset"]

    if cal_onset is None and acc_onset is None:
        verdict, supported = "neither family degraded within the observed generations", None
    elif acc_onset is None:
        verdict, supported = (f"calibration onset g{cal_onset}; accuracy right-censored "
                              f"(never degraded)"), True
    elif cal_onset is None:
        verdict, supported = (f"accuracy onset g{acc_onset}; calibration never "
                              f"degraded after correction"), False
    elif cal_onset < acc_onset:
        verdict, supported = f"calibration g{cal_onset} < accuracy g{acc_onset}", True
    elif cal_onset == acc_onset:
        verdict, supported = f"tie at g{cal_onset}", None
    else:
        verdict, supported = f"calibration g{cal_onset} > accuracy g{acc_onset}", False

    return {
        "task_type": tt, "chance_level": chance, "accuracy_g0": acc_g0,
        "floor": floor, "testable": True,
        "primary_signal": spec["confidence"], "primary_correctness": spec["correctness"],
        "calibration": results, "accuracy": acc,
        "holm": corrected,
        "calibration_onset": cal_onset, "accuracy_onset": acc_onset,
        "h1_supported": supported, "verdict": verdict,
        "n_generations": len(records_by_gen),
    }
