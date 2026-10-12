"""H2: matched-baseline cross-lingual comparison.

The plan calls this non-negotiable, and it is the single most likely reviewer
attack on H2. Bangla and English models differ in generation-0 accuracy. If you
compare raw degradation curves, a reviewer says -- correctly -- that you measured
the accuracy gap, not the collapse RATE.

Two corrections are provided, and the paper should report both:

  subsample  : drop items from the easier language until g0 accuracy matches.
               Simple, defensible, costs statistical power.
  normalised : express every generation as a fraction of its own g0 value.
               Keeps all items, but assumes proportional degradation is the right
               scale -- state that assumption explicitly.
"""
from __future__ import annotations

import numpy as np


def _acc(recs: list[dict]) -> float:
    return float(np.mean([int(r.get("correct", 0)) for r in recs])) if recs else float("nan")


def matched_baseline_subsample(recs_a: list[dict], recs_b: list[dict],
                               tol: float = 0.01, seed: int = 0,
                               max_iter: int = 20000) -> dict:
    """Subsample the higher-accuracy set until both match within tol.

    Drops CORRECT items from the stronger language at random. Dropping incorrect
    items instead would inflate the weaker language's apparent headroom.
    """
    rng = np.random.default_rng(seed)
    a, b = list(recs_a), list(recs_b)
    acc_a, acc_b = _acc(a), _acc(b)

    if not np.isfinite(acc_a) or not np.isfinite(acc_b):
        return {"error": "empty input", "n_a": len(a), "n_b": len(b)}

    if abs(acc_a - acc_b) <= tol:
        return {"matched": True, "n_a": len(a), "n_b": len(b),
                "acc_a": acc_a, "acc_b": acc_b, "dropped": 0,
                "note": "already matched, no subsampling needed",
                "recs_a": a, "recs_b": b}

    # Work on the stronger set.
    if acc_a > acc_b:
        strong, weak, target = a, b, acc_b
        strong_is_a = True
    else:
        strong, weak, target = b, a, acc_a
        strong_is_a = False

    correct_idx = [i for i, r in enumerate(strong) if int(r.get("correct", 0)) == 1]
    rng.shuffle(correct_idx)

    drop: set[int] = set()
    it = 0
    while it < max_iter and correct_idx:
        cur = [r for i, r in enumerate(strong) if i not in drop]
        if _acc(cur) <= target + tol:
            break
        drop.add(correct_idx.pop())
        it += 1

    strong_matched = [r for i, r in enumerate(strong) if i not in drop]
    out = {
        "matched": abs(_acc(strong_matched) - target) <= tol,
        "dropped": len(drop),
        "target_accuracy": target,
        "achieved_accuracy": _acc(strong_matched),
        "note": ("Subsampling drops correct items from the stronger language. "
                 "Report the dropped count -- it bounds the power loss."),
    }
    if strong_is_a:
        out.update({"recs_a": strong_matched, "recs_b": weak,
                    "n_a": len(strong_matched), "n_b": len(weak),
                    "acc_a": _acc(strong_matched), "acc_b": _acc(weak)})
    else:
        out.update({"recs_a": weak, "recs_b": strong_matched,
                    "n_a": len(weak), "n_b": len(strong_matched),
                    "acc_a": _acc(weak), "acc_b": _acc(strong_matched)})
    return out


def normalize_curve(curve: list[float | None]) -> list[float | None]:
    """Express a curve relative to its own generation-0 value."""
    vals = [None if v is None else float(v) for v in curve]
    if not vals or vals[0] in (None, 0) or (isinstance(vals[0], float) and np.isnan(vals[0])):
        return vals
    base = vals[0]
    return [None if v is None else v / base for v in vals]


def degradation_rate(curve: list[float | None]) -> float:
    """Least-squares slope per generation on the normalised curve.

    This is the quantity H2 actually compares between languages: not the level,
    the rate.
    """
    vals = [(g, v) for g, v in enumerate(normalize_curve(curve))
            if v is not None and np.isfinite(v)]
    if len(vals) < 2:
        return float("nan")
    xs = np.array([g for g, _ in vals], dtype=float)
    ys = np.array([v for _, v in vals], dtype=float)
    return float(np.polyfit(xs, ys, 1)[0])


def matched_comparison(curves_by_lang: dict[str, dict[str, list]],
                       metrics: tuple[str, ...] = ("accuracy", "ece_marker",
                                                   "ece_internal", "hedge_rate",
                                                   "probe_auroc", "ivd_l1")) -> dict:
    """Compare degradation RATES between languages on normalised curves."""
    langs = sorted(curves_by_lang)
    out: dict = {"languages": langs, "metrics": {}}
    for m in metrics:
        entry = {}
        for lang in langs:
            c = curves_by_lang[lang].get(m)
            if c is None:
                continue
            entry[lang] = {
                "raw": c,
                "normalised": normalize_curve(c),
                "slope_per_generation": degradation_rate(c),
                "g0": c[0] if c else None,
            }
        if len(entry) == 2:
            a, b = langs
            sa = entry.get(a, {}).get("slope_per_generation", float("nan"))
            sb = entry.get(b, {}).get("slope_per_generation", float("nan"))
            entry["slope_gap"] = (float(sb - sa)
                                  if np.isfinite(sa) and np.isfinite(sb) else float("nan"))
        out["metrics"][m] = entry

    out["caveat"] = (
        "Normalised slopes assume proportional degradation. Report the subsampled "
        "matched-baseline comparison alongside; if the two disagree, the raw "
        "accuracy gap is doing the work and H2 is not supported.")
    return out
