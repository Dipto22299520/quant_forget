"""Collapse-mitigation baselines -- and the instrumentation that breaks them (H4).

Self-verification filtering is *provably* collapse-preventing under a stated
"Confidence-Calibrated Agreement" assumption (OpenReview X5Hk8aMs6w). H4 tests
that assumption empirically: if calibration degrades across generations, then the
filter's own selection signal degrades, and the mitigation fails precisely in the
regime it was introduced for.

So every filter here returns not just the kept set but its measured quality
against ground truth. That quality trajectory IS the H4 result.
"""
from __future__ import annotations

import numpy as np

from .utils import LOG

FILTERS = ("none", "selfverify", "selfverify_marker", "sc_agreement", "cal", "selfverify_top",)


def _signal(rec: dict, name: str) -> float | None:
    v = rec.get(name)
    return None if v is None else float(v)


def apply_filter(records: list[dict], kind: str = "none", threshold: float = 0.5,
                 ) -> tuple[list[dict], dict]:
    """-> (kept_records, quality_report)

    kind:
      none               keep everything (the control arm)
      selfverify         threshold on internal confidence (mean token prob)
      selfverify_marker  threshold on the VERBALISED marker confidence.
                         The interesting one: a pipeline that filters on what the
                         model *says* about its certainty rather than its logits.
      sc_agreement       threshold on self-consistency agreement
      selfverify_top     keep the TOP `threshold` fraction by internal
                         confidence, re-ranked every generation. A fixed
                         threshold goes inert as the loop sharpens its own
                         distribution: at 0.5 the selfverify arm kept 99.7% of
                         candidates at g1 and 99.9% by g5 (median confidence
                         rose 0.81 -> 0.97), so it tested a halved budget, not
                         a filter. A quantile keeps selecting.
      cal                no data filtering -- CAL acts in the loss, so this is a
                         pass-through here and is handled by train.loss=cal
    """
    if kind not in FILTERS:
        raise ValueError(f"unknown filter {kind!r}, expected one of {FILTERS}")

    if kind in ("none", "cal"):
        kept = list(records)
        signal_name = None
    elif kind == "selfverify_top":
        signal_name = "mean_token_prob"
        scored = [(_signal(r, signal_name), r) for r in records]
        scored = [(v, r) for v, r in scored if v is not None]
        scored.sort(key=lambda vr: vr[0], reverse=True)
        n_keep = int(round(len(scored) * float(threshold)))
        kept = [r for _, r in scored[:n_keep]]
    else:
        signal_name = {"selfverify": "mean_token_prob",
                       "selfverify_marker": "marker_conf",
                       "sc_agreement": "sc_agreement"}[kind]
        kept = [r for r in records
                if (_signal(r, signal_name) or 0.0) >= threshold]

    report = filter_quality(records, kept, kind=kind, threshold=threshold,
                            signal=signal_name)
    LOG.info("filter %-18s kept %d/%d (%.1f%%) | precision %.3f recall %.3f",
             kind, len(kept), len(records),
             100 * report["keep_rate"], report["precision"], report["recall"])
    return kept, report


def filter_quality(all_records: list[dict], kept: list[dict], kind: str,
                   threshold: float, signal: str | None) -> dict:
    """Precision/recall of the filter against ground-truth correctness.

    precision = P(correct | kept)   -- how clean is the data we train on next?
    recall    = P(kept | correct)   -- how much good data do we throw away?

    H4's prediction: precision decays with generation, and decays faster in Bangla.
    """
    n_all = len(all_records)
    if n_all == 0:
        return {"kind": kind, "threshold": threshold, "signal": signal,
                "n_total": 0, "n_kept": 0, "keep_rate": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "base_rate": float("nan"), "lift": float("nan")}

    kept_uids = {r["uid"] for r in kept}
    # The filter selects GENERATED answers for the next SFT set, so its quality
    # must be judged against whether those generated answers were right.
    y = np.array([int(r.get("correct_generated", r.get("correct", 0)) or 0)
                  for r in all_records])
    k = np.array([r["uid"] in kept_uids for r in all_records])

    n_correct = int(y.sum())
    n_kept = int(k.sum())
    tp = int((y[k] == 1).sum()) if n_kept else 0

    precision = tp / n_kept if n_kept else float("nan")
    recall = tp / n_correct if n_correct else float("nan")
    base_rate = n_correct / n_all

    return {
        "kind": kind, "threshold": threshold, "signal": signal,
        "n_total": n_all, "n_kept": n_kept,
        "keep_rate": n_kept / n_all,
        "precision": precision,
        "recall": recall,
        "base_rate": base_rate,
        # lift > 0 means the filter actually beats "keep everything".
        # H4 predicts this trends to 0 across generations.
        "lift": (precision - base_rate) if n_kept else float("nan"),
        "f1": (2 * precision * recall / (precision + recall)
               if n_kept and n_correct and (precision + recall) > 0 else float("nan")),
    }


def sweep_threshold(records: list[dict], signal: str = "mean_token_prob",
                    thresholds: list[float] | None = None) -> list[dict]:
    """Filter quality across thresholds -- for choosing an operating point once,
    at g0, and then holding it fixed. Re-tuning per generation would hide the
    very degradation H4 is trying to measure."""
    thresholds = thresholds or [round(x, 2) for x in np.arange(0.0, 1.0, 0.05)]
    out = []
    for t in thresholds:
        kept = [r for r in records if (_signal(r, signal) or 0.0) >= t]
        q = filter_quality(records, kept, kind=f"threshold@{signal}",
                           threshold=t, signal=signal)
        out.append(q)
    return out


class RatioShortfall(RuntimeError):
    pass


def expand_prompts(pool: list[dict], k: int) -> list[dict]:
    """K samples per prompt at a fixed budget: the first len(pool)//k prompts, k times each.

    Copies get unique uids ("<uid>#s<j>") because everything downstream --
    pair building, the SFT mix, grading -- is keyed by uid. k == 1 returns the
    pool unchanged (same objects, same order), so the default path is untouched.
    """
    if k <= 1:
        return pool
    n = len(pool) // k
    return [dict(ex, uid=f"{ex['uid']}#s{j}") for ex in pool[:n] for j in range(k)]


def fresh_real_slice(real: list[dict], n_real: int, g: int, seed: int) -> tuple[list[dict], tuple[int, int]]:
    """The reference items for round g, disjoint from every other round's.

    One fixed permutation of the pool (by seed), cut into consecutive blocks of
    n_real. Fails closed when the pool cannot supply round g without reuse --
    silently wrapping around would reintroduce exactly the repetition this
    control exists to remove.
    """
    lo, hi = g * n_real, (g + 1) * n_real
    if hi > len(real):
        raise RatioShortfall(
            f"real_fresh: round {g} needs reference items [{lo}, {hi}) but the pool "
            f"has {len(real)}. Raise data.real_pool_size or lower the reference share.")
    order = np.random.default_rng(seed).permutation(len(real))
    return [real[i] for i in order[lo:hi]], (lo, hi)


def build_sft_mix(synthetic: list[dict], real: list[dict], ratio: float,
                  total: int, seed: int = 0, strict: bool = True,
                  tolerance: float = 0.02) -> tuple[list[dict], list[dict], dict]:
    """Mix synthetic and real examples at the configured ratio.

    ratio is the SYNTHETIC fraction: 1.0 = 100:0, 0.5 = 50:50, 0.25 = 25:75.

    FAIL CLOSED. The previous behaviour topped up a synthetic shortfall with real
    data and only logged a warning. That silently rewrote the experiment's primary
    independent variable: an r100 arm requesting 20,000 pairs from a 3,000-item
    prompt pool trained on 3,000 synthetic + 17,000 real = 15% synthetic, and the
    100/75/50 conditions all converged toward the same effective mixture.

    With strict=True a shortfall raises instead. Fix it by generating more
    synthetic answers or lowering loop.sft_size -- never by quietly changing the
    treatment. strict=False restores the old behaviour for exploratory runs and
    marks the result as not a controlled condition.
    """
    rng = np.random.default_rng(seed)
    available = len(synthetic) + len(real)
    if available == 0:
        raise ValueError("no examples to mix")
    total = min(total, available)
    n_syn = int(round(total * ratio))
    n_real = total - n_syn

    shortfall_syn = max(0, n_syn - len(synthetic))
    shortfall_real = max(0, n_real - len(real))

    if strict and (shortfall_syn or shortfall_real):
        raise RatioShortfall(
            f'cannot honour ratio={ratio} at sft_size={total}: need {n_syn} '
            f'synthetic (have {len(synthetic)}) and {n_real} real '
            f'(have {len(real)}).' + chr(10) +
            f'  Topping up would change the treatment to an effective ratio '
            f'of {len(synthetic) / max(1, total):.3f} and silently invalidate '
            f'the comparison across ratio arms.' + chr(10) +
            f'  Fix: raise data.prompt_pool_size, lower loop.sft_size, or set '
            f'loop.strict_ratio=false to accept an uncontrolled mixture.'
        )

    if shortfall_syn:
        LOG.warning("synthetic shortfall %d; topping up with real data -- "
                    "THIS IS NOT A CONTROLLED RATIO CONDITION", shortfall_syn)
        n_syn = len(synthetic)
        n_real = min(len(real), total - n_syn)
    if shortfall_real:
        LOG.warning("real shortfall %d", shortfall_real)
        n_real = len(real)

    syn_sel = [synthetic[i] for i in rng.permutation(len(synthetic))[:n_syn]]
    real_sel = [real[i] for i in rng.permutation(len(real))[:n_real]]

    realized = len(syn_sel) / max(1, len(syn_sel) + len(real_sel))
    info = {"requested_ratio": ratio, "n_synthetic": len(syn_sel),
            "n_real": len(real_sel), "total": len(syn_sel) + len(real_sel),
            "effective_ratio": realized,
            "shortfall_synthetic": shortfall_syn,
            "shortfall_real": shortfall_real,
            "strict": strict,
            # A run whose realized ratio drifted from the requested one is not a
            # valid member of the ratio grid; record that in the artefact itself.
            "controlled_condition": bool(abs(realized - ratio) <= tolerance)}
    LOG.info("sft mix: %d synthetic + %d real (requested %.2f, realized %.3f%s)",
             info["n_synthetic"], info["n_real"], ratio, realized,
             "" if info["controlled_condition"] else "  << UNCONTROLLED")
    return syn_sel, real_sel, info
