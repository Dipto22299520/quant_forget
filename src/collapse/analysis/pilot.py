"""Aggregate pilot runs across seeds -- DESCRIPTIVE screening only.

    !! THIS IS NOT THE CONFIRMATORY TEST FOR H1 !!

    The hypothesis decision comes from `collapse.cli h1`
    (analysis/inference.py), which uses the pre-declared primary endpoint,
    paired item-level bootstraps, Holm correction and a chance-relative
    accuracy floor.

This module screens ALL tracked curves for movement, including secondary ones
like diversity, using a heuristic effect-to-spread ratio across seeds. It is
useful for spotting where to look, and it has twice produced a "GO" from
metrics that were at chance or on a near-zero base. Treat its output as a
descriptive scan, never as evidence.

Original docstring follows.

Aggregate pilot runs across seeds and emit a GO / NO-GO verdict.

A single small run cannot separate a real effect from seed noise, so the pilot is
only meaningful pooled across seeds. This answers three questions, in the order
that matters for deciding whether to rent bigger GPUs:

  Q1  Does ANYTHING move across generations?           (the plan's week-4 kill switch)
  Q2  If so, does calibration move BEFORE accuracy?    (H1)
  Q3  Is the effect larger than seed-to-seed noise?    (is it worth scaling?)

A null answer to Q1 is a legitimate and useful result: it says the REGIME is
wrong, not that the code is broken -- and it saves the GPU budget.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

from ..utils import LOG, read_json, write_json
from .onset import DEGRADE_DIRECTION, onset_table, ordering_test

# Metrics the verdict may consider. Fixing this list up front stops us fishing
# across dozens of curves for whichever one happened to move.
PRIMARY = ("accuracy", "accuracy_scored", "ece_marker", "ece_internal",
           "overconf_marker", "overconf_internal", "auroc_marker",
           "probe_auroc", "hedge_rate", "ivd_l1", "distinct_2", "self_bleu",
           "vocab_size")

# --- Guards against statistically-significant but MEANINGLESS movement -------
#
# A tight z-score only says the seeds agree; it says nothing about whether the
# quantity was ever measuring anything. Three ways that goes wrong here, all
# observed in the first real pilot:
#
#   chance floor : AUROC 0.510 -> 0.504 is chance-to-chance. The signal carried
#                  no information at g0, so its "decline" is not degradation.
#   accuracy floor: with g0 accuracy at 1%, H1 ("calibration degrades BEFORE
#                  accuracy") is untestable -- accuracy has no room to fall.
#   negligible    : hedge_rate 0.019 -> 0.007 is 6 items out of 300 becoming 2.
#
# Without these, the verdict returns GO on noise and the GPU budget is spent.
AUROC_METRICS = ("auroc_marker", "auroc_internal", "probe_auroc")
CHANCE_BAND = 0.05      # |AUROC - 0.5| below this = uninformative at baseline
MIN_ABS_DELTA = 0.02    # rates/probabilities: smaller absolute moves are noise
MIN_REL_DELTA = 0.10    # or a >=10% relative change, for unbounded metrics
ACCURACY_FLOOR = 0.10   # below this at g0, H1 cannot be tested


def load_pilot_runs(runs_dir, prefix: str) -> list[dict]:
    runs_dir = Path(runs_dir)
    out = []
    for d in sorted(runs_dir.glob(prefix + "*")):
        f = d / "summary.json"
        if f.exists():
            blob = read_json(f)
            blob["_run"] = d.name
            out.append(blob)
        else:
            LOG.warning("skipping %s (no summary.json -- run incomplete)", d.name)
    return out


def _curves(blob: dict, eval_name=None) -> dict:
    curves = blob.get("summary", {}).get("curves", {})
    if not curves:
        return {}
    if eval_name and eval_name in curves:
        return curves[eval_name]
    return next(iter(curves.values()))


def aggregate(runs: list[dict], eval_name=None) -> dict:
    """-> {metric: {per_seed, mean, sd, n_seeds}}"""
    per_metric: dict = {}
    for blob in runs:
        for m, series in _curves(blob, eval_name).items():
            per_metric.setdefault(m, []).append(series)

    agg = {}
    for m, seriess in per_metric.items():
        n_gen = min(len(s) for s in seriess)
        arr = np.array([[np.nan if v is None else float(v) for v in s[:n_gen]]
                        for s in seriess], dtype=float)
        # All-NaN columns are expected (a metric may be absent for this eval
        # set), so suppress the resulting warnings rather than spamming output.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean = np.nanmean(arr, axis=0)
            sd = (np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1
                  else np.zeros(n_gen))
        agg[m] = {"per_seed": arr.tolist(), "mean": mean.tolist(),
                  "sd": sd.tolist(), "n_seeds": int(arr.shape[0])}
    return agg


def movement(agg: dict, metric: str, min_z: float = 2.0) -> dict:
    """Did this metric move from g0 to the final generation, relative to seed noise?

    z = |mean change| / SD of the change across seeds. Using the SD of the CHANGE
    rather than of the level is the point: seeds may start at different levels and
    still move together, and that shared movement is the evidence we want.
    """
    if metric not in agg:
        return {"metric": metric, "moved": None, "reason": "absent"}
    arr = np.array(agg[metric]["per_seed"], dtype=float)
    if arr.shape[1] < 2:
        return {"metric": metric, "moved": None, "reason": "single generation"}

    deltas = arr[:, -1] - arr[:, 0]
    deltas = deltas[np.isfinite(deltas)]
    if len(deltas) == 0:
        return {"metric": metric, "moved": None, "reason": "all NaN"}

    mean_d = float(deltas.mean())
    sd_d = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    # Zero variance across seeds is not infinite confidence -- with 2 seeds it is
    # a coincidence waiting to happen, and would manufacture a false GO. Demand
    # at least 3 concordant seeds before treating a zero-spread delta as real.
    if sd_d > 1e-12:
        z = abs(mean_d) / sd_d
    elif abs(mean_d) <= 1e-9:
        z = 0.0
    elif len(deltas) >= 3:
        z = float("inf")
    else:
        z = 0.0        # too few seeds to trust a zero-spread delta

    direction = DEGRADE_DIRECTION.get(metric, 0)
    degraded = (mean_d > 0 and direction > 0) or (mean_d < 0 and direction < 0)
    g0 = float(np.nanmean(arr[:, 0]))
    rel = abs(mean_d) / abs(g0) if abs(g0) > 1e-9 else float("nan")

    # --- meaningfulness guards, applied AFTER the statistical test -----------
    disqualified = None
    if metric in AUROC_METRICS and abs(g0 - 0.5) < CHANCE_BAND:
        disqualified = ("baseline AUROC %.3f is within %.2f of chance -- the "
                        "signal carried no information at g0, so its change is "
                        "not degradation" % (g0, CHANCE_BAND))
    elif abs(mean_d) < MIN_ABS_DELTA and (not np.isfinite(rel) or rel < MIN_REL_DELTA):
        disqualified = ("absolute change %.4f is below the %.2f noise floor and "
                        "the relative change is under %.0f%%"
                        % (abs(mean_d), MIN_ABS_DELTA, 100 * MIN_REL_DELTA))

    return {
        "disqualified": disqualified,
        "metric": metric,
        "g0_mean": g0,
        "final_mean": float(np.nanmean(arr[:, -1])),
        "delta": mean_d,
        "delta_sd_across_seeds": sd_d,
        "z": float(z),
        "relative_change": float(rel),
        "moved": bool(z >= min_z and disqualified is None),
        "statistically_moved": bool(z >= min_z),
        "in_degrading_direction": bool(degraded),
        "n_seeds": int(len(deltas)),
    }


def pilot_verdict(runs: list[dict], eval_name=None, min_z: float = 2.0) -> dict:
    agg = aggregate(runs, eval_name)
    n_seeds = max((v["n_seeds"] for v in agg.values()), default=0)

    moves = {m: movement(agg, m, min_z) for m in PRIMARY if m in agg}
    degrading = {m: v for m, v in moves.items()
                 if v.get("moved") and v.get("in_degrading_direction")}
    improving = {m: v for m, v in moves.items()
                 if v.get("moved") and not v.get("in_degrading_direction")}

    # ---- Q2: onset ordering, computed PER SEED then counted -----------------
    per_seed_h1 = []
    for blob in runs:
        c = _curves(blob, eval_name)
        if not c:
            continue
        t = onset_table(c)
        entry = {"run": blob.get("_run")}
        entry.update(ordering_test(t))
        per_seed_h1.append(entry)
    supported = sum(1 for r in per_seed_h1 if r.get("h1_supported") is True)
    refuted = sum(1 for r in per_seed_h1 if r.get("h1_supported") is False)

    # Is H1 even testable? If accuracy starts at the floor it cannot fall, so
    # "calibration degrades before accuracy" is unfalsifiable in this setup.
    # accuracy_scored is present but ALL-NaN for open QA (no option letters to
    # rank), so prefer it only when it actually carries values -- otherwise the
    # floor check silently evaluates NaN and never fires.
    acc_g0 = float("nan")
    for acc_key in ("accuracy_scored", "accuracy"):
        if acc_key not in agg:
            continue
        col = np.array(agg[acc_key]["per_seed"], dtype=float)[:, 0]
        if np.isfinite(col).any():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                acc_g0 = float(np.nanmean(col))
            break
    acc_floored = bool(np.isfinite(acc_g0) and acc_g0 < ACCURACY_FLOOR)

    disq = {m: v["disqualified"] for m, v in moves.items()
            if v.get("disqualified") and v.get("statistically_moved")}

    if n_seeds < 2:
        decision = "INCONCLUSIVE"
        reason = ("only %d seed(s). Seed noise cannot be separated from effect. "
                  "Run at least 3 seeds before deciding." % n_seeds)
    elif acc_floored:
        decision = "NO-GO (accuracy at floor -- H1 untestable)"
        reason = ("Generation-0 accuracy is %.3f, below the %.2f floor. H1 claims "
                  "calibration degrades BEFORE accuracy, but accuracy has no room "
                  "to fall, so the ordering cannot be tested at all. Fix the task "
                  "difficulty first: use few-shot prompting, a longer generation "
                  "budget, an easier eval set, or a stronger base model. %s"
                  % (acc_g0, ACCURACY_FLOOR,
                     ("Note that %d metric(s) DID move meaningfully (%s) -- that "
                      "signal is real and worth pursuing separately."
                      % (len(degrading), ", ".join(sorted(degrading)))
                      if degrading else "")))
    elif not degrading:
        decision = "NO-GO (no degradation)"
        reason = ("Nothing degraded beyond seed noise under deliberately "
                  "collapse-friendly settings (100%% synthetic, temperature 1.0, "
                  "6 generations). Scaling up will not create an effect that is "
                  "absent here -- change the REGIME first: narrower domain, more "
                  "generations, full fine-tuning instead of LoRA, or a weaker base "
                  "model with more headroom to lose.")
    elif supported > refuted:
        decision = "GO"
        reason = ("%d metric(s) degraded beyond seed noise and H1 ordering held in "
                  "%d/%d seeds. Scaling up is justified."
                  % (len(degrading), supported, len(per_seed_h1)))
    elif supported == 0 and refuted > 0:
        decision = "GO (H1 refuted, effect real)"
        reason = ("Degradation is real (%d metrics moved) but accuracy moved FIRST "
                  "in %d/%d seeds. That refutes H1 as stated -- publishable, but "
                  "the paper's framing must change before spending the GPU budget."
                  % (len(degrading), refuted, len(per_seed_h1)))
    else:
        decision = "MARGINAL"
        reason = ("Degradation present but H1 ordering inconsistent across seeds "
                  "(%d support / %d refute). More seeds or more generations needed."
                  % (supported, refuted))

    return {
        "decision": decision,
        "reason": reason,
        "n_seeds": n_seeds,
        "degrading_metrics": degrading,
        "improving_metrics": improving,
        "unmoved_metrics": [m for m, v in moves.items() if v.get("moved") is False],
        "disqualified_metrics": disq,
        "accuracy_g0": acc_g0,
        "accuracy_floored": bool(acc_floored),
        "h1_per_seed": per_seed_h1,
        "h1_supported_seeds": supported,
        "h1_refuted_seeds": refuted,
        "aggregate_curves": agg,
    }


def print_verdict(v: dict) -> None:
    bar = "=" * 74
    print(bar)
    print("DESCRIPTIVE SCREEN: %s   (%d seeds)" % (v["decision"], v["n_seeds"]))
    print("NOT a hypothesis test -- run `collapse.cli h1 --prefix <run prefix>`")
    print("for the pre-declared confirmatory analysis.")
    print(bar)
    print(v["reason"])
    print()
    header = "%-22s%9s%9s%9s%7s  verdict" % ("metric", "g0", "final", "delta", "z")
    print(header)
    print("-" * 74)
    rows = list(v["degrading_metrics"].values()) + list(v["improving_metrics"].values())
    for m in sorted(rows, key=lambda r: -abs(r.get("z", 0))):
        tag = "DEGRADED" if m["in_degrading_direction"] else "improved"
        z = m["z"]
        zs = "inf" if not np.isfinite(z) else ("%.1f" % z)
        print("%-22s%9.3f%9.3f%+9.3f%7s  %s"
              % (m["metric"], m["g0_mean"], m["final_mean"], m["delta"], zs, tag))
    if v.get("disqualified_metrics"):
        print()
        print("DISQUALIFIED (passed the z-test but not meaningful):")
        for m, why in v["disqualified_metrics"].items():
            print("   %-20s %s" % (m, why))
    if v["unmoved_metrics"]:
        print()
        print("no movement beyond seed noise: " + ", ".join(v["unmoved_metrics"]))
    print()
    print("H1 ordering: supported in %d seed(s), refuted in %d"
          % (v["h1_supported_seeds"], v["h1_refuted_seeds"]))
    for r in v["h1_per_seed"]:
        print("   %-24s %s" % (r.get("run", "?"), r.get("verdict", "")))
    print(bar)


def run_pilot_analysis(runs_dir, prefix: str, out=None, eval_name=None) -> dict:
    runs = load_pilot_runs(runs_dir, prefix)
    if not runs:
        raise FileNotFoundError(
            "no completed runs matching %r in %s" % (prefix, runs_dir))
    v = pilot_verdict(runs, eval_name=eval_name)
    print_verdict(v)
    if out:
        write_json(out, v)
        LOG.info("pilot verdict -> %s", out)
    return v
