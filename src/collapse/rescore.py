"""Recompute metrics from stored per-item records, without touching the GPU.

Why this exists: the eval jsonl written each generation already contains every
per-item field the metrics are derived from (response, choice_probs, answer_idx,
token probabilities, markers). So when a metric DEFINITION changes -- as it did
when accuracy moved to format-independent letter scoring -- the corrected numbers
can be recovered from finished runs instead of spending hours regenerating.

Probe results are the one exception: they need hidden states, hence a GPU. Those
are carried over unchanged from the existing metrics file and flagged as stale.
"""
from __future__ import annotations

from pathlib import Path

from .eval.calibration import accuracy_by_bucket, calibration_report
from .eval.diversity import diversity_report
from .eval.ivd import internal_verbal_divergence
from .utils import LOG, read_json, read_jsonl, write_json


def _repair_record(r: dict, max_new_tokens: int | None = None) -> dict:
    """Backfill fields introduced after this run was produced."""
    if "correct_generated" not in r:
        r["correct_generated"] = int(r.get("correct", 0) or 0)

    # Older records only stored `truncated` (post-processing changed the text),
    # which was ~0 while 99% of responses actually ran out of budget. Recover
    # the budget flag from the kept-token count: n_gen_tokens counts up to and
    # including EOS, so a full-budget span almost always means no EOS came.
    if "hit_budget" not in r and max_new_tokens:
        r["hit_budget"] = bool((r.get("n_gen_tokens") or 0) >= max_new_tokens)

    # Format-independent MCQ correctness, recoverable from the stored
    # per-option probabilities.
    cp, ai = r.get("choice_probs"), r.get("answer_idx")
    if cp and ai is not None:
        pred = max(range(len(cp)), key=lambda i: cp[i])
        r.setdefault("pred_idx_scored", pred)
        r.setdefault("correct_scored", int(pred == ai))
        # p_pred is recoverable from the stored per-option probabilities, so the
        # circular-metric fix needs no GPU re-run.
        r.setdefault("p_pred", float(cp[pred]))
    if "correct_scored" in r:
        r["correct"] = r["correct_scored"]

    if "refused" not in r:
        from .markers.extract import is_refusal
        r["refused"] = is_refusal(r.get("response", ""))
    return r


def _run_max_new_tokens(run_dir: Path) -> int | None:
    """gen.max_new_tokens from the run's resolved config, if it is there."""
    cfg = run_dir / "config.yaml"
    if not cfg.exists():
        return None
    try:
        import yaml
        return int(yaml.safe_load(cfg.read_text(encoding="utf-8"))["gen"]["max_new_tokens"])
    except Exception:
        return None


def rescore_run(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    gens = sorted(run_dir.glob("gen*/_metrics.json"),
                  key=lambda p: int(p.parent.name[3:]))
    if not gens:
        raise FileNotFoundError(f"no gen*/_metrics.json under {run_dir}")

    max_new = _run_max_new_tokens(run_dir)
    all_metrics = []
    for mpath in gens:
        gdir = mpath.parent
        old = read_json(mpath)
        g = old.get("generation", int(gdir.name[3:]))
        new = dict(old)
        new["eval"] = {}
        new["rescored"] = True

        for f in sorted(gdir.glob("eval_*.jsonl")):
            name = f.stem[len("eval_"):]
            rows = [_repair_record(r, max_new) for r in read_jsonl(f)]

            entry = calibration_report(rows, n_boot=200, seed=old.get("seed", 0))
            entry["diversity"] = diversity_report(rows)
            entry["popularity_buckets"] = accuracy_by_bucket(
                rows, "s_pop", [0, 100, 1000, 10000, 100000, 10**9])
            entry["ivd_tokenprob"] = internal_verbal_divergence(rows, probe_res=None)

            # Probes need hidden states -> GPU. Carry the old values across and
            # say so, rather than silently dropping or faking them.
            old_entry = (old.get("eval") or {}).get(name, {})
            if "probe" in old_entry:
                entry["probe"] = old_entry["probe"]
                entry["probe"]["stale"] = True
            if "ivd" in old_entry:
                entry["ivd"] = old_entry["ivd"]
                entry["ivd"]["stale"] = True
            for k in ("filter_sweep", "filter_sweep_marker"):
                if k in old_entry:
                    entry[k] = old_entry[k]

            new["eval"][name] = entry
            LOG.info("g%d %-14s acc=%.3f (generated %.3f / scored %s) hedge=%.3f",
                     g, name, entry["accuracy"], entry["accuracy_generated"],
                     ("%.3f" % entry["accuracy_scored"]
                      if entry["accuracy_scored"] is not None else "n/a"),
                     entry["hedge_rate"])

        write_json(mpath, new)
        all_metrics.append(new)

    from .loop import summarize
    summary = summarize(all_metrics)
    write_json(run_dir / "summary.json",
               {"config_run": run_dir.name, "metrics": all_metrics,
                "summary": summary, "rescored": True})
    LOG.info("rescored %d generations -> %s", len(all_metrics), run_dir / "summary.json")
    return summary
