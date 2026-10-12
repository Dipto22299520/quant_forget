"""Command line entry point.

    python -m collapse.cli run      --config configs/smoke.yaml
    python -m collapse.cli analyze  --run runs/smoke
    python -m collapse.cli check-data --lang bn
    python -m collapse.cli check-env
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import load_config, parse_overrides
from .utils import LOG, read_json, write_json


def cmd_run(args) -> int:
    from .loop import SelfTrainingLoop

    from .manifest import check_resume, save_manifest

    cfg = load_config(args.config, parse_overrides(args.set or []))
    if args.run_name:
        cfg.run_name = args.run_name

    # Compatibility is checked BEFORE the config is overwritten. The previous
    # order destroyed the very evidence needed to detect an incompatible resume.
    run_dir = Path(cfg.out_dir) / cfg.run_name
    manifest = check_resume(run_dir, cfg, force=getattr(args, "force", False))
    cfg.save(run_dir / "config.yaml")
    save_manifest(run_dir, manifest)
    LOG.info("run %s | model=%s lang=%s ratio=%s filter=%s gens=%d",
             cfg.run_name, cfg.model.name, cfg.data.lang, cfg.loop.ratio,
             cfg.loop.filter, cfg.loop.generations)

    loop = SelfTrainingLoop(cfg, offline=args.offline)
    loop.run()
    return 0


def cmd_analyze(args) -> int:
    from .analysis.figures import plot_all, plot_filter_decay
    from .analysis.onset import onset_table, ordering_test

    run_dir = Path(args.run)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        LOG.error("no summary.json in %s -- run the loop first", run_dir)
        return 1
    blob = read_json(summary_path)
    summary = blob["summary"]
    metrics = blob.get("metrics", [])

    results = {"run": str(run_dir), "eval_sets": {}}
    for name, curves in summary.get("curves", {}).items():
        onsets = onset_table(curves)
        results["eval_sets"][name] = {
            "onsets": onsets,
            "h1": ordering_test(onsets),
            "curves": curves,
        }
        LOG.info("[%s] %s", name, results["eval_sets"][name]["h1"]["verdict"])

    # H4: filter quality across generations, read from the training records.
    fq = []
    for g in range(len(metrics)):
        p = run_dir / "ckpt" / f"model_{g + 1}" / "_done.json"
        if p.exists():
            fq.append(read_json(p).get("filter_quality", {}))
    if fq:
        results["filter_decay"] = fq
        plot_filter_decay(fq, run_dir / "figures" / "fig3_filter_decay.png")

    figs = plot_all(summary, run_dir / "figures")
    results["figures"] = [str(f) for f in figs]
    write_json(run_dir / "analysis.json", results)
    LOG.info("analysis -> %s", run_dir / "analysis.json")
    return 0


def cmd_rescore(args) -> int:
    """Recompute metrics from stored records after a metric definition changed."""
    from .rescore import rescore_run
    rescore_run(args.run)
    return 0


def cmd_pilot(args) -> int:
    """Aggregate pilot seeds -> GO / NO-GO verdict."""
    from .analysis.pilot import run_pilot_analysis
    # Written OUTSIDE the run-name glob, or the verdict file gets picked up as
    # a run directory on the next invocation.
    out = Path(args.runs_dir) / ("_verdict_" + args.prefix.rstrip("_") + ".json")
    run_pilot_analysis(args.runs_dir, args.prefix, out=out,
                       eval_name=args.eval_set)
    return 0


def cmd_h1(args) -> int:
    """Pre-declared H1 analysis over one or more seed runs (no GPU)."""
    import numpy as np

    from .analysis.inference import h1_ordering
    from .rescore import _repair_record, _run_max_new_tokens
    from .utils import read_jsonl

    runs = sorted(Path(args.runs_dir).glob(args.prefix + "*"))
    runs = [r for r in runs if r.is_dir()]
    if not runs:
        LOG.error("no runs matching %s* in %s", args.prefix, args.runs_dir)
        return 1

    per_seed, results = [], {}
    for run in runs:
        gens = sorted(run.glob("gen*/"), key=lambda p: int(p.name[3:]))
        by_gen = []
        for g in gens:
            files = sorted(g.glob("eval_*.jsonl"))
            if args.eval_set:
                files = [f for f in files if f.stem == "eval_" + args.eval_set]
            if not files:
                break
            # Historical JSONL is never mutated (it is the raw artefact), so
            # fields added after a run -- p_pred, correct_scored, refused -- are
            # backfilled on load from what was saved.
            by_gen.append([_repair_record(r, _run_max_new_tokens(run))
                           for r in read_jsonl(files[0])])
        if len(by_gen) < 2:
            LOG.warning("skipping %s (needs >=2 generations)", run.name)
            continue
        r = h1_ordering(by_gen, seed=0)
        results[run.name] = r
        per_seed.append(r)
        if r.get("testable"):
            LOG.info("%-22s %s", run.name, r["verdict"])
        else:
            LOG.info("%-22s UNTESTABLE: acc_g0=%.3f vs floor %.3f",
                     run.name, r["accuracy_g0"], r["floor"])

    testable = [r for r in per_seed if r.get("testable")]
    sup = sum(1 for r in testable if r.get("h1_supported") is True)
    ref = sum(1 for r in testable if r.get("h1_supported") is False)
    summary = {"n_runs": len(per_seed), "n_testable": len(testable),
               "h1_supported_seeds": sup, "h1_refuted_seeds": ref,
               "per_run": results}

    print()
    print("=" * 74)
    print(f"PRE-DECLARED H1 ANALYSIS  ({len(per_seed)} runs, {len(testable)} testable)")
    print("=" * 74)
    for name, r in results.items():
        if not r.get("testable"):
            print(f"  {name:<24} UNTESTABLE  ({r['reason'][:60]}...)")
            continue
        print(f"  {name:<24} {r['verdict']}")
        for m, res in r["calibration"].items():
            sig = r["holm"].get(m, {}).get("significant")
            print(f"      {m:<16} onset={str(res['onset']):<5} "
                  f"p={r['holm'].get(m,{}).get('p', float('nan')):.4f} "
                  f"holm_sig={sig}")
        print(f"      {'accuracy':<16} onset={str(r['accuracy']['onset']):<5} "
              f"censored={r['accuracy']['censored']}")
    print("-" * 74)
    print(f"H1 supported in {sup} seed(s), refuted in {ref}, "
          f"{len(per_seed) - len(testable)} untestable")
    print("=" * 74)

    write_json(Path(args.runs_dir) / ("_h1_" + args.prefix.rstrip("_") + ".json"), summary)
    return 0


def cmd_check_data(args) -> int:
    """Try to load every registered dataset; report what is actually reachable."""
    from .data.registry import REGISTRY, by_lang
    from .data.loaders import load_dataset_key

    specs = by_lang(args.lang) if args.lang else list(REGISTRY.values())
    ok, bad = [], []
    for spec in specs:
        try:
            rows = load_dataset_key(spec.key, limit=5)
            ok.append((spec.key, len(rows), rows[0]["question"][:60] if rows else ""))
        except Exception as e:
            bad.append((spec.key, spec.hf_id, str(e).split("\n")[0][:160]))

    print("\n=== REACHABLE ===")
    for k, n, q in ok:
        print(f"  OK   {k:<14} n={n}  e.g. {q!r}")
    print("\n=== NOT REACHABLE ===")
    for k, hf, err in bad:
        print(f"  FAIL {k:<14} ({hf})\n       {err}")
    print(f"\n{len(ok)} reachable / {len(ok) + len(bad)} total")
    return 0 if not bad else 2


def cmd_check_env(args) -> int:
    import torch

    print(f"python      {sys.version.split()[0]}")
    print(f"torch       {torch.__version__}")
    print(f"cuda avail  {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu         {p.name}  {p.total_memory / 1e9:.1f} GB  sm_{p.major}{p.minor}")
    for m in ("transformers", "peft", "datasets", "sklearn", "bitsandbytes"):
        try:
            mod = __import__(m)
            print(f"{m:<12}{getattr(mod, '__version__', '?')}")
        except Exception as e:
            print(f"{m:<12}MISSING ({type(e).__name__})")

    from .modeling.loader import vram_precheck
    print("\n--- fits in this GPU? ---")
    for name, b in [("Llama-3.2-1B", 1.24), ("Qwen3-1.7B", 1.72), ("Qwen2.5-3B", 3.09)]:
        for mode in ("lora", "full"):
            r = vram_precheck(b, mode, "paged_adamw_8bit")
            print(f"  {name:<14} {mode:<5} need ~{r['need_gb']:>5.1f} GB  "
                  f"{'OK' if r['ok'] else 'TOO BIG'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser("collapse")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="run the self-training loop")
    p.add_argument("--config", required=True)
    p.add_argument("--run-name", default=None)
    p.add_argument("--offline", action="store_true",
                   help="use the built-in synthetic corpus (no downloads)")
    p.add_argument("--set", nargs="*", help="overrides, e.g. loop.ratio=0.5")
    p.add_argument("--force", action="store_true",
                   help="overwrite a run whose manifest describes a different experiment")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="onset + figures from a finished run")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("rescore",
                       help="recompute metrics from saved records (no GPU)")
    p.add_argument("--run", required=True)
    p.set_defaults(func=cmd_rescore)

    p = sub.add_parser("pilot", help="aggregate pilot seeds into a GO/NO-GO verdict")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--prefix", required=True,
                   help="run-name prefix, e.g. pilot_en_s")
    p.add_argument("--eval-set", default=None)
    p.set_defaults(func=cmd_pilot)

    p = sub.add_parser("h1", help="pre-declared H1 ordering analysis (no GPU)")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--prefix", required=True)
    p.add_argument("--eval-set", default=None)
    p.set_defaults(func=cmd_h1)

    p = sub.add_parser("check-data", help="probe every registered dataset")
    p.add_argument("--lang", default=None, choices=["en", "bn"])
    p.set_defaults(func=cmd_check_data)

    p = sub.add_parser("check-env", help="report versions and VRAM feasibility")
    p.set_defaults(func=cmd_check_env)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
