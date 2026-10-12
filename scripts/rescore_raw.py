r"""Recover UNDEBIASED option scores for g0 and g5 of every valid arm.

    .venv\Scripts\python.exe scripts\rescore_raw.py            # all valid arms
    .venv\Scripts\python.exe scripts\rescore_raw.py --arm main_en_r100_s1

The loop stored only the debiased per-option probabilities. The final weights
(ckpt/model_5) are never pruned and g0 is the released model, so both endpoints
can be re-scored offline. Writes paper/raw_scores/<arm>_g{0,5}.jsonl with the
raw and debiased distributions, and prints a reproduction check: the re-scored
debiased p_pred must match what the run stored.

No pipeline code is touched, so run fingerprints are unaffected.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from collapse.utils import read_jsonl, write_jsonl, free_vram        # noqa: E402
from collapse.modeling.loader import load_model, load_tokenizer     # noqa: E402
from collapse.modeling.generate import score_choices                # noqa: E402
from paper_assets import discover                                   # noqa: E402

OUT = ROOT / "paper" / "raw_scores"


def score(model_name: str, cfg: dict, items: list[dict]) -> list[dict]:
    tok = load_tokenizer(model_name, cfg["model"]["trust_remote_code"])
    model = load_model(model_name, dtype=cfg["model"]["dtype"], attn_impl=cfg["model"]["attn_impl"],
                       trust_remote_code=cfg["model"]["trust_remote_code"], device=cfg["device"])
    res = score_choices(model, tok, items, style=cfg["gen"]["prompt_style"],
                        batch_size=max(1, cfg["gen"]["batch_size"] // 2),
                        max_len=cfg["model"]["max_len"])
    del model, tok
    free_vram()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=None)
    ap.add_argument("--force", action="store_true", help="re-score endpoints that already exist")
    ap.add_argument("--lang", default=None, help="only arms in this language")
    ap.add_argument("--size", default=None, help="only arms of this model size, e.g. 1.7B")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ap_size = args.size
    arms = [a for a in discover() if a["valid"] and (args.arm is None or a["name"] == args.arm)
            and (args.lang is None or a["lang"] == args.lang)
            and (ap_size is None or a["size"] == ap_size)]
    base_cache: dict[tuple, list[dict]] = {}
    for a in arms:
        run = ROOT / "runs" / a["name"]
        cfg = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
        es = a["eval_set"]
        items = [{k: v for k, v in r.items() if k != "prompt"}
                 for r in read_jsonl(run / "pools" / f"eval_{es}.jsonl")]
        for g, model_name in ((0, cfg["model"]["name"]), (5, str(run / "ckpt" / "model_5"))):
            out = OUT / f"{a['name']}_g{g}.jsonl"
            if out.exists() and not args.force:
                print(f"{a['name']} g{g}: exists, skipping"); continue
            key = (model_name, es, cfg["model"]["max_len"])
            if g == 0 and key in base_cache:
                res = base_cache[key]
            else:
                print(f"{a['name']} g{g}: scoring {model_name}")
                res = score(model_name, cfg, items)
                if g == 0:
                    base_cache[key] = res
            # reproduction check against what the run stored
            stored = {r["uid"]: r for r in read_jsonl(run / f"gen{g}" / f"eval_{es}.jsonl")}
            diffs = [abs(r["p_pred"] - stored[r["uid"]]["p_pred"]) for r in res if r["uid"] in stored]
            agree = np.mean([r["pred_idx"] == stored[r["uid"]]["pred_idx_scored"] for r in res if r["uid"] in stored])
            print(f"   reproduction: max |Δp_pred| = {max(diffs):.2e}, choice agreement = {agree:.3f}")
            rows = [{"uid": r["uid"], "answer_idx": r["answer_idx"],
                     "choice_probs": r["choice_probs"], "choice_probs_raw": r["choice_probs_raw"],
                     "pred_idx": r["pred_idx"], "pred_idx_raw": r["pred_idx_raw"],
                     "p_pred": r["p_pred"], "p_pred_raw": float(max(r["choice_probs_raw"])),
                     "choice_logits": r["choice_logits"], "choice_logits_raw": r["choice_logits_raw"],
                     "correct_scored": int(r["correct"] or 0),
                     "correct_raw": int(r["pred_idx_raw"] == r["answer_idx"])} for r in res]
            write_jsonl(out, rows)
    print("done ->", OUT)


if __name__ == "__main__":
    main()
