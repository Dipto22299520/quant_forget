"""Is there a usable VERBALISED confidence signal at generation 0?

Fast, no training, no self-training loop. This decides whether H3 (internal vs
verbalised divergence) has an instrument at all before 2.4 h of GPU is spent.

It compares, on the same items:
    internal   p_correct from option-letter scoring (known good: AUROC .69/.60)
    verbalised the number the model states when asked "Confidence: N%"

Read the result like this:
    coverage < ~0.5   the model mostly ignores the instruction -> no instrument
    AUROC   ~ 0.50    the stated number is uninformative       -> H3 is dead
    AUROC   > 0.60    a real verbalised signal exists          -> H3 is alive

    python scripts/diagnose_verbalized.py
    python scripts/diagnose_verbalized.py --model Qwen/Qwen3-4B --n 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# The Windows console defaults to cp1252, which cannot encode Bangla. Without
# this, printing a single example response kills the whole diagnostic before it
# reaches the verdict.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from collapse.data.loaders import load_dataset_key
from collapse.eval.calibration import auroc, ece, overconfidence
from collapse.eval.grade import grade_all
from collapse.modeling.generate import generate_batch, score_choices
from collapse.modeling.loader import load_model, load_tokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sets", nargs="*", default=["global_mmlu_en", "global_mmlu_bn"])
    ap.add_argument("--max-new-tokens", type=int, default=40)
    args = ap.parse_args()

    tok = load_tokenizer(args.model)
    model = load_model(args.model, device="cuda")

    print()
    print("=" * 78)
    print(f"VERBALISED CONFIDENCE DIAGNOSTIC  |  {args.model}  |  n={args.n}")
    print("=" * 78)

    rows = []
    for key in args.sets:
        items = load_dataset_key(key, limit=args.n)
        lang = items[0].get("lang", "en")

        # --- internal reference: exact option-letter probability -------------
        scored = score_choices(model, tok, items, batch_size=8, max_len=1024)
        by_uid = {s["uid"]: s for s in scored}
        int_conf = [s["p_pred"] for s in scored]
        int_corr = [int(s["pred_idx"] == s["answer_idx"]) for s in scored]

        # --- verbalised: ask the model to state a number ---------------------
        gens = generate_batch(model, tok, items, style="verbalized",
                              max_new_tokens=args.max_new_tokens, batch_size=16,
                              temperature=0.7, top_p=0.9, max_len=1024, seed=0)
        graded = grade_all(gens, lang=lang)

        pairs = [(g["verbalized_conf"], by_uid[g["uid"]])
                 for g in graded
                 if g.get("verbalized_conf") is not None and g["uid"] in by_uid]
        coverage = len(pairs) / max(1, len(graded))

        print()
        print(f"--- {key}  (lang={lang}) ---")
        print(f"  internal  p_correct : acc={sum(int_corr)/len(int_corr):.3f}  "
              f"ECE={ece(int_conf, int_corr):.3f}  AUROC={auroc(int_conf, int_corr):.3f}")

        if len(pairs) < 20:
            print(f"  verbalised          : coverage={coverage:.2f} "
                  f"({len(pairs)}/{len(graded)}) -- TOO FEW to measure. "
                  f"The model is ignoring the instruction; H3 has no instrument here.")
            rows.append((key, coverage, None, None))
            for g in graded[:3]:
                print(f"      e.g. {g['response'][:90]!r}")
            continue

        v_conf = [p for p, _ in pairs]
        v_corr = [int(s["pred_idx"] == s["answer_idx"]) for _, s in pairs]
        a = auroc(v_conf, v_corr)
        print(f"  verbalised stated % : coverage={coverage:.2f} "
              f"({len(pairs)}/{len(graded)})  mean={sum(v_conf)/len(v_conf):.3f}  "
              f"ECE={ece(v_conf, v_corr):.3f}  AUROC={a:.3f}  "
              f"overconf={overconfidence(v_conf, v_corr):+.3f}")
        uniq = len(set(round(c, 2) for c in v_conf))
        print(f"  distinct stated values: {uniq}  "
              f"{'(near-constant -> carries almost no information)' if uniq <= 3 else ''}")
        for g in graded[:3]:
            print(f"      e.g. {g['response'][:90]!r}")
        rows.append((key, coverage, a, uniq))

    # ---- verdict ------------------------------------------------------------
    print()
    print("=" * 78)
    alive = [r for r in rows if r[2] is not None and r[2] > 0.60 and r[1] > 0.5]
    dead = [r for r in rows if r[2] is None or r[2] <= 0.55]
    if alive:
        print("H3 ALIVE: a verbalised signal exists on " +
              ", ".join(r[0] for r in alive) +
              ". Use prompt_style=verbalized as the primary condition.")
    elif dead:
        print("H3 DEAD at this scale: the stated confidence is uninformative or "
              "mostly absent.\n"
              "  -> Drop the verbalised axis, or retry on a larger model.\n"
              "  -> The internal p_correct signal still works, so H1/H2/H4 remain "
              "testable on MCQ.")
    else:
        print("H3 MARGINAL: signal present but weak. Try a larger model before "
              "committing the paper to it.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
