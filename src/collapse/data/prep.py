"""Build the three pools each run needs, with contamination enforcement.

    prompt_pool : questions re-answered by the model at every generation
    real_pool   : real (human) SFT examples, mixed in at ratio (1 - synthetic_frac)
    eval_sets   : never trained on, ever

The contamination guard is not decoration. Plan 4a/4b make held-out purity a
correctness property of the paper: if BanglaMedQA or BLUCK leaks into the SFT
mix, every downstream accuracy curve is invalid. So leakage raises, it does not warn.
"""
from __future__ import annotations

import random
import re
import unicodedata
from pathlib import Path

from ..utils import LOG, write_jsonl
from .loaders import load_dataset_key, synthetic_examples
from .registry import resolve


def norm_question(q: str) -> str:
    """Aggressive normalisation for overlap detection (not for grading)."""
    q = unicodedata.normalize("NFKC", q).lower().strip()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[^\w\sঀ-৿]", "", q)   # keep Bangla block
    return q


class ContaminationError(RuntimeError):
    pass


def resolve_overlap(train: list[dict], evals: list[dict], name: str = "",
                    holdout_only: bool = False,
                    max_incidental_frac: float = 0.01) -> list[dict]:
    """Remove training items that appear in a held-out set. Returns the clean pool.

    Independent public QA corpora genuinely share a few questions -- TriviaQA and
    MMLU overlap by a handful of items out of tens of thousands. That is
    incidental, and hard-failing on it just stops the run for no scientific gain.
    What is NOT incidental is a large overlap, or ANY overlap with a set marked
    holdout_only: both mean a seed source and an eval source share provenance,
    which silently invalidates every accuracy curve.

    So: drop-and-log for incidental overlap, hard error past the threshold.
    Either way the overlapping items never reach training.
    """
    ev = {norm_question(e["question"]) for e in evals}
    hits = [t for t in train if norm_question(t["question"]) in ev]
    if not hits:
        return train

    frac = len(hits) / max(1, len(train))
    if holdout_only or frac > max_incidental_frac:
        sample = [h['question'][:80] for h in hits[:5]]
        why = ' which is marked holdout_only' if holdout_only else ''
        raise ContaminationError(
            f'{len(hits)}/{len(train)} training items ({frac:.2%}) overlap the '
            f'held-out set {name!r}{why}.' + chr(10) +
            f'  examples: {sample}' + chr(10) +
            '  This is too much to be coincidence -- a seed source and an eval '
            'source share provenance. Fix the pool definition.'
        )

    keep_ids = {id(h) for h in hits}
    clean = [t for t in train if id(t) not in keep_ids]
    LOG.warning("dropped %d/%d training items (%.3f%%) overlapping held-out %r "
                "-- incidental overlap, under the %.1f%% threshold",
                len(hits), len(train), 100 * frac, name, 100 * max_incidental_frac)
    return clean


def build_pools(cfg, offline: bool = False) -> dict[str, list[dict]]:
    """Returns {'prompt_pool', 'real_pool', 'eval:<key>' ...}."""
    rng = random.Random(cfg.seed)
    dcfg = cfg.data
    cache = dcfg.cache_dir

    # ---------------- eval sets (load first so we can guard training) --------
    eval_sets: dict[str, list[dict]] = {}
    for key in dcfg.eval_sets:
        if offline:
            eval_sets[key] = synthetic_examples(dcfg.lang, min(dcfg.eval_size, 200),
                                                source=f"offline_{key}", split="eval")
            continue
        eval_sets[key] = load_dataset_key(key, limit=dcfg.eval_size, cache_dir=cache)

    # ---------------- training-side pools ------------------------------------
    if offline:
        pool = synthetic_examples(dcfg.lang, dcfg.prompt_pool_size + dcfg.real_pool_size,
                                  source="offline_seed", split="train")
    else:
        pool = []
        for key in dcfg.seed_pool:
            spec = resolve(key)
            if spec.holdout_only:
                raise ContaminationError(
                    f"{key!r} is marked holdout_only and must never appear in "
                    f"data.seed_pool. Remove it from the config."
                )
            per = max(1, (dcfg.prompt_pool_size + dcfg.real_pool_size) // max(1, len(dcfg.seed_pool)))
            pool.extend(load_dataset_key(key, limit=per, cache_dir=cache))

    rng.shuffle(pool)

    # Guard BEFORE splitting, against every eval set. Incidental overlap is
    # dropped; provenance-level overlap raises.
    for key, ev in eval_sets.items():
        pool = resolve_overlap(pool, ev, name=key,
                               holdout_only=resolve(key).holdout_only
                               if not offline else False)

    # Deduplicate within the pool itself -- repeated questions bias the loop.
    seen, deduped = set(), []
    for ex in pool:
        h = norm_question(ex["question"])
        if h in seen:
            continue
        seen.add(h)
        deduped.append(ex)
    if len(deduped) < len(pool):
        LOG.info("pool dedup: %d -> %d", len(pool), len(deduped))
    pool = deduped

    n_prompt = min(dcfg.prompt_pool_size, len(pool))
    prompt_pool = pool[:n_prompt]
    real_pool = pool[n_prompt:n_prompt + dcfg.real_pool_size]

    if not prompt_pool:
        raise RuntimeError("empty prompt pool -- check data.seed_pool and sizes")
    if not real_pool:
        LOG.warning("real_pool is EMPTY: only ratio=1.0 (100%% synthetic) runs are valid")

    LOG.info("pools: prompt=%d real=%d eval=%s",
             len(prompt_pool), len(real_pool),
             {k: len(v) for k, v in eval_sets.items()})

    out = {"prompt_pool": prompt_pool, "real_pool": real_pool}
    for k, v in eval_sets.items():
        out[f"eval:{k}"] = v
    return out


def save_pools(pools: dict[str, list[dict]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    for name, rows in pools.items():
        fn = name.replace(":", "_") + ".jsonl"
        write_jsonl(out_dir / fn, rows)
    LOG.info("saved pools -> %s", out_dir)
