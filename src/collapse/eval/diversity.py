"""Lexical diversity / collapse signals.

The radiology precedent (2601.12946) is the reason these are first-class metrics:
vocabulary collapsed 8,186 -> 94 unique words while standard quality scores stayed
flat. Diversity moves when accuracy does not.
"""
from __future__ import annotations

import math
import random
import unicodedata
from collections import Counter

_PUNCT_TABLE = str.maketrans({c: " " for c in "।॥.,!?;:\"'()[]{}—–-"})


def _tokens(text: str) -> list[str]:
    t = unicodedata.normalize("NFKC", text or "").lower().translate(_PUNCT_TABLE)
    return [w for w in t.split() if w]


def distinct_n(texts: list[str], n: int = 1) -> float:
    """Unique n-grams / total n-grams."""
    total, uniq = 0, set()
    for t in texts:
        toks = _tokens(t)
        for i in range(len(toks) - n + 1):
            uniq.add(tuple(toks[i:i + n]))
            total += 1
    return len(uniq) / total if total else float("nan")


def type_token_ratio(texts: list[str]) -> float:
    toks = [w for t in texts for w in _tokens(t)]
    return len(set(toks)) / len(toks) if toks else float("nan")


def mattr(texts: list[str], window: int = 100) -> float:
    """Moving-average TTR: TTR is length-biased, and late generations produce
    shorter outputs, so raw TTR would confound length with diversity."""
    toks = [w for t in texts for w in _tokens(t)]
    if len(toks) < window:
        return type_token_ratio(texts)
    ratios = [len(set(toks[i:i + window])) / window
              for i in range(0, len(toks) - window + 1, max(1, window // 4))]
    return sum(ratios) / len(ratios) if ratios else float("nan")


def vocab_size(texts: list[str]) -> int:
    return len({w for t in texts for w in _tokens(t)})


def ngram_overconcentration(texts: list[str], n: int = 3, top_k: int = 10) -> float:
    """Share of all n-grams taken by the top-k most frequent. Rises under collapse."""
    counts = Counter()
    for t in texts:
        toks = _tokens(t)
        for i in range(len(toks) - n + 1):
            counts[tuple(toks[i:i + n])] += 1
    total = sum(counts.values())
    if not total:
        return float("nan")
    return sum(c for _, c in counts.most_common(top_k)) / total


def entropy_unigram(texts: list[str]) -> float:
    counts = Counter(w for t in texts for w in _tokens(t))
    total = sum(counts.values())
    if not total:
        return float("nan")
    return -sum((c / total) * math.log(c / total) for c in counts.values())


def self_bleu(texts: list[str], sample: int = 200, seed: int = 0) -> float:
    """Mean BLEU of each sample against the others. Higher = more homogeneous."""
    try:
        from sacrebleu.metrics import BLEU
    except ImportError:
        return float("nan")
    texts = [t for t in texts if t and t.strip()]
    if len(texts) < 5:
        return float("nan")
    rng = random.Random(seed)
    pool = texts if len(texts) <= sample else rng.sample(texts, sample)
    bleu = BLEU(effective_order=True)
    scores = []
    for i, hyp in enumerate(pool):
        refs = pool[:i] + pool[i + 1:]
        if not refs:
            continue
        # sacrebleu wants one list per reference stream; cap the streams for speed
        streams = [[r] for r in refs[:32]]
        try:
            scores.append(bleu.sentence_score(hyp, [s[0] for s in streams]).score)
        except Exception:
            continue
    return sum(scores) / len(scores) if scores else float("nan")


def diversity_report(records: list[dict], text_key: str = "response",
                     with_self_bleu: bool = True) -> dict:
    texts = [r.get(text_key, "") or "" for r in records]
    rep = {
        "n_texts": len(texts),
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "ttr": type_token_ratio(texts),
        "mattr_100": mattr(texts, 100),
        "vocab_size": vocab_size(texts),
        "unigram_entropy": entropy_unigram(texts),
        "trigram_top10_share": ngram_overconcentration(texts, 3, 10),
        "mean_len_tokens": (sum(len(_tokens(t)) for t in texts) / len(texts)
                            if texts else float("nan")),
    }
    if with_self_bleu:
        rep["self_bleu"] = self_bleu(texts)
    return rep
