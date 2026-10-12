"""Correctness grading.

Bangla needs its own normalisation. The standard SQuAD normalisation strips
English articles and ASCII punctuation, which is a no-op on Bangla text and would
leave Bangla scored more strictly than English -- exactly the confound that would
fake an H2 result. So we normalise digits, strip Bangla punctuation (dari), and
drop common Bangla postpositional case endings before matching.
"""
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

from ..markers import extract_verbalized_confidence, marker_implied_confidence

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT = set(string.punctuation) | {"।", "॥", "‚", "”", "“", "’", "‘"}
_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

# Frequent Bangla case/postpositional suffixes. Trimmed conservatively: only
# applied to tokens longer than 4 chars so we never eat a short content word.
_BN_SUFFIXES = ("েরই", "টির", "টিতে", "গুলোর", "গুলো", "খানা", "টার", "েতে",
                "দের", "কের", "রের", "ের", "য়ে", "তে", "কে", "রা", "টি", "টা", "র")

_ANSWER_PREFIX = re.compile(
    r"^\s*(answer|ans|উত্তর)\s*[:：\-]\s*", re.IGNORECASE)


def _is_bangla(s: str) -> bool:
    return bool(re.search(r"[ঀ-৿]", s or ""))


def _strip_bn_suffix(tok: str) -> str:
    if len(tok) <= 4:
        return tok
    for suf in _BN_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def normalize_answer(s: str) -> str:
    """Language-aware normalisation for EM/F1."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s)).strip()
    s = _ANSWER_PREFIX.sub("", s)
    s = s.translate(_BN_DIGITS)
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = s.lower()
    if _is_bangla(s):
        toks = [_strip_bn_suffix(t) for t in s.split()]
    else:
        s = _ARTICLES.sub(" ", s)
        toks = s.split()
    return " ".join(t for t in toks if t)


def exact_match(pred: str, golds: list[str]) -> int:
    p = normalize_answer(pred)
    return int(any(p == normalize_answer(g) for g in golds))


def contains_match(pred: str, golds: list[str]) -> int:
    """Lenient: gold appears inside a hedged sentence.

    This is the primary open-QA metric here, because the hedged prompt style
    deliberately produces sentences, not bare spans -- strict EM would score
    hedging itself as an error and confound H1.
    """
    p = normalize_answer(pred)
    if not p:
        return 0
    for g in golds:
        gn = normalize_answer(g)
        if gn and gn in p:
            return 1
    return 0


def token_f1(pred: str, golds: list[str]) -> float:
    p_toks = normalize_answer(pred).split()
    best = 0.0
    for g in golds:
        g_toks = normalize_answer(g).split()
        if not p_toks or not g_toks:
            continue
        common = Counter(p_toks) & Counter(g_toks)
        n_same = sum(common.values())
        if n_same == 0:
            continue
        prec = n_same / len(p_toks)
        rec = n_same / len(g_toks)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


_LETTERS = "ABCDEFGHIJ"

# An option letter must LOOK like a selection, not merely be an isolated capital.
# The old pattern r"\b([A-J])\b" matched the "A" in "A politician is someone who"
# and the "I" in "I think it is Paris", silently converting ordinary prose into
# option A / option I. On a 4-option MCQ that manufactures ~25% accuracy from
# text that never named an option.
# A bare letter must be followed by PUNCTUATION or end-of-string, never merely a
# space: "A politician is someone who..." and "I think it is Paris" both begin
# with a valid option letter followed by a space, and both are prose, not
# selections. Requiring punctuation costs us "B journalist", which then falls
# through to the option-text match below and resolves correctly anyway.
_MCQ_RE = re.compile(
    r"^\s*\(?([A-J])\)?\s*(?:[.):\-]|$)"             # "B", "B.", "(B)", "B)"
    r"|(?:answer|option|ans|choice)\s*(?:is)?\s*[:.\-]?\s*\(?([A-J])\)?\b"
    r"|(?:উত্তর|বিকল্প)\s*[:.\-]?\s*\(?([A-J])\)?",
    re.IGNORECASE)


def parse_mcq_choice(pred: str, choices: list[str]) -> int | None:
    """Recover the selected option index from free text."""
    if not pred:
        return None
    m = _MCQ_RE.search(pred.strip()[:60])
    if m:
        letter = next((g for g in m.groups() if g), "").upper()
        if letter:
            return _LETTERS.index(letter)
    pn = normalize_answer(pred)
    for i, c in enumerate(choices or []):
        cn = normalize_answer(c)
        if cn and cn in pn:
            return i
    return None


def grade_record(rec: dict, lang: str | None = None, aggregation: str = "min_weakener") -> dict:
    """Attach correctness + all confidence signals to one generation record."""
    lang = lang or rec.get("lang", "en")
    resp = rec.get("response", "")

    if rec.get("task_type") == "mcq" and rec.get("choices"):
        pred_idx = rec.get("pred_idx")
        if pred_idx is None:
            pred_idx = parse_mcq_choice(resp, rec["choices"])
        gold = rec.get("answer_idx")
        correct = int(pred_idx == gold) if (pred_idx is not None and gold is not None) else 0
        em, f1 = correct, float(correct)
        rec = {**rec, "pred_idx": pred_idx}
    else:
        golds = rec.get("answers", []) or []
        em = exact_match(resp, golds)
        correct = contains_match(resp, golds)
        f1 = token_f1(resp, golds)

    mk = marker_implied_confidence(resp, lang=lang, aggregation=aggregation)
    verb = extract_verbalized_confidence(resp)

    out = dict(rec)
    out.update({
        # Correctness of the GENERATED TEXT. Every signal read off the generation
        # (markers, token probabilities, self-consistency) must be calibrated
        # against this, because that is what those signals describe.
        "correct_generated": int(correct),
        # Primary correctness. For MCQ the loop overwrites this with the
        # format-independent letter-scored value; see loop.evaluate_generation.
        "correct": int(correct),
        "exact_match": int(em),
        "token_f1": float(f1),
        "marker_conf": mk["marker_conf"],
        "hedged": mk["hedged"],
        "refused": mk["refused"],
        "n_markers": mk["n_markers"],
        "n_weakeners": mk["n_weakeners"],
        "n_strengtheners": mk["n_strengtheners"],
        "markers": mk["markers"],
        "verbalized_conf": verb,
    })
    return out


def grade_all(records: list[dict], lang: str | None = None,
              aggregation: str = "min_weakener") -> list[dict]:
    return [grade_record(r, lang=lang, aggregation=aggregation) for r in records]
