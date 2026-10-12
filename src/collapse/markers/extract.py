"""Epistemic marker extraction and marker-implied confidence.

Three confidence signals come out of a generation, and the paper needs all three
kept separate:

  1. marker-implied  -- from the hedging lexicon (this file)
  2. verbalized      -- an explicit number the model states ("Confidence: 70%")
  3. internal        -- token logprobs / probe readout (see eval/probes.py)

H3 is literally a statement about (1) and (3) diverging over generations, so the
extractor must not quietly borrow information from the other two.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_LEX_DIR = Path(__file__).parent

# Bangla codepoint block, used to decide whether \b word boundaries are safe.
_BN_RE = re.compile(r"[ঀ-৿]")


@dataclass
class MarkerHit:
    marker: str
    kind: str          # "strengthener" | "weakener"
    implied: float
    start: int
    end: int

    def to_dict(self) -> dict:
        return {"marker": self.marker, "kind": self.kind, "implied": self.implied,
                "start": self.start, "end": self.end}


@lru_cache(maxsize=8)
def load_lexicon(lang: str) -> dict:
    path = _LEX_DIR / f"lexicon_{lang}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"no marker lexicon for lang={lang!r} at {path}. "
            f"Add one -- do not fall back to English, that would silently invalidate H2."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


class MarkerExtractor:
    """Longest-match-first, non-overlapping marker scanner."""

    def __init__(self, lang: str, aggregation: str = "min_weakener"):
        self.lang = lang
        self.aggregation = aggregation
        lex = load_lexicon(lang)
        self.meta = lex.get("_meta", {})
        self.bare_conf = float(self.meta.get("bare_assertion_conf", 0.85))
        self.table: dict[str, tuple[str, float]] = {}
        for kind in ("strengtheners", "weakeners"):
            for m, v in lex.get(kind, {}).items():
                self.table[_norm(m).lower()] = (kind[:-1], float(v))

        # Longest first: guarantees "i'm not sure" beats "sure",
        # and "নিশ্চিত নই" beats "নিশ্চিত".
        markers = sorted(self.table, key=len, reverse=True)
        self._pattern = self._compile(markers)

    def _compile(self, markers: list[str]) -> re.Pattern:
        parts = []
        for m in markers:
            esc = re.escape(m)
            if _BN_RE.search(m):
                # Bangla: no reliable \b at script boundaries in all engines; guard
                # with explicit non-letter lookarounds instead.
                parts.append(rf"(?<![ঀ-৿]){esc}(?![ঀ-৿])")
            else:
                parts.append(rf"\b{esc}\b")
        return re.compile("|".join(parts), re.IGNORECASE | re.UNICODE)

    def find(self, text: str) -> list[MarkerHit]:
        text = _norm(text or "")
        hits: list[MarkerHit] = []
        for m in self._pattern.finditer(text.lower()):
            surface = m.group(0)
            kind, implied = self.table[surface]
            hits.append(MarkerHit(surface, kind, implied, m.start(), m.end()))
        return hits

    def implied_confidence(self, text: str) -> tuple[float, bool, list[MarkerHit]]:
        """-> (implied_confidence, is_hedged, hits)

        Aggregation policy matters and is reported in the paper:
          min_weakener (default) -- the weakest commitment governs the utterance.
                                    Matches the finding (2605.28778) that the real
                                    signal is hedge / no-hedge, not graded semantics.
          mean                   -- average over all hits.
          first                  -- leftmost marker only.
        """
        hits = self.find(text)
        if not hits:
            return self.bare_conf, False, hits

        weak = [h for h in hits if h.kind == "weakener"]
        strong = [h for h in hits if h.kind == "strengthener"]
        is_hedged = bool(weak)

        if self.aggregation == "mean":
            val = sum(h.implied for h in hits) / len(hits)
        elif self.aggregation == "first":
            val = min(hits, key=lambda h: h.start).implied
        else:  # min_weakener
            val = min(h.implied for h in weak) if weak else max(h.implied for h in strong)
        return float(val), is_hedged, hits


@lru_cache(maxsize=8)
def _extractor(lang: str, aggregation: str) -> MarkerExtractor:
    return MarkerExtractor(lang, aggregation)


def extract_markers(text: str, lang: str, aggregation: str = "min_weakener") -> list[dict]:
    return [h.to_dict() for h in _extractor(lang, aggregation).find(text)]


def marker_implied_confidence(text: str, lang: str,
                              aggregation: str = "min_weakener",
                              refusal_conf: float = 0.10) -> dict:
    conf, hedged, hits = _extractor(lang, aggregation).implied_confidence(text)
    refused = is_refusal(text)
    # An abstention with no hedge marker must not inherit the bare-assertion
    # default of 0.85 -- that would record a refusal as near-certainty.
    if refused and not hits:
        conf = refusal_conf
    return {
        "marker_conf": conf,
        "hedged": hedged,
        "refused": refused,
        "n_markers": len(hits),
        "n_weakeners": sum(1 for h in hits if h.kind == "weakener"),
        "n_strengtheners": sum(1 for h in hits if h.kind == "strengthener"),
        "markers": [h.marker for h in hits],
    }


# -------------------------------------------------------------------------
# Verbalized numeric confidence ("Confidence: 80%", "আত্মবিশ্বাস: ০.৮")
# -------------------------------------------------------------------------

_BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

_CONF_PATTERNS = [
    re.compile(r"confidence\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*%", re.I),
    re.compile(r"confidence\s*[:=]?\s*([0-9]*\.?[0-9]+)", re.I),
    re.compile(r"([0-9]*\.?[0-9]+)\s*%\s*confident", re.I),
    re.compile(r"আত্মবিশ্বাস\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*%?"),
    re.compile(r"নিশ্চয়তা\s*[:=]?\s*([0-9]*\.?[0-9]+)\s*%?"),
]


def extract_verbalized_confidence(text: str) -> float | None:
    """Parse an explicit stated confidence, normalised to [0, 1]. None if absent."""
    if not text:
        return None
    t = _norm(text).translate(_BN_DIGITS)
    for pat in _CONF_PATTERNS:
        m = pat.search(t)
        if not m:
            continue
        try:
            v = float(m.group(1))
        except ValueError:
            continue
        # "Confidence: 0.9" and "Confidence: 90%" are unambiguous. A BARE
        # "Confidence: 9" is not -- 9%, 9/10 and a 1-10 scale are all plausible
        # and they differ by an order of magnitude. Returning a guess would
        # inject fabricated precision into every calibration number, so bare
        # ambiguous integers are rejected instead.
        if pat.pattern.endswith("%") or "%" in m.group(0):
            v = v / 100.0
        elif v > 1.0:
            if v <= 10.0:
                continue            # ambiguous scale -- refuse to guess
            v = v / 100.0
        if 0.0 <= v <= 1.0:
            return v
    return None


# -------------------------------------------------------------------------
# Refusals / abstentions.
#
# These are epistemically meaningful but CATEGORICALLY DIFFERENT from hedging,
# and conflating them would distort H1. "I don't have access to that information"
# is not a confident assertion, but it is also not a hedged answer -- it is an
# abstention, and a model that collapses into abstention is failing in a
# different way from one that collapses into confident wrongness.
#
# Left uncounted, refusals score as bare assertions at 0.85 implied confidence,
# which is badly wrong in the direction that would FAKE support for H1.
# -------------------------------------------------------------------------

_REFUSAL_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"\bi (?:can ?not|can't|cannot|am unable to|'m unable to) (?:answer|provide|help|find|access|determine)",
        r"\bi (?:do not|don't) have (?:access|enough|any|sufficient|the) (?:information|data|context|knowledge)",
        r"\bas an ai (?:language )?model",
        r"\bi'm sorry,? but\b",
        r"\bi am sorry,? but\b",
        r"\bno information (?:is )?available",
        r"\bunable to (?:answer|determine|verify)",
        r"\bthere is no (?:information|data) (?:about|on|regarding)",
    ]
] + [
    re.compile(p) for p in [
        r"আমি\s+(?:এই\s+)?(?:প্রশ্নের\s+)?উত্তর\s+দিতে\s+পারছি\s+না",
        r"আমার\s+কাছে\s+(?:এই\s+বিষয়ে\s+)?(?:কোনো\s+)?তথ্য\s+নেই",
        r"পর্যাপ্ত\s+তথ্য\s+নেই",
        r"একজন\s+এআই\s+(?:ভাষা\s+)?মডেল",
    ]
]


def is_refusal(text: str) -> bool:
    if not text:
        return False
    t = _norm(text)
    return any(p.search(t) for p in _REFUSAL_PATTERNS)


def lexicon_report(lang: str) -> dict:
    """Surface the provisional-lexicon warning wherever results are produced."""
    lex = load_lexicon(lang)
    meta = lex.get("_meta", {})
    return {
        "lang": lang,
        "status": meta.get("status", "unknown"),
        "n_strengtheners": len(lex.get("strengtheners", {})),
        "n_weakeners": len(lex.get("weakeners", {})),
        "publication_ready": "PROVISIONAL" not in meta.get("status", "").upper(),
        "required_action": meta.get("REQUIRED_ACTION"),
    }
