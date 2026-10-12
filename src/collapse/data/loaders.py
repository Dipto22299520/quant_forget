"""HF row -> unified schema mappers, plus an offline synthetic corpus for smoke tests."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable

from ..utils import LOG, read_jsonl
from .registry import DatasetSpec, resolve

_LETTERS = "ABCDEFGHIJ"


def _uid(source: str, split: str, i: int) -> str:
    return f"{source}:{split}:{i}"


def _clean(s: Any) -> str:
    return str(s).strip() if s is not None else ""


def _is_script_error(e: Exception) -> bool:
    msg = str(e).lower()
    return ("dataset scripts are no longer supported" in msg
            or "loading script" in msg
            or "trust_remote_code" in msg)


def _struct(v: Any) -> Any:
    """Parquet-branch conversions sometimes stringify struct columns.

    So `answers` can arrive as a real dict OR as "{'text': [...], ...}". Callers
    must not care which, or a dataset silently yields zero usable rows.
    """
    if isinstance(v, str) and v.startswith(("{", "[")):
        try:
            import ast
            return ast.literal_eval(v)
        except (ValueError, SyntaxError):
            return v
    return v


def _truthy(v: Any) -> bool:
    """HF columns arrive as bool, int, or the STRINGS '1'/'0'/'true'/'false'.

    BanglaRQA ships is_answerable as the string '1', so a plain `is False` test
    silently keeps every unanswerable question in the pool.
    """
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "t"}
    return bool(v)


# --------------------------------------------------------------------------
# Mappers. Each takes (row, spec, index) -> dict | None (None = drop the row)
# --------------------------------------------------------------------------

def map_generic_qa(row, spec, i):
    q = _clean(row.get("question") or row.get("query") or row.get("instruction"))
    a = row.get("answer") or row.get("answers") or row.get("response")
    if isinstance(a, dict):                       # SQuAD style {"text": [...]}
        a = a.get("text", [])
    if isinstance(a, str):
        a = [a]
    answers = [_clean(x) for x in (a or []) if _clean(x)]
    if not q or not answers:
        return None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=q, answers=answers, choices=None,
                answer_idx=None, context=_clean(row.get("context")) or None, meta={})


def map_popqa(row, spec, i):
    q = _clean(row.get("question"))
    poss = row.get("possible_answers")
    if isinstance(poss, str):
        # PopQA ships this column as a JSON-ish string
        import json
        try:
            poss = json.loads(poss)
        except Exception:
            poss = [poss]
    answers = [_clean(x) for x in (poss or []) if _clean(x)]
    obj = _clean(row.get("obj"))
    if obj and obj not in answers:
        answers.insert(0, obj)
    if not q or not answers:
        return None
    s_pop = row.get("s_pop")
    try:
        s_pop = float(s_pop) if s_pop is not None else None
    except (TypeError, ValueError):
        s_pop = None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=q, answers=answers, choices=None,
                answer_idx=None, context=None,
                meta={"s_pop": s_pop, "prop": _clean(row.get("prop")),
                      "subj": _clean(row.get("subj"))})


def map_triviaqa(row, spec, i):
    q = _clean(row.get("question"))
    ans = _struct(row.get("answer")) or {}
    gold = _clean(ans.get("value") if isinstance(ans, dict) else ans)
    aliases = ans.get("aliases", []) if isinstance(ans, dict) else []
    answers = [gold] + [_clean(x) for x in aliases if _clean(x)]
    answers = [a for a in answers if a]
    if not q or not answers:
        return None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=q, answers=answers[:20], choices=None,
                answer_idx=None, context=None, meta={})


def map_nq_open(row, spec, i):
    q = _clean(row.get("question"))
    answers = [_clean(x) for x in (row.get("answer") or []) if _clean(x)]
    if not q or not answers:
        return None
    if not q.endswith("?"):
        q += "?"
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=q, answers=answers, choices=None,
                answer_idx=None, context=None, meta={})


def map_dolly(row, spec, i):
    q = _clean(row.get("instruction"))
    ctx = _clean(row.get("context")) or None
    a = _clean(row.get("response"))
    if not q or not a:
        return None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=q, answers=[a], choices=None,
                answer_idx=None, context=ctx,
                meta={"category": _clean(row.get("category"))})


def map_squad_like(row, spec, i):
    q = _clean(row.get("question"))
    ans = _struct(row.get("answers")) or {}
    texts = ans.get("text", []) if isinstance(ans, dict) else ans
    answers = [_clean(x) for x in (texts or []) if _clean(x)]
    if not q or not answers:
        return None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=q, answers=answers, choices=None,
                answer_idx=None, context=_clean(row.get("context")) or None, meta={})


def map_banglarqa(row, spec, i, drop_confirmation: bool = True):
    """BanglaRQA -> unified schema.

    Real schema (verified 2026-08-18 via the parquet branch):
      question_text, is_answerable ('1'/'0' STRINGS), question_type,
      answers = {'answer_text': [...], 'answer_type': [...]}

    Two filters that matter for calibration:
      * unanswerable questions have no gold span -- they are not a QA signal here
      * question_type == 'confirmation' is yes/no ('হ্যাঁ'/'না'). A coin flip scores
        ~50% and would inflate baseline accuracy while making confidence
        meaningless, so they are dropped by default.
    """
    q = _clean(row.get("question_text") or row.get("question"))
    if not _truthy(row.get("is_answerable", row.get("answerable", True))):
        return None
    qtype = _clean(row.get("question_type"))
    if drop_confirmation and qtype.lower() == "confirmation":
        return None

    a = _struct(row.get("answers") or row.get("answer"))
    if isinstance(a, dict):
        a = a.get("answer_text") or a.get("text") or []
    if isinstance(a, str):
        a = [a]
    answers = [_clean(x) for x in (a or []) if _clean(x)]
    if not q or not answers:
        return None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=q, answers=answers, choices=None,
                answer_idx=None,
                context=_clean(row.get("context") or row.get("passage")) or None,
                meta={"qtype": qtype})


def _mcq(row, spec, i, q_key, choices, ans_field, meta=None):
    q = _clean(q_key)
    choices = [_clean(c) for c in choices if _clean(c) != ""]
    if not q or len(choices) < 2:
        return None
    idx = None
    if isinstance(ans_field, int):
        idx = ans_field
    elif isinstance(ans_field, str):
        s = ans_field.strip()
        if len(s) == 1 and s.upper() in _LETTERS:
            idx = _LETTERS.index(s.upper())
        elif s in choices:
            idx = choices.index(s)
        elif s.isdigit():
            idx = int(s)
    if idx is None or not (0 <= idx < len(choices)):
        return None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="mcq", question=q, answers=[choices[idx]], choices=choices,
                answer_idx=idx, context=None, meta=meta or {})


def map_mmlu(row, spec, i):
    ch = row.get("choices") or row.get("options") or [
        row.get("A"), row.get("B"), row.get("C"), row.get("D")]
    return _mcq(row, spec, i, row.get("question") or row.get("Question"), ch,
                row.get("answer") if row.get("answer") is not None else row.get("Answer"),
                meta={"subject": _clean(row.get("subject") or row.get("Subject"))})


def map_mmmlu(row, spec, i):
    ch = [row.get("A"), row.get("B"), row.get("C"), row.get("D")]
    return _mcq(row, spec, i, row.get("Question"), ch, row.get("Answer"),
                meta={"subject": _clean(row.get("Subject"))})


def map_global_mmlu(row, spec, i):
    """Global-MMLU (verified schema 2026-08-18).

    Columns: question, option_a..option_d, answer ('A'-'D'), subject,
             subject_category, cultural_sensitivity_label, is_annotated

    cultural_sensitivity_label ('CS' culturally-sensitive / 'CA' culturally-
    agnostic) is carried into meta because it splits the eval exactly along the
    axis the tail-knowledge argument needs: an English-centric generator should
    damage culturally-sensitive items first.
    """
    ch = [row.get("option_a"), row.get("option_b"),
          row.get("option_c"), row.get("option_d")]
    ex = _mcq(row, spec, i, row.get("question"), ch, row.get("answer"),
              meta={"subject": _clean(row.get("subject")),
                    "subject_category": _clean(row.get("subject_category")),
                    "cultural_sensitivity": _clean(row.get("cultural_sensitivity_label")),
                    "is_annotated": _clean(row.get("is_annotated"))})
    return ex


def map_benqa(row, spec, i):
    q = row.get("question_bn") if spec.lang == "bn" else row.get("question_en")
    q = q or row.get("question")
    ch = row.get("options_bn") if spec.lang == "bn" else row.get("options_en")
    ch = ch or row.get("options") or row.get("choices") or []
    return _mcq(row, spec, i, q, ch, row.get("answer"),
                meta={"subject": _clean(row.get("subject")),
                      "parallel_id": _clean(row.get("id"))})


def map_flores(row, spec, i):
    """Flores is raw sentences -- used for perplexity, not QA."""
    text = _clean(row.get("sentence") or row.get("text"))
    if not text:
        return None
    return dict(uid=_uid(spec.key, spec.split, i), lang=spec.lang, source=spec.key,
                task_type="open_qa", question=text, answers=[text], choices=None,
                answer_idx=None, context=None, meta={"raw_text": True})


def map_local_jsonl(row, spec, i):
    if row.get("choices"):
        return _mcq(row, spec, i, row.get("question"), row["choices"],
                    row.get("answer_idx", row.get("answer")), meta=row.get("meta", {}))
    return map_generic_qa(row, spec, i)


MAPPERS: dict[str, Callable] = {
    "generic_qa": map_generic_qa, "popqa": map_popqa, "triviaqa": map_triviaqa,
    "nq_open": map_nq_open, "dolly": map_dolly, "squad_like": map_squad_like,
    "banglarqa": map_banglarqa, "mmlu": map_mmlu, "mmmlu": map_mmmlu,
    "benqa": map_benqa, "flores": map_flores, "local_jsonl": map_local_jsonl,
    "global_mmlu": map_global_mmlu,
}


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_dataset_key(key: str, limit: int | None = None, cache_dir: str | None = None,
                     split: str | None = None) -> list[dict]:
    """Load one registry key into the unified schema."""
    spec = resolve(key)
    split = split or spec.split

    if spec.hf_id.startswith("LOCAL:"):
        path = Path(spec.hf_id[len("LOCAL:"):])
        if not path.exists():
            raise FileNotFoundError(
                f"[{key}] expects a local file at {path}. {spec.note}"
            )
        rows = read_jsonl(path)
    elif spec.hf_id.startswith("SYNTHETIC:"):
        rows = synthetic_rows(spec.lang, limit or 500)
    else:
        try:
            from datasets import load_dataset
        except ImportError as e:
            raise ImportError("pip install datasets") from e
        try:
            ds = load_dataset(spec.hf_id, spec.config, split=split, cache_dir=cache_dir)
        except Exception as e:
            # datasets>=5 removed loading-script support, which breaks several
            # older Bangla sets (squad_bn, BanglaRQA, IndicQA). The Hub keeps an
            # auto-converted parquet branch for exactly these repos, so retry
            # there before giving up.
            if _is_script_error(e):
                LOG.warning("%s uses a loading script (unsupported in datasets>=5); "
                            "retrying via refs/convert/parquet", key)
                try:
                    ds = load_dataset(spec.hf_id, spec.config, split=split,
                                      cache_dir=cache_dir,
                                      revision="refs/convert/parquet")
                except Exception as e2:
                    raise RuntimeError(
                        f"failed to load {key} ({spec.hf_id}) from both the normal "
                        f"path and the parquet branch: {e2}"
                    ) from e2
            else:
                hint = "" if spec.checked else (
                    f"\n  This id was NEVER VERIFIED. {spec.note}"
                    f"\n  Fix src/collapse/data/registry.py rather than guessing at runtime."
                )
                raise RuntimeError(
                    f"failed to load {key} ({spec.hf_id}, config={spec.config}, "
                    f"split={split}): {e}{hint}"
                ) from e
        rows = ds

    mapper = MAPPERS[spec.mapper]
    out, dropped = [], 0
    for i, row in enumerate(rows):
        if limit is not None and len(out) >= limit:
            break
        try:
            ex = mapper(dict(row), spec, i)
        except Exception:
            ex = None
        if ex is None:
            dropped += 1
            continue
        out.append(ex)
    LOG.info("loaded %-12s n=%-6d dropped=%-5d lang=%s type=%s",
             key, len(out), dropped, spec.lang, spec.task_type)
    if dropped > 0 and len(out) == 0:
        raise RuntimeError(
            f"[{key}] every row was dropped by mapper {spec.mapper!r} -- the upstream "
            f"schema has changed. Inspect one row and fix the mapper."
        )
    return out


def load_many(keys: list[str], limit_each: int | None = None,
              cache_dir: str | None = None) -> list[dict]:
    out = []
    for k in keys:
        out.extend(load_dataset_key(k, limit=limit_each, cache_dir=cache_dir))
    return out


# --------------------------------------------------------------------------
# Offline synthetic corpus -- lets the whole loop be smoke-tested with zero
# downloads. NOT for real results; it exists so `run_smoke` is meaningful.
# --------------------------------------------------------------------------

_SYNTH_EN = [
    ("What is the capital of France?", ["Paris"]),
    ("Who wrote the play Hamlet?", ["William Shakespeare", "Shakespeare"]),
    ("What is the chemical symbol for gold?", ["Au"]),
    ("How many continents are there?", ["seven", "7"]),
    ("What is the largest planet in the solar system?", ["Jupiter"]),
    ("In which year did the Second World War end?", ["1945"]),
    ("What is the tallest mountain on Earth?", ["Mount Everest", "Everest"]),
    ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "da Vinci"]),
    ("What is the freezing point of water in Celsius?", ["0", "zero"]),
    ("Which ocean is the largest?", ["Pacific", "the Pacific Ocean"]),
]

_SYNTH_BN = [
    ("ফ্রান্সের রাজধানীর নাম কী?", ["প্যারিস"]),
    ("বাংলাদেশের জাতীয় কবি কে?", ["কাজী নজরুল ইসলাম", "নজরুল"]),
    ("সোনার রাসায়নিক প্রতীক কী?", ["Au"]),
    ("পৃথিবীতে মোট কয়টি মহাদেশ আছে?", ["সাতটি", "৭"]),
    ("সৌরজগতের বৃহত্তম গ্রহ কোনটি?", ["বৃহস্পতি"]),
    ("দ্বিতীয় বিশ্বযুদ্ধ কোন সালে শেষ হয়?", ["১৯৪৫", "1945"]),
    ("পৃথিবীর সর্বোচ্চ পর্বতশৃঙ্গের নাম কী?", ["মাউন্ট এভারেস্ট", "এভারেস্ট"]),
    ("'গীতাঞ্জলি' কাব্যগ্রন্থের রচয়িতা কে?", ["রবীন্দ্রনাথ ঠাকুর", "রবীন্দ্রনাথ"]),
    ("পানির হিমাঙ্ক সেলসিয়াসে কত?", ["০", "0", "শূন্য"]),
    ("বিশ্বের বৃহত্তম মহাসাগর কোনটি?", ["প্রশান্ত মহাসাগর", "প্রশান্ত"]),
]


# Disjoint eval half. The offline fixture MUST NOT share questions with the
# training half, or the contamination guard in prep.py fires -- correctly.
_SYNTH_EN_EVAL = [
    ("Which planet is known as the Red Planet?", ["Mars"]),
    ("Who developed the theory of general relativity?", ["Albert Einstein", "Einstein"]),
    ("What is the chemical symbol for iron?", ["Fe"]),
    ("How many strings does a standard violin have?", ["four", "4"]),
    ("What is the longest river in Africa?", ["Nile", "the Nile"]),
    ("In which city is the Colosseum located?", ["Rome"]),
    ("What gas do plants absorb from the atmosphere?", ["carbon dioxide", "CO2"]),
    ("Who wrote the novel Pride and Prejudice?", ["Jane Austen", "Austen"]),
    ("What is the boiling point of water in Celsius?", ["100"]),
    ("Which country gifted the Statue of Liberty to the USA?", ["France"]),
]

_SYNTH_BN_EVAL = [
    ("কোন গ্রহকে লাল গ্রহ বলা হয়?", ["মঙ্গল"]),
    ("আপেক্ষিকতার সাধারণ তত্ত্ব কে আবিষ্কার করেন?", ["আলবার্ট আইনস্টাইন", "আইনস্টাইন"]),
    ("লোহার রাসায়নিক প্রতীক কী?", ["Fe"]),
    ("একটি সাধারণ বেহালায় কয়টি তার থাকে?", ["চারটি", "৪"]),
    ("আফ্রিকার দীর্ঘতম নদীর নাম কী?", ["নীল নদ", "নীল"]),
    ("কলোসিয়াম কোন শহরে অবস্থিত?", ["রোম"]),
    ("গাছ বাতাস থেকে কোন গ্যাস গ্রহণ করে?", ["কার্বন ডাই অক্সাইড"]),
    ("'পথের পাঁচালী' উপন্যাসের লেখক কে?", ["বিভূতিভূষণ বন্দ্যোপাধ্যায়", "বিভূতিভূষণ"]),
    ("পানির স্ফুটনাঙ্ক সেলসিয়াসে কত?", ["১০০", "100"]),
    ("যুক্তরাষ্ট্রকে স্ট্যাচু অব লিবার্টি কোন দেশ উপহার দিয়েছিল?", ["ফ্রান্স"]),
]


def _synth_base(lang: str, split: str) -> list[tuple[str, list[str]]]:
    if split == "eval":
        return _SYNTH_BN_EVAL if lang == "bn" else _SYNTH_EN_EVAL
    return _SYNTH_BN if lang == "bn" else _SYNTH_EN


def synthetic_rows(lang: str, n: int, split: str = "train") -> list[dict]:
    """Deterministically inflate the seed list to n rows."""
    base = _synth_base(lang, split)
    return [{"question": base[i % len(base)][0], "answers": base[i % len(base)][1]}
            for i in range(n)]


def synthetic_examples(lang: str, n: int, source: str = "synthetic",
                       split: str = "train") -> list[dict]:
    """Unified-schema synthetic examples with a fake popularity field.

    split="train" and split="eval" draw from DISJOINT question lists so the
    offline smoke test exercises the real held-out code path.
    """
    base = _synth_base(lang, split)
    out = []
    for i in range(n):
        q, a = base[i % len(base)]
        # deterministic pseudo-popularity so tail-bucket code paths are exercised
        h = int(hashlib.md5(f"{lang}{split}{i}".encode()).hexdigest()[:6], 16)
        out.append(dict(uid=f"{source}:{lang}:{i}", lang=lang, source=source,
                        task_type="open_qa", question=q, answers=list(a), choices=None,
                        answer_idx=None, context=None,
                        meta={"s_pop": float(h % 10000) + 1.0, "synthetic_seed": True}))
    return out
