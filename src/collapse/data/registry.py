"""Dataset registry: HF ids -> unified schema.

Unified example schema (every loader must emit exactly this):
    uid        str          stable unique id, "{source}:{split}:{i}"
    lang       "en"|"bn"
    source     str          registry key
    task_type  "open_qa"|"mcq"
    question   str
    answers    list[str]    gold + aliases (open_qa); [gold_text] for mcq
    choices    list[str]|None
    answer_idx int|None     mcq only
    context    str|None
    meta       dict         e.g. PopQA popularity, BLUCK subcategory

VERIFY column: `checked` means the HF id was confirmed to exist as of 2026-08-18.
`unverified` means download will fail loudly with a message telling you the
suspected id -- fix the registry, do not silently substitute.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


@dataclass
class DatasetSpec:
    key: str
    hf_id: str
    lang: Literal["en", "bn"]
    task_type: Literal["open_qa", "mcq"]
    role: Literal["train", "eval", "both"]
    config: Optional[str] = None
    split: str = "train"
    mapper: str = "generic_qa"
    checked: bool = False
    licence: str = "see dataset card"
    note: str = ""
    # If the dataset must never enter training (plan 4b), this is enforced in prep.py
    holdout_only: bool = False


# --------------------------------------------------------------------------
# ENGLISH
# --------------------------------------------------------------------------
_EN = [
    DatasetSpec(
        key="popqa", hf_id="akariasai/PopQA", lang="en", task_type="open_qa",
        role="both", split="test", mapper="popqa", checked=True,
        licence="MIT",
        note="s_pop popularity field -> tail-knowledge buckets. Core H1 evidence: "
             "tail facts should die first. Only 'test' split exists upstream.",
    ),
    DatasetSpec(
        key="triviaqa", hf_id="mandarjoshi/trivia_qa", lang="en", task_type="open_qa",
        role="train", config="rc.nocontext", split="train", mapper="triviaqa",
        checked=True, licence="Apache-2.0",
        note="Rich alias lists -> good for lenient EM grading.",
    ),
    DatasetSpec(
        key="nq_open", hf_id="google-research-datasets/nq_open", lang="en",
        task_type="open_qa", role="train", split="train", mapper="nq_open",
        checked=True, licence="CC BY-SA 3.0",
    ),
    DatasetSpec(
        key="dolly", hf_id="databricks/databricks-dolly-15k", lang="en",
        task_type="open_qa", role="train", split="train", mapper="dolly",
        checked=True, licence="CC BY-SA 3.0",
        note="Instruction-style anchoring data for the real half of the SFT mix.",
    ),
    DatasetSpec(
        key="mmlu_en", hf_id="cais/mmlu", lang="en", task_type="mcq",
        role="eval", config="all", split="test", mapper="mmlu", checked=True,
        licence="MIT", note="English side of the parallel MMLU comparison.",
    ),
    DatasetSpec(
        key="alpaca", hf_id="tatsu-lab/alpaca", lang="en", task_type="open_qa",
        role="train", split="train", mapper="dolly", checked=True,
        licence="CC BY-NC 4.0",
        note="Alpaca is itself SYNTHETIC (self-instruct from text-davinci-003). "
             "Using it as 'real' anchoring data would quietly contaminate the "
             "generation-0 baseline. Prefer dolly (human-written) for the real half; "
             "keep alpaca only for a deliberately-contaminated control arm.",
    ),
]

# --------------------------------------------------------------------------
# BANGLA
# --------------------------------------------------------------------------
_BN = [
    DatasetSpec(
        key="squad_bn", hf_id="csebuetnlp/squad_bn", lang="bn", task_type="open_qa",
        role="train", split="train", mapper="squad_like", checked=False,
        licence="CC BY-NC-SA 4.0",
        note="NC licence -- fine for research, but check before releasing derived data.",
    ),
    DatasetSpec(
        key="banglarqa", hf_id="sartajekram/BanglaRQA", lang="bn", task_type="open_qa",
        role="train", split="train", mapper="banglarqa", checked=False,
        note="Has unanswerable questions; we drop them for the QA pool.",
    ),
    DatasetSpec(
        key="benqa", hf_id="LOCAL:data/raw/benqa.jsonl", lang="bn", task_type="mcq",
        role="both", split="test", mapper="benqa", checked=True,
        licence="see repo",
        note="NOT ON HUGGING FACE -- verified 2026-08-18. Distributed via GitHub: "
             "https://github.com/sheikhshafayat/BEnQA (ACL Findings 2024, 2403.10900). "
             "Clone it and convert to jsonl at the path above. Worth the manual step: "
             "the SAME item exists in bn and en, which makes it the cleanest "
             "matched-baseline instrument for H2.",
    ),
    DatasetSpec(
        key="bnmmlu", hf_id="hishab/bangla-mmlu", lang="bn", task_type="mcq",
        role="both", split="test", mapper="mmlu", checked=True,
        note="VERIFIED id (BanglaLLM/bangla-mmlu does NOT exist). 116k MCQs from "
             "Bangladeshi job/BCS/admission exams. Confirm the column names against "
             "the mmlu mapper on first load -- it is not a literal MMLU clone.",
    ),
    DatasetSpec(
        key="mmlu_bn_alt", hf_id="rishiraj/MMLU-BN", lang="bn", task_type="mcq",
        role="eval", split="test", mapper="mmlu", checked=True,
        note="Backup Bangla MMLU if hishab's schema proves awkward.",
    ),
    DatasetSpec(
        key="global_mmlu_bn", hf_id="CohereLabs/Global-MMLU", lang="bn",
        task_type="mcq", role="eval", config="bn", split="test",
        mapper="global_mmlu", checked=True, licence="Apache-2.0",
        note="VERIFIED, 14042 rows. Carries cultural_sensitivity_label (CS/CA), "
             "which splits the eval along exactly the axis the tail-knowledge "
             "argument needs. Its English counterpart is config 'en', so this also "
             "gives a second parallel bn/en instrument for H2.",
    ),
    DatasetSpec(
        key="global_mmlu_en", hf_id="CohereLabs/Global-MMLU", lang="en",
        task_type="mcq", role="eval", config="en", split="test",
        mapper="global_mmlu", checked=True, licence="Apache-2.0",
        note="English side of Global-MMLU -- item-matched with global_mmlu_bn.",
    ),
    DatasetSpec(
        key="mmmlu_bn", hf_id="openai/MMMLU", lang="bn", task_type="mcq",
        role="eval", config="BN_BD", split="test", mapper="mmmlu", checked=True,
        licence="MIT",
        note="Professionally translated MMLU. Config BN_BD. Pairs with mmlu_en for "
             "the parallel-knowledge axis.",
    ),
    DatasetSpec(
        key="bluck", hf_id="LOCAL:data/raw/bluck.jsonl", lang="bn", task_type="mcq",
        role="eval", split="test", mapper="local_jsonl", checked=True,
        holdout_only=True, licence="CC BY-NC-SA 4.0 (paper); confirm for data",
        note="NOT ON HUGGING FACE under any obvious id -- verified 2026-08-18, and "
             "the arXiv abstract page states no release URL. Paper: 2505.21092 / "
             "ACL Anthology 2025.banglalp-1.14 (BLP-2025). 2,366 MCQs, 23 categories "
             "of Bangladeshi culture, history and Bengali linguistics. ACTION: get "
             "the PDF's data statement or email the authors in week 1. This is the "
             "single most important bn eval set -- it is exactly the cultural tail an "
             "English-centric generator will not preserve -- so start the request early.",
    ),
    DatasetSpec(
        key="banglamedqa", hf_id="LOCAL:data/raw/banglamedqa.jsonl", lang="bn",
        task_type="mcq", role="eval", split="test", mapper="local_jsonl",
        checked=True, holdout_only=True,
        note="Local Bangla medical MCQ set, held out entirely (not used in the paper). "
             "Drop a jsonl here with fields: question, choices, answer_idx. Two public "
             "datasets named BanglaMedQA exist on the Hub (ajwad-abrar/BanglaMedQA, "
             "Oni279/BanglaMedQA); declare any overlap with them.",
    ),
    DatasetSpec(
        key="flores_bn", hf_id="openlanguagedata/flores_plus", lang="bn",
        task_type="open_qa", role="eval", config="ben_Beng", split="dev",
        mapper="flores", checked=True, licence="CC BY-SA 4.0",
        note="GATED -- verified 2026-08-18: requires `huggingface-cli login` plus "
             "accepting the terms on the dataset page. Used only for held-out "
             "perplexity, so it is optional; skip it if you do not want the token "
             "dance. Ungated fallback for register-split PPL: hold out a slice of "
             "squad_bn contexts instead.",
    ),
    DatasetSpec(
        key="banglarqa_test", hf_id="sartajekram/BanglaRQA", lang="bn",
        task_type="open_qa", role="eval", split="test", mapper="banglarqa",
        checked=True,
        note="Held-out Bangla OPEN QA. The pilot needs this because Bangla MCQ is "
             "unusable for the hedging half of H1: measured hedge rate on Bangla "
             "MMLU was 0.001, i.e. there is no hedging to degrade. Free-text QA "
             "elicits epistemic markers. Train on squad_bn, evaluate here -- "
             "different corpora, so no contamination.",
    ),
    DatasetSpec(
        key="indicqa_bn", hf_id="ai4bharat/IndicQA", lang="bn", task_type="open_qa",
        role="train", config="indicqa.bn", split="test", mapper="squad_like",
        checked=False,
        note="BROKEN via datasets>=5: the loading script is unsupported and the "
             "auto-parquet branch exposes only a 'default' config, losing the "
             "per-language split. Not worth fixing -- squad_bn alone yields 68k "
             "Bangla QA pairs. Left registered only as a documented dead end.",
    ),
]

REGISTRY: dict[str, DatasetSpec] = {s.key: s for s in (_EN + _BN)}


def resolve(key: str) -> DatasetSpec:
    if key not in REGISTRY:
        raise KeyError(
            f"unknown dataset key {key!r}. Known: {sorted(REGISTRY)}"
        )
    return REGISTRY[key]


def by_lang(lang: str, role: str | None = None) -> list[DatasetSpec]:
    out = [s for s in REGISTRY.values() if s.lang == lang]
    if role:
        out = [s for s in out if s.role in (role, "both")]
    return out


def unverified() -> list[DatasetSpec]:
    return [s for s in REGISTRY.values() if not s.checked]
