"""Prompt construction.

Three styles, and the distinction between them is load-bearing:

  hedged     -- PERMITS uncertainty, does not demand it. This is the main condition.
                The prompt must never instruct the model to hedge, or the marker
                distribution measures our instruction rather than the model's
                epistemic state, and H1/H3 become circular.
  bare       -- forbids hedging. Control: isolates accuracy from expression.
  verbalized -- asks for an explicit numeric confidence. Feeds the verbalized
                calibration baseline (plan 4c).

Bangla prompts are written natively, not translated, for the same reason the
lexicon must be: a translated instruction shifts register and changes which
markers the model reaches for.
"""
from __future__ import annotations

PROMPT_STYLES = ("hedged", "bare", "verbalized")

_SYS = {
    ("en", "hedged"): (
        "You are a helpful assistant. Answer the question as accurately as you can. "
        "Express however much or little certainty you actually have."
    ),
    ("en", "bare"): (
        "You are a helpful assistant. Answer the question with the answer only. "
        "Do not explain and do not comment on your certainty."
    ),
    ("en", "verbalized"): (
        "You are a helpful assistant. Answer the question, then on a new line write "
        "'Confidence: N%' where N is your probability that your answer is correct."
    ),
    ("bn", "hedged"): (
        "তুমি একজন সহায়ক সহকারী। প্রশ্নের উত্তর যতটা সম্ভব সঠিকভাবে দাও। "
        "তোমার প্রকৃত নিশ্চয়তা যতটুকু, ঠিক ততটুকুই প্রকাশ করো।"
    ),
    ("bn", "bare"): (
        "তুমি একজন সহায়ক সহকারী। শুধু উত্তরটি লেখো। "
        "কোনো ব্যাখ্যা দিও না এবং নিশ্চয়তা নিয়ে কোনো মন্তব্য কোরো না।"
    ),
    ("bn", "verbalized"): (
        "তুমি একজন সহায়ক সহকারী। প্রশ্নের উত্তর দাও, তারপর নতুন লাইনে "
        "'আত্মবিশ্বাস: N%' লেখো, যেখানে N হলো তোমার উত্তর সঠিক হওয়ার সম্ভাবনা।"
    ),
}

_Q_LABEL = {"en": ("Question", "Answer"), "bn": ("প্রশ্ন", "উত্তর")}

_MCQ_INSTR = {
    "en": "Choose the correct option and reply with its letter.",
    "bn": "সঠিক বিকল্পটি বেছে নাও এবং তার অক্ষরটি লেখো।",
}

_LETTERS = "ABCDEFGHIJ"


def _user_block(ex: dict, lang: str) -> str:
    q_lab, _ = _Q_LABEL.get(lang, _Q_LABEL["en"])
    parts = []
    if ex.get("context"):
        parts.append(ex["context"].strip())
    parts.append(f"{q_lab}: {ex['question'].strip()}")
    if ex.get("task_type") == "mcq" and ex.get("choices"):
        opts = "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(ex["choices"]))
        parts.append(opts)
        parts.append(_MCQ_INSTR.get(lang, _MCQ_INSTR["en"]))
    return "\n".join(parts)


def build_prompt(ex: dict, style: str = "hedged", tokenizer=None,
                 use_chat_template: bool = True) -> str:
    """Render one example into a prompt string ready for tokenisation.

    If the record already carries a `prompt` (i.e. it came back from
    generate_batch), that exact string is reused. Re-rendering would risk
    producing a DIFFERENT prompt than the one the answer was generated from --
    which is how `context` was previously lost between generation and training.
    """
    preserved = ex.get("prompt")
    if preserved:
        return preserved

    if style not in PROMPT_STYLES:
        raise ValueError(f"style must be one of {PROMPT_STYLES}, got {style!r}")
    lang = ex.get("lang", "en")
    system = _SYS.get((lang, style)) or _SYS[("en", style)]
    user = _user_block(ex, lang)

    has_template = (
        use_chat_template and tokenizer is not None
        and getattr(tokenizer, "chat_template", None)
    )
    if has_template:
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
        # REASONING MODELS: Qwen3 and friends emit a <think>...</think> block
        # before answering. With a short generation budget the entire budget is
        # spent on the preamble and the answer never appears -- which silently
        # turns every calibration measurement into a measurement of reasoning
        # text. `enable_thinking=False` is the documented switch; templates that
        # do not know the kwarg raise, so we retry without it.
        for kwargs in ({"enable_thinking": False}, {}):
            try:
                return tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **kwargs)
            except Exception:
                continue
        # some templates reject a system role entirely; fall through to plain text

    _, a_lab = _Q_LABEL.get(lang, _Q_LABEL["en"])
    return f"{system}\n\n{user}\n{a_lab}:"


# Slack left when trimming a passage, so that re-tokenising the rebuilt prompt
# (whose token boundaries can shift by a few tokens) still fits.
_TRIM_MARGIN = 8


def fit_prompt(ex: dict, style: str = "hedged", tokenizer=None,
               max_len: int | None = None, use_chat_template: bool = True,
               ) -> tuple[str, bool]:
    """build_prompt, then trim the CONTEXT passage until the prompt fits max_len.

    Returns (prompt, trimmed). Why this exists: the tokenizer's own truncation
    is positional. Right-truncation drops the END of an over-long prompt -- the
    question and the assistant-turn marker -- so the model never sees a question
    and simply continues the passage. Left-truncation drops the system prompt
    instead. Bangla reading-comprehension prompts overflowed a 768-token window
    75% of the time; English closed-book prompts never did, which turned a
    tokenizer default into a language-correlated corruption of the self-training
    data (measured: 75% of Bangla synthetic targets were passage continuation,
    not answers). Trimming the passage keeps system prompt, question and marker
    intact in every language, and the flag lets the paper report how often it
    happened.

    Preserved prompts (records that came back from generate_batch) are returned
    untouched: they were fitted when they were generated.
    """
    prompt = build_prompt(ex, style=style, tokenizer=tokenizer,
                          use_chat_template=use_chat_template)
    if tokenizer is None or not max_len or ex.get("prompt"):
        return prompt, False
    n = len(tokenizer.encode(prompt))
    if n <= max_len or not ex.get("context"):
        return prompt, False

    ctx_ids = tokenizer.encode(ex["context"], add_special_tokens=False)
    trimmed = dict(ex)
    trimmed.pop("prompt", None)
    for _ in range(4):
        keep = len(ctx_ids) - (n - max_len + _TRIM_MARGIN)
        if keep <= 0:
            # Cannot fit even with the passage gone; leave the remainder to the
            # tokenizer's (left) safety net rather than emit an empty passage.
            break
        ctx_ids = ctx_ids[:keep]
        # A cut inside a multi-byte character decodes to U+FFFD; drop it.
        trimmed["context"] = tokenizer.decode(ctx_ids).replace("\ufffd", "").strip()
        prompt = build_prompt(trimmed, style=style, tokenizer=tokenizer,
                              use_chat_template=use_chat_template)
        n = len(tokenizer.encode(prompt))
        if n <= max_len:
            break
    return prompt, True


def build_sft_text(ex: dict, response: str, style: str = "hedged", tokenizer=None,
                   use_chat_template: bool = True,
                   max_len: int | None = None) -> tuple[str, str]:
    """-> (prompt, completion). Loss is masked to the completion only."""
    prompt, _ = fit_prompt(ex, style=style, tokenizer=tokenizer, max_len=max_len,
                           use_chat_template=use_chat_template)
    completion = response.strip()
    if tokenizer is not None and getattr(tokenizer, "eos_token", None):
        completion = completion + tokenizer.eos_token
    return prompt, completion


def gold_response(ex: dict, lang: str | None = None) -> str:
    """The 'real data' target for generation-0 / anchoring examples."""
    if ex.get("task_type") == "mcq" and ex.get("answer_idx") is not None:
        idx = ex["answer_idx"]
        return f"{_LETTERS[idx]}. {ex['choices'][idx]}"
    return ex["answers"][0]
