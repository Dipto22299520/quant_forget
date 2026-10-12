"""Batched generation that also returns the confidence signals we need.

CRITICAL: we request `output_logits=True`, not `output_scores=True`.
`scores` are logits AFTER temperature / top-p warping. Calibrating against warped
scores would make ECE a function of the sampling config rather than the model, and
every number in the paper would be wrong in a way that is very hard to notice.
`logits` are the raw pre-warper distribution.
"""
from __future__ import annotations

import math
import re
from typing import Any

import torch
import torch.nn.functional as F

from ..utils import LOG, batched, pbar
from .prompts import build_prompt, fit_prompt

_LETTERS = "ABCDEFGHIJ"

# Sequences that mean the model has stopped answering and started hallucinating
# another turn of the conversation. Without these, a base model happily continues
# "Answer:\nsystem\nYou are a helpful assistant.\nuser\n..." and that text then
# pollutes marker extraction, diversity, and grading alike.
STOP_STRINGS = [
    "\nQuestion:", "\nQ:", "\nAnswer:", "\nA:",
    "\nsystem", "\nuser", "\nassistant",
    "<|im_end|>", "<|im_start|>", "<|endoftext|>", "</s>",
    "\nপ্রশ্ন:", "\nউত্তর:",
]

# Whether to ALSO stop generation early inside HF's loop. Off by default: it is
# pure overhead here, because truncate_response() already yields the same text.
# Measured on this machine (Qwen1.5-0.5B, 64 items, 48 new tokens, batch 8):
# early stopping was slower than generating in full and cutting afterwards.
USE_STOP_STRINGS = False


# Weak/base models echo the answer cue before answering. Left in place it
# pollutes marker extraction and diversity, and it is not part of the answer.
_ECHO_PREFIX = re.compile(r"^\s*(?:answer|ans|উত্তর)\s*[:：\-]\s*", re.IGNORECASE)


_THINK_CLOSED = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Remove <think>...</think> chain-of-thought before any metric sees it.

    A CLOSED block is dropped and the answer after it kept. An UNCLOSED block
    means the generation budget ran out mid-thought, so there is no answer at
    all -- and the honest representation of that is an empty string, not the
    reasoning text. Scoring the preamble would measure the model's musings
    rather than its answer.
    """
    if not text or "<think>" not in text.lower():
        return text
    text = _THINK_CLOSED.sub(" ", text)
    text = _THINK_OPEN.sub(" ", text)
    return text.strip()


def truncate_response(text: str) -> str:
    """Strip reasoning, cut at the first hallucinated turn boundary, drop cues."""
    if not text:
        return text
    text = strip_reasoning(text)
    if not text:
        return ""
    cut = len(text)
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            cut = min(cut, i)
    out = text[:cut].strip()
    # Strip a leading "Answer:" echo. If that is ALL the model produced, the
    # response is empty -- which is the honest reading, and grading/markers must
    # see it as empty rather than as a confident bare assertion.
    out = _ECHO_PREFIX.sub("", out).strip()
    # A generation budget that ends inside a multi-byte character decodes to
    # U+FFFD. Left in, it becomes a real token in the SFT target and the model
    # learns to emit it; Bangla (3-byte characters) hit this on a third of all
    # targets, English never did. It carries no text, so it is dropped.
    out = out.strip("\ufffd").strip()
    return out


class EntropyNeutralTopP:
    """Logits processor: top-p sampling at the temperature that preserves entropy.

    Self-training on exact samples from p has zero expected gradient, so the
    loop's systematic sharpening comes from the sampler: top-p truncation makes
    every step whose top token already has p >= top_p greedy, and the targets
    are sharper than the model. Here, per step and per row, T* is chosen by
    bisection so that H(TopP(softmax(z/T*))) = H(softmax(z)); the returned
    scores are z/T* with every token outside the nucleus set to -inf, so
    sampling from softmax(scores) is exactly the entropy-matched truncated q.
    Use with temperature=1.0 and top_p=1.0 so no other warper interferes.
    """

    def __init__(self, top_p: float = 0.95, t_max: float = 8.0, iters: int = 22):
        self.top_p, self.t_max, self.iters = top_p, t_max, iters
        self.t_sum, self.t_n = 0.0, 0

    def _trunc(self, zs: torch.Tensor, T: torch.Tensor):
        lp = torch.log_softmax(zs / T[:, None], dim=-1)
        p = lp.exp()
        keep = (p.cumsum(-1) - p) < self.top_p
        q = p * keep
        Z = q.sum(-1, keepdim=True)
        lq = torch.where(keep, lp - Z.log(), torch.zeros_like(lp))
        return -(q / Z * lq).sum(-1), keep

    @torch.no_grad()
    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        z = scores.float()
        lp = torch.log_softmax(z, -1)
        H = -(lp.exp() * lp).sum(-1)
        zs, idx = z.sort(-1, descending=True)
        lo = torch.ones(z.shape[0], device=z.device)
        hi = torch.full_like(lo, self.t_max)
        for _ in range(self.iters):              # entropy of q_T rises with T
            mid = (lo + hi) / 2
            h, _ = self._trunc(zs, mid)
            up = h < H
            lo = torch.where(up, mid, lo)
            hi = torch.where(up, hi, mid)
        T = (lo + hi) / 2
        _, keep_sorted = self._trunc(zs, T)
        keep = torch.zeros_like(keep_sorted).scatter(-1, idx, keep_sorted)
        out = torch.where(keep, z / T[:, None], torch.full_like(z, float("-inf")))
        self.t_sum += float(T.sum()); self.t_n += int(T.numel())
        return out.to(scores.dtype)


def _valid_mask(gen_ids: torch.Tensor, eos_ids: list[int]) -> torch.Tensor:
    """True for real generated tokens, up to and including the first EOS."""
    is_eos = torch.zeros_like(gen_ids, dtype=torch.bool)
    for e in eos_ids:
        is_eos |= gen_ids == e
    after_first = (is_eos.cumsum(dim=1) - is_eos.long()) > 0
    return ~after_first


def _kept_token_mask(tokenizer, ids, valid_mask, raw: str, kept: str):
    """Map the KEPT text span back to the exact generated tokens that produced it.

    Taking the first N tokens is only valid for suffix truncation. When a PREFIX
    is removed -- a <think> block, or an echoed "Answer:" cue -- the first N
    tokens are the preamble, so its log probabilities get reported as if they
    were the answer's. That silently mis-measures every confidence signal.

    Here we locate the kept text inside the raw decode, then convert that
    character span to token indices using cumulative incremental decodes, so the
    retained span is the real one.

    Returns (mask, ok). ok=False means the span was not contiguously locatable
    and the caller should treat the row's token statistics as unavailable rather
    than silently wrong.
    """
    import torch as _t

    if not kept:
        return _t.zeros_like(valid_mask), True          # genuinely no answer

    start = raw.find(kept)
    if start < 0:
        stripped = raw.strip()
        start = stripped.find(kept)
        if start < 0:
            return valid_mask, False                    # not locatable
        start += len(raw) - len(raw.lstrip())
    end = start + len(kept)

    idx = _t.nonzero(valid_mask, as_tuple=False).flatten().tolist()
    if not idx:
        return valid_mask, False

    keep_positions, prev = [], 0
    for k, pos in enumerate(idx):
        cur = len(tokenizer.decode(ids[idx[0]:pos + 1], skip_special_tokens=True))
        # token covers characters [prev, cur) of the decoded generation
        if cur > start and prev < end:
            keep_positions.append(pos)
        prev = cur

    if not keep_positions:
        return valid_mask, False

    out = _t.zeros_like(valid_mask)
    out[_t.tensor(keep_positions, device=valid_mask.device)] = True
    return out, True


@torch.no_grad()
def generate_batch(model, tokenizer, examples: list[dict], style: str = "hedged",
                   max_new_tokens: int = 48, temperature: float = 0.7,
                   top_p: float = 0.9, do_sample: bool = True, batch_size: int = 16,
                   max_len: int = 1024, seed: int | None = None,
                   use_chat_template: bool = True,
                   use_stop_strings: bool | None = None,
                   sampler: str = "standard") -> list[dict]:
    """Generate one response per example, with per-token confidence statistics.

    sampler: standard | ancestral | entropy_neutral (see GenCfg.sampler). The
    recorded token statistics are always the model's own T=1 probabilities
    (raw logits), whatever the sampler.
    """
    if sampler not in ("standard", "ancestral", "entropy_neutral"):
        raise ValueError(f"unknown sampler {sampler!r}")
    en_proc = EntropyNeutralTopP(top_p=top_p) if sampler == "entropy_neutral" else None
    device = next(model.parameters()).device
    eos_ids = [tokenizer.eos_token_id]
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id not in eos_ids:
        eos_ids.append(tokenizer.pad_token_id)
    eos_ids = [e for e in eos_ids if e is not None]

    if seed is not None:
        torch.manual_seed(seed)

    results: list[dict] = []
    n_batches = (len(examples) + batch_size - 1) // batch_size
    for chunk in pbar(batched(examples, batch_size), desc="generate",
                      total=n_batches, unit="batch"):
        # Fit each prompt into the window minus the generation budget, so that
        # prompt + response also fits the training and probe windows exactly.
        fitted = [fit_prompt(ex, style=style, tokenizer=tokenizer,
                             max_len=max_len - max_new_tokens,
                             use_chat_template=use_chat_template) for ex in chunk]
        prompts = [f[0] for f in fitted]
        trimmed = [f[1] for f in fitted]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(device)
        prompt_len = enc["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            return_dict_in_generate=True,
            output_logits=True,
        )
        # Per-sequence early stop at hallucinated turn boundaries. HF implements
        # this by decoding every sequence at every step, which is CPU-bound and
        # measurably slower than simply generating to max_new_tokens and cutting
        # afterwards. truncate_response() produces the identical final text and
        # the token statistics are recomputed over the kept span either way, so
        # this is purely a speed knob -- see USE_STOP_STRINGS.
        want_stop = USE_STOP_STRINGS if use_stop_strings is None else use_stop_strings
        if want_stop and STOP_STRINGS:
            gen_kwargs["stop_strings"] = list(STOP_STRINGS)
            gen_kwargs["tokenizer"] = tokenizer
        if do_sample:
            if sampler == "standard":
                gen_kwargs.update(temperature=temperature, top_p=top_p)
            elif sampler == "ancestral":
                gen_kwargs.update(temperature=1.0, top_p=1.0)
            else:
                from transformers import LogitsProcessorList
                gen_kwargs.update(temperature=1.0, top_p=1.0,
                                  logits_processor=LogitsProcessorList([en_proc]))

        out = model.generate(**enc, **gen_kwargs)
        seq = out.sequences
        gen_ids = seq[:, prompt_len:]
        mask = _valid_mask(gen_ids, eos_ids)

        n_steps = gen_ids.shape[1]
        B = gen_ids.shape[0]
        tok_lp = torch.zeros(B, n_steps, dtype=torch.float32, device=device)
        tok_ent = torch.zeros(B, n_steps, dtype=torch.float32, device=device)

        # Step-at-a-time so we never materialise [B, T, V] in fp32 at once.
        for t, step_logits in enumerate(out.logits):
            lp = F.log_softmax(step_logits.float(), dim=-1)
            tok_lp[:, t] = lp.gather(-1, gen_ids[:, t:t + 1]).squeeze(-1)
            tok_ent[:, t] = -(lp.exp() * lp).sum(-1)
            del lp

        # Per-row statistics are computed below, after truncation, because the
        # kept span can be shorter than the EOS-delimited span.
        raw_texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for i, ex in enumerate(chunk):
            raw = raw_texts[i]
            text = truncate_response(raw)

            # If truncation removed a tail, the token statistics must be
            # recomputed over the KEPT tokens only -- otherwise mean_token_prob
            # is an average over text we discarded, and every calibration number
            # derived from it is measuring the wrong span.
            row_mask = mask[i]
            span_ok = True
            if text != raw.strip():
                row_mask, span_ok = _kept_token_mask(
                    tokenizer, gen_ids[i], row_mask, raw, text)

            rm = row_mask.float()
            nv = int(row_mask.sum().item())
            # Did the model stop on its own, or did we cut it off? `truncated`
            # below means post-processing changed the text; this is the budget.
            eos_seen = any(int(t) in eos_ids for t in gen_ids[i][mask[i]].tolist())
            hit_budget = (not eos_seen) and int(mask[i].sum().item()) >= max_new_tokens
            # An empty answer has no token statistics. Reporting the preamble's
            # numbers instead would invent confidence the answer never had.
            has_tokens = nv > 0
            denom = max(1.0, float(rm.sum()))
            row_sum_lp = float((tok_lp[i] * rm).sum())
            row_mean_lp = row_sum_lp / denom
            row_mean_ent = float((tok_ent[i] * rm).sum() / denom)
            if nv > 0:
                row_min_lp = float(tok_lp[i][row_mask].min())
            else:
                row_min_lp = row_mean_lp

            results.append({
                "uid": ex["uid"],
                "lang": ex.get("lang", "en"),
                "source": ex.get("source", ""),
                "task_type": ex.get("task_type", "open_qa"),
                "question": ex["question"],
                # PRESERVE the exact generation input. Rebuilding the prompt
                # later from question alone silently dropped `context`, so an
                # answer generated WITH a passage was then trained against the
                # question WITHOUT it. Bangla pools carry context and English
                # ones do not, making that loss language-correlated -- i.e. a
                # candidate cause of the Bangla-only degradation we measured.
                "context": ex.get("context"),
                "prompt": prompts[i],
                "answers": ex.get("answers", []),
                "choices": ex.get("choices"),
                "answer_idx": ex.get("answer_idx"),
                "meta": ex.get("meta", {}),
                "prompt_style": style,
                "response": text,
                "response_raw": raw.strip() if raw.strip() != text else None,
                # post-processing (think-block / turn-boundary / cue / U+FFFD)
                # changed the text. NOT "ran out of tokens" -- that is hit_budget.
                "truncated": raw.strip() != text,
                "hit_budget": bool(hit_budget),
                "sampler": sampler,
                "sampler_T_mean": (en_proc.t_sum / max(1, en_proc.t_n)) if en_proc else None,
                "context_trimmed": bool(trimmed[i]),
                "n_gen_tokens": nv,
                "token_span_ok": bool(span_ok and has_tokens),
                # --- internal confidence signals -------------------------
                "seq_logprob": row_sum_lp if has_tokens else None,
                "mean_logprob": row_mean_lp if has_tokens else None,
                "mean_token_prob": (float(math.exp(min(0.0, row_mean_lp))) if has_tokens else None),
                "min_token_prob": (float(math.exp(min(0.0, row_min_lp))) if has_tokens else None),
                "mean_entropy": row_mean_ent if has_tokens else None,
                "token_logprobs": [round(float(x), 5)
                                   for x in tok_lp[i][row_mask].tolist()],
            })
        del out, seq, gen_ids, tok_lp, tok_ent
    return results


@torch.no_grad()
def self_consistency(model, tokenizer, examples: list[dict], n_samples: int = 5,
                     temperature: float = 1.0, **kw) -> dict[str, dict]:
    """Sample n times per item; agreement rate is a confidence baseline (plan 4c)."""
    from ..eval.grade import normalize_answer

    per_uid: dict[str, list[str]] = {ex["uid"]: [] for ex in examples}
    for s in pbar(range(n_samples), desc="self-consistency", total=n_samples,
                  unit="sample"):
        kw_s = dict(kw)
        kw_s["seed"] = (kw.get("seed") or 0) + 1000 * (s + 1)
        gens = generate_batch(model, tokenizer, examples, do_sample=True,
                              temperature=temperature, **kw_s)
        for g in gens:
            per_uid[g["uid"]].append(g["response"])

    out = {}
    for uid, resps in per_uid.items():
        norm = [normalize_answer(r) for r in resps]
        if not norm:
            out[uid] = {"sc_agreement": 0.0, "sc_n_unique": 0, "sc_mode": ""}
            continue
        counts: dict[str, int] = {}
        for r in norm:
            counts[r] = counts.get(r, 0) + 1
        mode, top = max(counts.items(), key=lambda kv: kv[1])
        out[uid] = {"sc_agreement": top / len(norm),
                    "sc_n_unique": len(counts),
                    "sc_mode": mode}
    return out


ANSWER_CUE = {"en": "Answer:", "bn": "উত্তর:"}


@torch.no_grad()
def score_choices(model, tokenizer, examples: list[dict], style: str = "hedged",
                  batch_size: int = 8, max_len: int = 1024,
                  use_chat_template: bool = True,
                  debias: bool = True) -> list[dict]:
    """MCQ scoring by next-token option-letter probability.

    Two corrections that this measurement does NOT work without, both learned
    the hard way on Qwen3-1.7B / Bangla MMLU:

    1. ANSWER CUE. Scoring the letter distribution at the end of a chat prompt
       asks "how likely is the token 'A' to start the assistant's reply", which
       is dominated by preamble tokens, not by the answer. We append an explicit
       cue so the very next token genuinely is the option letter.

    2. POSITION-BIAS DEBIASING. Even with a cue, models carry a large prior over
       option letters. Uncorrected, Qwen3-1.7B picked "A" for 998/1000 Bangla
       items, and the resulting "accuracy" was just the rate at which the gold
       answer happened to be A. We therefore subtract each letter's MARGINAL log
       probability over the evaluation set (batch calibration) before
       normalising, which removes the constant part of that prior.

    A `degenerate_pick_rate` is reported so this failure can never again hide
    behind a plausible-looking accuracy number.
    """
    device = next(model.parameters()).device
    raw: list[tuple[dict, list[float]]] = []

    n_batches = (len(examples) + batch_size - 1) // batch_size
    for chunk in pbar(batched(examples, batch_size), desc="score options",
                      total=n_batches, unit="batch"):
        prompts = []
        for ex in chunk:
            # Fitted, not just built: an over-long prompt truncated from the
            # right loses the choices and the cue, and the letter distribution
            # is then scored on a passage fragment.
            base, _ = fit_prompt(ex, style=style, tokenizer=tokenizer,
                                 max_len=max_len - 8,
                                 use_chat_template=use_chat_template)
            cue = ANSWER_CUE.get(ex.get("lang", "en"), ANSWER_CUE["en"])
            prompts.append(base if base.rstrip().endswith(cue) else base + " " + cue)

        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        truncation=True, max_length=max_len).to(device)
        logits = model(**enc).logits[:, -1, :].float()
        lp = F.log_softmax(logits, dim=-1)

        for i, ex in enumerate(chunk):
            n_opt = len(ex.get("choices") or [])
            if n_opt == 0:
                continue
            letter_lp = []
            for j in range(n_opt):
                cands = [" " + _LETTERS[j], _LETTERS[j], _LETTERS[j].lower(),
                         " " + _LETTERS[j].lower()]
                best = -float("inf")
                for c in cands:
                    ids = tokenizer.encode(c, add_special_tokens=False)
                    if len(ids) == 1:
                        best = max(best, float(lp[i, ids[0]]))
                letter_lp.append(best)
            if all(x == -float("inf") for x in letter_lp):
                continue
            raw.append((ex, letter_lp))
        del enc, logits, lp

    if not raw:
        return []

    # ---- batch calibration: remove the marginal per-letter prior -------------
    max_opt = max(len(v) for _, v in raw)
    prior = []
    for j in range(max_opt):
        vals = [v[j] for _, v in raw if j < len(v) and v[j] > -float("inf")]
        prior.append(sum(vals) / len(vals) if vals else 0.0)

    results = []
    for ex, letter_lp in raw:
        adj = [lpj - prior[j] if debias else lpj for j, lpj in enumerate(letter_lp)]
        probs = F.softmax(torch.tensor(adj), dim=-1)
        probs_raw = F.softmax(torch.tensor(letter_lp), dim=-1)
        pred = int(probs.argmax())
        gold = ex.get("answer_idx")
        results.append({
            "uid": ex["uid"], "lang": ex.get("lang", "en"),
            "source": ex.get("source", ""), "task_type": "mcq",
            "question": ex["question"], "choices": ex["choices"],
            "answer_idx": gold, "meta": ex.get("meta", {}),
            "pred_idx": pred,
            "pred_idx_raw": int(probs_raw.argmax()),
            "correct": int(pred == gold) if gold is not None else None,
            "p_pred": float(probs[pred]),
            "p_correct": float(probs[gold]) if gold is not None else None,
            "choice_probs": [float(x) for x in probs],
            # Exact log-space scores. The float32 softmax above saturates: on
            # Qwen3-1.7B 283/600 items had p_pred == 1.0 exactly, so ranking
            # (AUROC) over stored probabilities is dominated by one huge tie,
            # and any method that merely softens the model "gains" AUROC by
            # making those items rankable. Rank and temperature-scale on these.
            "choice_logits": [float(x) for x in adj],
            "choice_logits_raw": [float(x) for x in letter_lp],
            "choice_probs_raw": [float(x) for x in probs_raw],
            "mean_entropy": float(-(probs * probs.clamp_min(1e-12).log()).sum()),
        })

    _warn_if_degenerate(results)
    return results


def _warn_if_degenerate(results: list[dict], thresh: float = 0.8) -> float:
    """Shout if the model is just picking one letter regardless of the question."""
    if not results:
        return 0.0
    from collections import Counter
    for key, label in (("pred_idx_raw", "before debiasing"), ("pred_idx", "AFTER debiasing")):
        picks = Counter(r[key] for r in results if r.get(key) is not None)
        if not picks:
            continue
        top, n = picks.most_common(1)[0]
        rate = n / sum(picks.values())
        if rate >= thresh:
            LOG.warning(
                "DEGENERATE MCQ SCORING (%s): option %s chosen for %.1f%% of items. "
                "Accuracy from this signal is just the rate at which the gold answer "
                "is %s -- it is NOT a measure of knowledge. Do not report it.",
                label, _LETTERS[top], 100 * rate, _LETTERS[top])
        for r in results:
            r["degenerate_pick_rate" + ("_raw" if key == "pred_idx_raw" else "")] = rate
    return 0.0
