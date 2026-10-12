"""Held-out perplexity, reported by register.

Plan 4b wants PPL split by register (news vs literary) because collapse does not
hit registers uniformly, and a single aggregate PPL would average the effect away.
"""
from __future__ import annotations

import math

import torch

from ..utils import LOG, batched


@torch.no_grad()
def held_out_perplexity(model, tokenizer, texts: list[str], max_len: int = 512,
                        batch_size: int = 4, stride: int | None = None) -> dict:
    """Token-level perplexity over a list of raw texts."""
    device = next(model.parameters()).device
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    total_nll, total_tokens = 0.0, 0
    try:
        for chunk in batched([t for t in texts if t and t.strip()], batch_size):
            enc = tokenizer(chunk, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
            ids, attn = enc["input_ids"], enc["attention_mask"]
            if ids.shape[1] < 2:
                continue
            logits = model(input_ids=ids, attention_mask=attn).logits.float()

            shift_logits = logits[:, :-1, :]
            shift_labels = ids[:, 1:]
            shift_mask = attn[:, 1:].bool()

            lp = torch.log_softmax(shift_logits, dim=-1)
            tok_lp = lp.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
            total_nll += float(-(tok_lp[shift_mask]).sum())
            total_tokens += int(shift_mask.sum())
            del enc, logits, lp, tok_lp
    finally:
        tokenizer.padding_side = old_side

    if total_tokens == 0:
        return {"perplexity": float("nan"), "nll": float("nan"), "n_tokens": 0}
    nll = total_nll / total_tokens
    return {"perplexity": float(math.exp(min(nll, 50.0))), "nll": float(nll),
            "n_tokens": total_tokens}


def perplexity_by_register(model, tokenizer, registers: dict[str, list[str]],
                           **kw) -> dict:
    out = {}
    for name, texts in registers.items():
        out[name] = held_out_perplexity(model, tokenizer, texts, **kw)
        LOG.info("ppl[%s] = %.2f (%d tokens)", name,
                 out[name]["perplexity"], out[name]["n_tokens"])
    return out
