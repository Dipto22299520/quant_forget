"""Residual-stream extraction for confidence probes (H3).

We read the hidden state at the LAST token of the model's own generated answer,
having re-fed prompt+response in a single forward pass. Reading at the last prompt
token instead would probe what the model knew before committing, which is a
different quantity and not the one H3 is about.
"""
from __future__ import annotations

import numpy as np
import torch

from ..utils import LOG, batched, pbar
from .prompts import build_prompt


@torch.no_grad()
def extract_hidden_states(model, tokenizer, records: list[dict],
                          layer_stride: int = 4, batch_size: int = 8,
                          max_len: int = 1024, use_chat_template: bool = True,
                          ) -> tuple[np.ndarray, list[int]]:
    """-> (features [N, L, H], layer_indices)

    records must carry 'question'/'prompt_style'/'response' (i.e. generation output).
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    layers = list(range(0, n_layers + 1, layer_stride))
    if layers[-1] != n_layers:
        layers.append(n_layers)          # always include the final layer

    # Right padding here: we index the true last token explicitly, and right
    # padding keeps that index arithmetic simple.
    old_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    feats: list[np.ndarray] = []
    try:
        n_batches = (len(records) + batch_size - 1) // batch_size
        for chunk in pbar(batched(records, batch_size), desc="hidden states",
                          total=n_batches, unit="batch"):
            texts = []
            for r in chunk:
                p = build_prompt(r, style=r.get("prompt_style", "hedged"),
                                 tokenizer=tokenizer, use_chat_template=use_chat_template)
                texts.append(p + " " + (r.get("response") or ""))

            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=max_len).to(device)
            out = model(**enc, output_hidden_states=True)
            hs = out.hidden_states           # tuple of [B, T, H], len = n_layers + 1

            attn = enc["attention_mask"]
            last_idx = attn.sum(dim=1) - 1   # last real token per row

            batch_feats = []
            for li in layers:
                h = hs[li]                                     # [B, T, H]
                idx = last_idx.view(-1, 1, 1).expand(-1, 1, h.shape[-1])
                picked = h.gather(1, idx).squeeze(1)            # [B, H]
                batch_feats.append(picked.float().cpu().numpy())
            feats.append(np.stack(batch_feats, axis=1))         # [B, L, H]
            del out, hs, enc
    finally:
        tokenizer.padding_side = old_side

    X = np.concatenate(feats, axis=0) if feats else np.zeros((0, len(layers), 1))
    LOG.info("hidden states: %s (layers %s)", X.shape, layers)
    return X, layers
