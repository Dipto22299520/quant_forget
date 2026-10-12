"""Model / tokenizer loading, sized for a 16GB card.

transformers 5.x notes (this machine runs 5.15):
  * `torch_dtype=` was REMOVED -> use `dtype=`
  * flash-attention has no sm_120 Windows wheel -> attn_implementation="sdpa"
"""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..utils import LOG, estimate_full_ft_vram

_DTYPES = {
    "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
    "float16": torch.float16, "fp16": torch.float16,
    "float32": torch.float32, "fp32": torch.float32,
}


def load_tokenizer(name: str, trust_remote_code: bool = False):
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code)
    if tok.pad_token is None:
        # Never introduce a new token id: resizing embeddings mid-loop would make
        # generation g and g+1 architecturally different models.
        tok.pad_token = tok.eos_token
    # Decoder-only batched generation requires left padding.
    tok.padding_side = "left"
    # Safety net only -- prompts are fitted to the window by fit_prompt() before
    # they reach the tokenizer. If one still overflows, losing the START of the
    # system prompt is recoverable; losing the END (the question and the
    # assistant marker, which is what the default "right" does) is not: the
    # model then continues the passage instead of answering.
    tok.truncation_side = "left"
    return tok


def load_model(name: str, dtype: str = "bfloat16", attn_impl: str = "sdpa",
               trust_remote_code: bool = False, device: str = "cuda",
               adapter_path: str | Path | None = None, for_training: bool = False):
    torch_dtype = _DTYPES.get(dtype, torch.bfloat16)
    if device == "cuda" and not torch.cuda.is_available():
        LOG.warning("cuda requested but unavailable -- falling back to cpu (slow)")
        device = "cpu"
        torch_dtype = torch.float32

    kwargs = dict(dtype=torch_dtype, attn_implementation=attn_impl,
                  trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.to(device)

    if adapter_path is not None:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))
        if not for_training:
            # Merging removes the adapter indirection so generation at g+1 runs at
            # base-model speed and the loop cannot accidentally stack adapters.
            model = model.merge_and_unload()
        LOG.info("loaded adapter %s (merged=%s)", adapter_path, not for_training)

    if not for_training:
        model.eval()
        model.config.use_cache = True
    return model


def model_param_count(model) -> float:
    """Billions of parameters."""
    return sum(p.numel() for p in model.parameters()) / 1e9


def vram_precheck(n_params_b: float, mode: str, optim: str) -> dict:
    """Fail fast instead of OOMing 40 minutes into a run."""
    # A CUDA build with the GPU hidden (CUDA_VISIBLE_DEVICES="") still reports
    # is_available() but has no device 0; that is a CPU run too.
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return {"ok": True, "reason": "cpu"}
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    if mode == "full":
        need = estimate_full_ft_vram(n_params_b, optim)
        need_act = need + 2.0          # activations w/ grad checkpointing, bs=1
    else:
        # LoRA: frozen bf16 base + tiny adapter grads/state
        need = 2 * n_params_b
        need_act = need + 1.5
    ok = need_act < total * 0.92
    info = {"ok": ok, "total_gb": round(total, 1), "need_gb": round(need_act, 1),
            "mode": mode, "optim": optim, "params_b": round(n_params_b, 2)}
    if not ok:
        LOG.error(
            "VRAM PRECHECK FAILED: %s FT of %.2fB needs ~%.1f GB but card has %.1f GB.\n"
            "  Options: mode=lora | optim=paged_adamw_8bit | a smaller base model.",
            mode, n_params_b, need_act, total)
    else:
        LOG.info("vram precheck ok: %s FT of %.2fB ~%.1f GB / %.1f GB",
                 mode, n_params_b, need_act, total)
    return info
