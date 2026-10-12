"""SFT training loop.

Hand-rolled rather than TRL's SFTTrainer for two reasons:
  1. CAL and TCE need PER-TOKEN loss weights derived from the model's own
     confidence in the same forward pass. Threading that through SFTTrainer is
     more fragile than writing the 60 lines.
  2. TRL's API has churned hard across 0.x -> 1.x; this loop only depends on
     stable torch/peft surface.

Loss variants (these ARE the H4 mitigation baselines):
  ce   -- standard next-token cross entropy
  cal  -- ForTIFAI/CAL: down-weight tokens the model is already confident in,
          w_t = (1 - p_t)^gamma. Breaks the overconfident-generation feedback loop.
  tce  -- Truncated Cross Entropy: zero the loss on tokens with p_t > threshold.
"""
from __future__ import annotations

import math
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..utils import LOG, cuda_mem_summary, free_vram, pbar, set_seed
from .loader import load_model, load_tokenizer, model_param_count, vram_precheck
from .prompts import build_sft_text

IGNORE = -100


class SFTDataset(Dataset):
    """(prompt, completion) pairs; loss is masked to completion tokens only."""

    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_len: int = 1024):
        self.pairs = pairs
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        prompt, completion = self.pairs[i]
        p_ids = self.tok.encode(prompt, add_special_tokens=False)
        c_ids = self.tok.encode(completion, add_special_tokens=False)
        # Truncate the PROMPT from the left, never the completion: losing the
        # answer would silently turn a training example into noise.
        budget = self.max_len - len(c_ids)
        if budget < 1:
            c_ids = c_ids[: self.max_len - 1]
            budget = 1
        p_ids = p_ids[-budget:]
        ids = p_ids + c_ids
        labels = [IGNORE] * len(p_ids) + list(c_ids)
        return {"input_ids": ids, "labels": labels}


def collate(batch, pad_id: int):
    n = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        pad = n - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * pad)
        labels.append(b["labels"] + [IGNORE] * pad)
        attn.append([1] * len(b["input_ids"]) + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def weighted_lm_loss(logits: torch.Tensor, labels: torch.Tensor, loss_type: str = "ce",
                     cal_gamma: float = 1.0, tce_threshold: float = 0.95,
                     ls_eps: float = 0.1, ent_beta: float = 0.1,
                     ) -> tuple[torch.Tensor, dict]:
    """Next-token loss: plain, confidence-reweighted, or entropy-regularised.

    ce    cross entropy on the target token
    cal   ForTIFAI/CAL: weight (1 - p_t)^gamma, detached
    tce   truncated CE (Shabgahi et al. 2025): drop tokens with p_t >= threshold
    ls    label smoothing: (1 - eps) * CE + eps * CE-to-uniform
    ent   confidence penalty (Pereyra et al. 2017): CE - beta * H(p)

    ls and ent regularise the whole predictive distribution, so unlike cal/tce
    their extra term carries gradient.
    """
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != IGNORE

    safe_labels = shift_labels.clone()
    safe_labels[~mask] = 0

    logprobs = F.log_softmax(shift_logits, dim=-1)
    tok_lp = logprobs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    ce = -tok_lp

    tok_loss = ce
    ent_mean = None
    if loss_type in ("ce", "ls", "ent"):
        w = torch.ones_like(ce)
        if loss_type == "ls":
            # CE against the uniform distribution is the negative mean log-prob.
            tok_loss = (1.0 - ls_eps) * ce + ls_eps * (-logprobs.mean(-1))
        elif loss_type == "ent":
            ent = -(logprobs.exp() * logprobs).sum(-1)
            tok_loss = ce - ent_beta * ent
            ent_mean = ent.detach()
    else:
        # Detached: the reweighting is a curriculum, not a gradient path.
        p = tok_lp.detach().exp()
        if loss_type == "cal":
            w = (1.0 - p).clamp(min=0.0) ** cal_gamma
        elif loss_type == "tce":
            w = (p < tce_threshold).float()
        else:
            raise ValueError(f"unknown loss {loss_type!r}, expected ce|cal|tce|ls|ent")

    w = w * mask.float()
    denom = w.sum().clamp(min=1e-6)
    loss = (tok_loss * w).sum() / denom

    with torch.no_grad():
        n_tok = mask.sum().clamp(min=1)
        raw_ce = (ce * mask.float()).sum() / n_tok
        stats = {
            "raw_ce": float(raw_ce),
            "mean_weight": float(w.sum() / n_tok),
            "frac_tokens_kept": float((w > 0).float().sum() / n_tok),
        }
        if ent_mean is not None:
            stats["pred_entropy"] = float((ent_mean * mask.float()).sum() / n_tok)
    return loss, stats


def build_pairs(examples: list[dict], responses: dict[str, str] | None,
                tokenizer, style: str = "hedged",
                max_len: int | None = None) -> list[tuple[str, str]]:
    """examples -> (prompt, completion). If responses is None, use gold answers.

    max_len: window the PROMPT is fitted into (passage trimmed, see fit_prompt).
    Synthetic examples carry their generation prompt and are left untouched.
    """
    from .prompts import gold_response

    pairs = []
    for ex in examples:
        if responses is not None:
            resp = responses.get(ex["uid"])
            if not resp:
                continue
        else:
            resp = gold_response(ex)
        if not resp or not resp.strip():
            continue
        pairs.append(build_sft_text(ex, resp, style=style, tokenizer=tokenizer,
                                    max_len=max_len))
    return pairs


def train_sft(cfg, pairs: list[tuple[str, str]], out_dir: str | Path,
              base_model: str | None = None, init_adapter: str | Path | None = None,
              merge_before_save: bool = False) -> dict:
    """Fine-tune and save to out_dir. Returns training stats.

    merge_before_save: with loop.chain="weights", the LoRA update must be folded
    into the weights and saved as a standalone model, so the NEXT generation
    trains on top of it. Without this, every generation would restart from the
    pristine base and the recursion would only chain through the data.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tcfg = cfg.train
    set_seed(tcfg.seed)

    if not pairs:
        raise ValueError("no training pairs -- check filtering and the SFT mix")

    model_name = base_model or cfg.model.name
    tokenizer = load_tokenizer(model_name, cfg.model.trust_remote_code)
    # for_training=True keeps any init adapter UNMERGED so it can keep training.
    model = load_model(model_name, dtype=cfg.model.dtype, attn_impl=cfg.model.attn_impl,
                       trust_remote_code=cfg.model.trust_remote_code, device=cfg.device,
                       adapter_path=init_adapter, for_training=True)

    n_params = model_param_count(model)
    check = vram_precheck(n_params, tcfg.mode, tcfg.optim)
    if not check["ok"]:
        raise RuntimeError(
            f"refusing to start: {check}. Change train.mode or train.optim.")

    if tcfg.mode == "lora" and init_adapter is None:
        from peft import LoraConfig, get_peft_model
        lcfg = LoraConfig(r=tcfg.lora_r, lora_alpha=tcfg.lora_alpha,
                          lora_dropout=tcfg.lora_dropout, bias="none",
                          task_type="CAUSAL_LM", target_modules=list(tcfg.lora_targets))
        model = get_peft_model(model, lcfg)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOG.info("LoRA: %.2fM trainable / %.2fB total", trainable / 1e6, n_params)

    if tcfg.grad_ckpt:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.train()

    ds = SFTDataset(pairs, tokenizer, max_len=tcfg.max_len)
    dl = DataLoader(ds, batch_size=tcfg.per_device_bs, shuffle=True,
                    collate_fn=lambda b: collate(b, tokenizer.pad_token_id),
                    num_workers=0, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = _make_optimizer(params, tcfg)

    steps_per_epoch = math.ceil(len(dl) / tcfg.grad_accum)
    total_steps = max(1, steps_per_epoch * tcfg.epochs)
    warmup = max(1, int(total_steps * tcfg.warmup_ratio))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: (s + 1) / warmup if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    LOG.info("train: n=%d pairs, %d steps, mode=%s loss=%s lr=%g | %s",
             len(pairs), total_steps, tcfg.mode, tcfg.loss, tcfg.lr, cuda_mem_summary())

    device = next(model.parameters()).device
    hist, step = [], 0
    running, n_running = 0.0, 0
    for epoch in range(tcfg.epochs):
        optimizer.zero_grad(set_to_none=True)
        bar = pbar(total=len(dl), desc=f"train e{epoch + 1}/{tcfg.epochs}",
                   unit="mb")
        for i, batch in enumerate(dl):
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"])
            loss, stats = weighted_lm_loss(
                out.logits, batch["labels"], loss_type=tcfg.loss,
                cal_gamma=tcfg.cal_gamma, tce_threshold=tcfg.tce_threshold,
                ls_eps=tcfg.ls_eps, ent_beta=tcfg.ent_beta)
            (loss / tcfg.grad_accum).backward()
            running += float(loss.detach())
            n_running += 1
            bar.update(1)
            bar.set_postfix(loss=f"{float(loss.detach()):.3f}",
                            ce=f"{stats['raw_ce']:.3f}", step=step)

            if (i + 1) % tcfg.grad_accum == 0 or (i + 1) == len(dl):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                sched.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1 or step == total_steps:
                    # Mean over the micro-batches actually seen since the last log,
                    # not a fixed divisor -- the last accumulation window is short.
                    avg = running / max(1, n_running)
                    LOG.info("  step %d/%d loss %.4f raw_ce %.4f kept %.2f | %s",
                             step, total_steps, avg, stats["raw_ce"],
                             stats["frac_tokens_kept"], cuda_mem_summary())
                    hist.append({"step": step, "loss": avg, **stats})
                    running, n_running = 0.0, 0
            del out, loss
        bar.close()

    # ---- save -------------------------------------------------------------
    model.config.use_cache = True
    if tcfg.mode == "lora" and merge_before_save:
        if tcfg.grad_ckpt:
            model.gradient_checkpointing_disable()
        model = model.merge_and_unload()
        model.save_pretrained(str(out_dir), safe_serialization=True)
        saved_as = "merged_full_model"
    elif tcfg.mode == "lora":
        model.save_pretrained(str(out_dir))        # adapter only
        saved_as = "adapter"
    else:
        model.save_pretrained(str(out_dir), safe_serialization=True)
        saved_as = "full_model"
    tokenizer.save_pretrained(str(out_dir))

    stats = {"n_pairs": len(pairs), "steps": step, "history": hist,
             "mode": tcfg.mode, "loss": tcfg.loss, "saved_as": saved_as,
             "vram": cuda_mem_summary()}
    del model, optimizer
    free_vram()
    return stats


def _make_optimizer(params, tcfg):
    if "8bit" in tcfg.optim:
        try:
            import bitsandbytes as bnb
            cls = (bnb.optim.PagedAdamW8bit if "paged" in tcfg.optim
                   else bnb.optim.AdamW8bit)
            return cls(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        except Exception as e:
            LOG.warning("bitsandbytes 8-bit optimiser unavailable (%s); "
                        "falling back to torch AdamW -- VRAM use will be higher", e)
    return torch.optim.AdamW(params, lr=tcfg.lr, weight_decay=tcfg.weight_decay)


def prune_checkpoint(path: str | Path) -> None:
    """Delete weights but keep the config/metrics trail (plan 8: metrics, not weights)."""
    path = Path(path)
    if not path.exists():
        return
    removed = []
    for f in path.iterdir():
        if f.suffix in {".safetensors", ".bin", ".pt"}:
            removed.append(f.name)
            f.unlink(missing_ok=True)
    # Record the pruning explicitly: an empty-but-present directory otherwise
    # looks like a usable checkpoint to any resume logic.
    import json as _json
    (path / "_pruned.json").write_text(
        _json.dumps({"pruned": True, "removed": removed}, indent=2), encoding="utf-8")
    LOG.info("pruned %d weight files from %s", len(removed), path)
