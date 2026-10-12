r"""How much entropy does the self-training SAMPLER remove, per generation step?

    .venv\Scripts\python.exe scripts\sampler_entropy_probe.py --device cpu

Self-training on exact samples from p has zero expected gradient
(E_{y~p}[grad log p(y)] = 0), so systematic sharpening must come from somewhere
specific. Candidate: the sampler. With top-p = 0.95, any step whose top token
already has p >= 0.95 is sampled greedily, so the training targets are drawn
from a truncated distribution q that is sharper than the model's own p.

For each generation step this reports H(p), H(q), the fraction of steps where
the nucleus is a single token, and T*: the temperature at which the truncated
sampler carries the model's full entropy, H(TopP(softmax(z/T*))) = H(softmax(z))
-- the entropy-neutral sampler.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from collapse.modeling.loader import load_model, load_tokenizer   # noqa: E402
from collapse.modeling.prompts import fit_prompt                   # noqa: E402
from collapse.utils import read_jsonl                              # noqa: E402

TOP_P = 0.95


def trunc_entropy(z_sorted: torch.Tensor, T: torch.Tensor, top_p: float = TOP_P) -> torch.Tensor:
    """Entropy of top-p-truncated softmax(z/T). z_sorted: [B, V] descending."""
    lp = torch.log_softmax(z_sorted / T[:, None], dim=-1)
    p = lp.exp()
    keep = (p.cumsum(-1) - p) < top_p          # smallest top set reaching top_p
    q = p * keep
    Z = q.sum(-1, keepdim=True)
    lq = torch.where(keep, lp - Z.log(), torch.zeros_like(lp))
    return -(q / Z * lq).sum(-1), keep.sum(-1)


def step_stats(z: torch.Tensor):
    z = z.float()
    lp = torch.log_softmax(z, -1)
    Hp = -(lp.exp() * lp).sum(-1)
    zs, _ = z.sort(-1, descending=True)
    ones = torch.ones(z.shape[0])
    Hq, nucleus = trunc_entropy(zs, ones)
    lo, hi = torch.ones(z.shape[0]), torch.full((z.shape[0],), 8.0)
    for _ in range(22):                       # bisection: H(q_T) increases with T
        mid = (lo + hi) / 2
        h, _ = trunc_entropy(zs, mid)
        up = h < Hp
        lo = torch.where(up, mid, lo)
        hi = torch.where(up, hi, mid)
    return Hp, Hq, nucleus, (lo + hi) / 2


def probe(model_path: str, n: int, device: str, seed: int = 1):
    tok = load_tokenizer(model_path)
    model = load_model(model_path, dtype="float32" if device == "cpu" else "bfloat16", device=device)
    model.eval()
    pool = read_jsonl(ROOT / "runs" / "main_en_r100_s1" / "pools" / "prompt_pool.jsonl")[:n]
    eos = {tok.eos_token_id, tok.pad_token_id} - {None}
    torch.manual_seed(seed)
    Hp_all, Hq_all, nuc_all, T_all = [], [], [], []
    for i in range(0, n, 16):
        chunk = pool[i:i + 16]
        prompts = [fit_prompt(ex, style="hedged", tokenizer=tok, max_len=736)[0] for ex in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=32, do_sample=True, temperature=1.0,
                                 top_p=TOP_P, pad_token_id=tok.pad_token_id,
                                 return_dict_in_generate=True, output_logits=True)
        gen = out.sequences[:, enc["input_ids"].shape[1]:]
        alive = torch.ones(gen.shape[0], dtype=torch.bool)
        for t, z in enumerate(out.logits):
            Hp, Hq, nuc, Ts = step_stats(z.cpu())
            Hp_all += Hp[alive].tolist(); Hq_all += Hq[alive].tolist()
            nuc_all += nuc[alive].tolist(); T_all += Ts[alive].tolist()
            alive &= ~torch.tensor([int(x) in eos for x in gen[:, t].tolist()])
        print(f"   {min(i + 16, n)}/{n} prompts", flush=True)
    del model
    Hp, Hq, nuc, T = map(np.array, (Hp_all, Hq_all, nuc_all, T_all))
    return {"steps": len(Hp), "H_model": Hp.mean(), "H_sampler": Hq.mean(),
            "ratio": Hq.mean() / Hp.mean(), "greedy_frac": float((nuc == 1).mean()),
            "T_star_median": float(np.median(T)), "T_star_mean": float(T.mean()),
            "T_star_p90": float(np.percentile(T, 90))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=64)
    args = ap.parse_args()
    results = {}
    for label, path in (("base Qwen3-1.7B (g0)", "Qwen/Qwen3-1.7B"),
                        ("after 5 rounds, 100% synthetic (g5)", str(ROOT / "runs" / "main_en_r100_s1" / "ckpt" / "model_5"))):
        print(f"{label}:", flush=True)
        r = probe(path, args.n, args.device)
        print(f"   steps {r['steps']} | H(model) {r['H_model']:.3f} nats | H(top-p sampler) {r['H_sampler']:.3f} "
              f"| sampler keeps {r['ratio']:.0%} of the model's entropy | greedy steps {r['greedy_frac']:.0%} "
              f"| entropy-neutral T*: median {r['T_star_median']:.2f}, mean {r['T_star_mean']:.2f}, p90 {r['T_star_p90']:.2f}",
              flush=True)
        results["g0" if "g0" in label else "g5"] = {k: float(v) for k, v in r.items()}
    import json
    (ROOT / "paper" / "sampler_probe.json").write_text(json.dumps(results, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
