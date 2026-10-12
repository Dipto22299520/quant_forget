"""Config tree. YAML in, dataclasses out.

Deliberately flat and boring: every experiment is fully described by one YAML file
that gets copied into the run directory, so a run is reproducible from its own
output folder.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints

import yaml


def _merge(base: dict, over: dict) -> dict:
    """Recursive dict merge; `over` wins."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _build(cls, d: dict):
    """Construct a dataclass from a dict, rejecting unknown keys loudly.

    `from __future__ import annotations` makes every field .type a STRING, so
    nested dataclasses must be resolved through get_type_hints -- otherwise
    cfg.model silently stays a plain dict and every attribute access fails later.
    """
    known = {f.name for f in fields(cls)}
    unknown = set(d) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown config keys {sorted(unknown)}")

    hints = get_type_hints(cls)
    kwargs = {}
    for f in fields(cls):
        if f.name not in d:
            continue
        v = d[f.name]
        ftype = hints.get(f.name, f.type)
        if is_dataclass(ftype) and isinstance(v, dict):
            kwargs[f.name] = _build(ftype, v)
        else:
            kwargs[f.name] = v
    return cls(**kwargs)


@dataclass
class ModelCfg:
    name: str = "Qwen/Qwen3-1.7B"
    dtype: str = "bfloat16"
    attn_impl: str = "sdpa"          # no flash-attn wheel for sm_120 on Windows
    trust_remote_code: bool = False
    max_len: int = 1024


@dataclass
class GenCfg:
    max_new_tokens: int = 48
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    batch_size: int = 16
    # Self-consistency sampling is one of the confidence baselines (plan 4c).
    sc_samples: int = 5
    sc_temperature: float = 1.0
    # "hedged" instructs natural hedging; "bare" forbids it; "verbalized" asks for
    # an explicit numeric confidence. We need all three across the paper.
    prompt_style: str = "hedged"
    # Sampler for the SELF-TRAINING data only (evaluation generation is unchanged):
    #   standard         temperature + top_p as configured
    #   ancestral        top_p = 1.0: exact samples from the model (diagnostic)
    #   entropy_neutral  per-step temperature T* such that the top-p-truncated
    #                    sampler carries the model's full entropy:
    #                    H(TopP(softmax(z/T*))) = H(softmax(z)).
    # Measured on Qwen3-1.7B: top_p=0.95 keeps 81% of the model's entropy at g0
    # and 64% at g5 (73% -> 90% of steps greedy) -- a sharpening feedback loop.
    sampler: str = "standard"
    seed: int = 0


@dataclass
class TrainCfg:
    mode: str = "lora"               # lora | full
    lr: float = 1e-4                 # full-FT should override to ~2e-5
    epochs: int = 1
    per_device_bs: int = 1
    grad_accum: int = 16
    max_len: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    optim: str = "paged_adamw_8bit"  # 16GB VRAM: 8-bit states are not optional for full-FT
    grad_ckpt: bool = True
    # Loss variant -- this is how we implement the H4 mitigation baselines.
    loss: str = "ce"                 # ce | cal | tce | ls | ent
    cal_gamma: float = 1.0           # ForTIFAI/CAL down-weight strength
    tce_threshold: float = 0.95      # Truncated Cross Entropy cutoff
    ls_eps: float = 0.1              # label smoothing mass (loss=ls)
    ent_beta: float = 0.1            # entropy-bonus weight, Pereyra et al. 2017 (loss=ent)
    seed: int = 0


@dataclass
class DataCfg:
    lang: str = "en"                 # en | bn
    seed_pool: list = field(default_factory=list)   # registry keys, real data
    eval_sets: list = field(default_factory=list)   # registry keys, never trained on
    prompt_pool_size: int = 3000     # questions regenerated each generation
    real_pool_size: int = 30000      # real SFT examples available for mixing
    eval_size: int = 1000            # per eval set
    probe_size: int = 2000           # items used to fit confidence probes
    cache_dir: str = "data/cache"


@dataclass
class LoopCfg:
    generations: int = 5             # g = 0 .. generations-1
    ratio: float = 1.0               # synthetic fraction of the SFT mix (1.0 = 100:0)
    sft_size: int = 30000            # total SFT examples per generation
    filter: str = "none"             # none | selfverify | selfverify_marker | sc_agreement | cal
    filter_threshold: float = 0.5    # confidence threshold for selfverify
    keep_checkpoints: bool = False   # storage: we keep metrics, not weights

    # HOW THE RECURSION CHAINS. This is a scientific choice, not a detail --
    # it changes what "generation" means and how fast collapse can appear.
    #
    #   weights   : model_{g+1} = SFT(model_g, data from model_g).
    #               Degradation accumulates in BOTH the weights and the data.
    #               Matches iterative self-improvement setups (KITE). Stronger
    #               signal within 5 generations, which matters given the risk
    #               register's "no visible degradation" concern.
    #
    #   data_only : model_{g+1} = SFT(base, data from model_g).
    #               Every generation restarts from the pristine base model, so
    #               collapse propagates purely through the DATA. This is closer
    #               to Shumailov et al.'s original recursion.
    #
    # Report which one you ran. A reviewer will ask, and the two answer
    # different questions.
    chain: str = "weights"

    # Fail closed when the synthetic pool cannot honour `ratio` at `sft_size`.
    # False restores top-up behaviour and marks the run as an uncontrolled
    # mixture rather than a member of the ratio grid.
    strict_ratio: bool = True

    # Draw the reference share of each round's mix from a DISJOINT slice of the
    # reference pool, so no reference item is trained on twice across rounds.
    # Separates "a small anchor helps" from "a finite pool is memorised".
    real_fresh: bool = False

    # Samples per prompt for the self-training data, at a FIXED pair budget:
    # K > 1 uses the first len(prompt_pool) // K prompts, K samples each.
    # One sample per prompt is maximum likelihood on n = 1 per context, whose
    # plug-in entropy is biased low; the bias shrinks roughly as 1/K.
    samples_per_prompt: int = 1


@dataclass
class ProbeCfg:
    layer_stride: int = 4            # residual stream every 4th layer
    kind: str = "logistic"           # logistic | linear
    C: float = 1.0
    n_splits: int = 5
    max_items: int = 2000


@dataclass
class Config:
    run_name: str = "unnamed"
    out_dir: str = "runs"
    seed: int = 0
    device: str = "cuda"
    model: ModelCfg = field(default_factory=ModelCfg)
    gen: GenCfg = field(default_factory=GenCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    data: DataCfg = field(default_factory=DataCfg)
    loop: LoopCfg = field(default_factory=LoopCfg)
    probe: ProbeCfg = field(default_factory=ProbeCfg)

    @property
    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_name

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=False, allow_unicode=True),
                        encoding="utf-8")


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> Config:
    """Load YAML, follow an optional `_base_` chain, apply dotted CLI overrides."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    if "_base_" in raw:
        base_path = (path.parent / raw.pop("_base_")).resolve()
        base_raw = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
        base_raw.pop("_base_", None)
        raw = _merge(base_raw, raw)

    if overrides:
        for dotted, val in overrides.items():
            node = raw
            parts = dotted.split(".")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = val

    return _build(Config, raw)


def parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """`--set train.lr=2e-5 loop.ratio=0.5` -> {'train.lr': 2e-5, 'loop.ratio': 0.5}"""
    out: dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            raise ValueError(f"override must be key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k] = yaml.safe_load(v)   # gives us int/float/bool/list parsing for free
    return out
