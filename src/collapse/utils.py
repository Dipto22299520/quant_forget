"""Seeding, IO, logging, and 16GB-VRAM bookkeeping."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "collapse") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt="%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


LOG = get_logger()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------- JSONL IO

def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"missing jsonl: {path}")
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default),
                    encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def batched(seq: list, n: int) -> Iterator[list]:
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ---------------------------------------------------------------- VRAM

def cuda_mem_summary() -> str:
    try:
        import torch

        if not torch.cuda.is_available():
            return "cuda unavailable"
        alloc = torch.cuda.memory_allocated() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"VRAM {alloc:.2f}/{total:.1f} GB (peak {peak:.2f})"
    except Exception as e:  # pragma: no cover
        return f"VRAM query failed: {e}"


def free_vram() -> None:
    try:
        import gc

        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


def estimate_full_ft_vram(n_params_b: float, optim: str = "paged_adamw_8bit") -> float:
    """Rough GB needed for full fine-tuning, excluding activations.

    bf16 weights + bf16 grads + optimiser state. This is the check that says
    whether a model fits in this machine's 16GB before we waste an hour finding out.
    """
    p = n_params_b * 1e9
    weights = 2 * p
    grads = 2 * p
    if "8bit" in optim:
        opt = 2 * p          # 2 x uint8 moments
    elif "adafactor" in optim:
        opt = 0.5 * p
    else:
        opt = 12 * p         # fp32 m, v, + fp32 master copy
    return (weights + grads + opt) / 1e9


# ---------------------------------------------------------------- provenance

def code_fingerprint() -> str:
    """Short hash of the package source.

    The loop is resumable: a generation with a done-marker is skipped. That is
    right when a run was interrupted, and WRONG when the code changed underneath
    it -- stale results then survive silently and get re-analysed as if fresh.
    (This cost a full pilot: a reasoning-mode fix was made, the pilot re-run
    finished in 0.0 h, and every number was the pre-fix cache.)

    Every done-marker records this, and a mismatch forces recomputation.
    """
    import hashlib

    root = Path(__file__).parent
    h = hashlib.md5()
    for f in sorted(root.rglob("*.py")):
        h.update(f.relative_to(root).as_posix().encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:12]


def stale(marker: dict, name: str = "") -> bool:
    """True if a cached result was produced by different code."""
    old = (marker or {}).get("code_fingerprint")
    cur = code_fingerprint()
    if old is None:
        LOG.warning("%s has no code fingerprint (produced before provenance "
                    "tracking); treating as STALE and recomputing", name)
        return True
    if old != cur:
        LOG.warning("%s was produced by code %s but current code is %s -- "
                    "recomputing rather than reusing stale results", name, old, cur)
        return True
    return False


# ---------------------------------------------------------------- progress

_PROGRESS_ENABLED = os.environ.get("COLLAPSE_NO_PROGRESS", "") == ""


def pbar(iterable=None, desc: str = "", total: int | None = None,
         unit: str = "it", leave: bool = False, **kw):
    """tqdm wrapper used by every long-running loop.

    Writes to stderr so it never interleaves with the stdout log lines. Inner
    bars use leave=False so a finished stage collapses instead of leaving a wall
    of stale bars behind. Set COLLAPSE_NO_PROGRESS=1 to disable (useful when
    piping to a file).
    """
    if not _PROGRESS_ENABLED:
        return iterable if iterable is not None else _NullBar()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return iterable if iterable is not None else _NullBar()
    return tqdm(iterable, desc=desc, total=total, unit=unit, leave=leave,
                dynamic_ncols=True, file=sys.stderr,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
                **kw)


class _NullBar:
    def update(self, *a, **k): pass
    def set_postfix(self, *a, **k): pass
    def set_description(self, *a, **k): pass
    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *a): return False
