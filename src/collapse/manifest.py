"""Immutable run manifest and resume compatibility checking (audit finding #5).

The previous identity check hashed package `.py` files only. That missed YAML
configs, marker lexicons and scripts, so the same run name with a new ratio,
seed or lexicon silently reused old results. Worse, `cmd_run` overwrote the
saved config BEFORE checking compatibility, destroying the evidence needed to
detect the mismatch.

A manifest records everything that defines the experiment:

    config      the fully resolved config (post-`_base_`, post-CLI override)
    code        hash of the package source
    lexicons    hash of each marker lexicon (they change marker_conf directly)
    configs_dir hash of the YAML tree
    model       base model id
    data        dataset keys and sizes
    env         python / torch / transformers versions

Resume is allowed only when the identity-critical parts match. Anything else is
a different experiment that happens to share a directory name, and it must not
inherit the old generations.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict
from pathlib import Path

from .utils import LOG, code_fingerprint, read_json, write_json

# Changing any of these makes the stored generations invalid. `seed` is included:
# seeds are the replication unit, so a seed change is a different run, not a
# resumption of this one.
IDENTITY_KEYS = (
    "code", "config_core", "lexicons", "model", "data_keys", "seed",
)


def _hash_files(paths) -> str:
    h = hashlib.md5()
    for p in sorted(paths):
        p = Path(p)
        if not p.exists():
            continue
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _core_config(cfg) -> dict:
    """The parts of the config that define the EXPERIMENT, not just its logging.

    out_dir and run_name are excluded (they name the artefact, not the science).
    Everything that changes what the model sees or how it is trained is included.
    """
    d = asdict(cfg)
    d.pop("out_dir", None)
    d.pop("run_name", None)
    return d


def build_manifest(cfg, extra: dict | None = None) -> dict:
    root = Path(__file__).parent
    lex = sorted((root / "markers").glob("lexicon_*.json"))
    cfg_dir = Path("configs")

    core = _core_config(cfg)
    man = {
        "code": code_fingerprint(),
        "config_core": hashlib.md5(
            json.dumps(core, sort_keys=True, default=str).encode()).hexdigest()[:12],
        "config_resolved": core,
        "lexicons": _hash_files(lex),
        "lexicon_files": [p.name for p in lex],
        "configs_dir": _hash_files(cfg_dir.glob("*.yaml")) if cfg_dir.exists() else None,
        "model": cfg.model.name,
        "data_keys": {"seed_pool": list(cfg.data.seed_pool),
                      "eval_sets": list(cfg.data.eval_sets)},
        "seed": cfg.seed,
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    for mod in ("torch", "transformers", "peft", "datasets"):
        try:
            man["env"][mod] = __import__(mod).__version__
        except Exception:
            man["env"][mod] = None
    if extra:
        man.update(extra)
    return man


class IncompatibleResume(RuntimeError):
    pass


def check_resume(run_dir, cfg, force: bool = False) -> dict:
    """Compare the new manifest with the stored one BEFORE anything is written.

    Returns the new manifest. Raises IncompatibleResume when the stored run was
    a different experiment, unless force=True.
    """
    run_dir = Path(run_dir)
    path = run_dir / "manifest.json"
    new = build_manifest(cfg)

    if not path.exists():
        return new

    old = read_json(path)
    diffs = [k for k in IDENTITY_KEYS if old.get(k) != new.get(k)]
    if not diffs:
        LOG.info("resume OK: manifest matches (%s)", new["config_core"])
        return new

    detail = []
    for k in diffs:
        o, n = old.get(k), new.get(k)
        if k == "config_core":
            o, n = _describe_config_diff(old.get("config_resolved"),
                                         new.get("config_resolved"))
            detail.append(f"    config: {o}  ->  {n}")
        else:
            detail.append(f"    {k}: {o!r}  ->  {n!r}")
    msg = (f"run {run_dir.name!r} already exists but describes a DIFFERENT "
           f"experiment; resuming would mix incompatible generations:\n"
           + "\n".join(detail) +
           f"\n  Use a new --run-name, or delete {run_dir}, or pass --force "
           f"to overwrite deliberately.")
    if force:
        LOG.warning("FORCED resume over an incompatible manifest:\n%s", msg)
        return new
    raise IncompatibleResume(msg)


def _describe_config_diff(old: dict | None, new: dict | None):
    """Name the fields that actually changed, not just the hash."""
    if not isinstance(old, dict) or not isinstance(new, dict):
        return "unknown", "unknown"

    def flat(d, prefix=""):
        out = {}
        for k, v in (d or {}).items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(flat(v, key + "."))
            else:
                out[key] = v
        return out

    fo, fn = flat(old), flat(new)
    changed = [k for k in set(fo) | set(fn) if fo.get(k) != fn.get(k)]
    if not changed:
        return "(no field diff)", "(no field diff)"
    shown = sorted(changed)[:6]
    return (", ".join(f"{k}={fo.get(k)!r}" for k in shown),
            ", ".join(f"{k}={fn.get(k)!r}" for k in shown))


def save_manifest(run_dir, manifest: dict) -> None:
    write_json(Path(run_dir) / "manifest.json", manifest)


def checkpoint_usable(ckpt_dir) -> tuple[bool, str]:
    """Weights actually present? Pruning leaves the metadata folder behind, so
    directory existence alone is not evidence a checkpoint can be loaded."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return False, "missing"
    marker = ckpt_dir / "_pruned.json"
    if marker.exists():
        return False, "weights pruned (keep_checkpoints was false)"
    weights = (list(ckpt_dir.glob("*.safetensors")) + list(ckpt_dir.glob("*.bin"))
               + list(ckpt_dir.glob("adapter_model*")))
    if not weights:
        return False, "no weight files present"
    return True, "ok"
