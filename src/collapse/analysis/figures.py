"""The paper's figures. Matplotlib only, no seaborn, no style dependencies."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..utils import LOG

# Colour-blind-safe, prints legibly in greyscale.
C = {"accuracy": "#0072B2", "calibration": "#D55E00", "marker": "#CC79A7",
     "probe": "#009E73", "diversity": "#999999", "en": "#0072B2", "bn": "#D55E00"}


def _clean(xs, ys):
    pts = [(x, y) for x, y in zip(xs, ys)
           if y is not None and np.isfinite(y)]
    return ([p[0] for p in pts], [p[1] for p in pts])


def plot_ordering(curves: dict, out_path: str | Path, title: str = "") -> Path:
    """Main figure 1 (H1): normalised trajectories, calibration vs accuracy."""
    gens = list(range(len(curves.get("accuracy", []))))
    series = [
        ("accuracy", "Accuracy", C["accuracy"], "-", "o"),
        ("ece_marker", "ECE (verbalised)", C["marker"], "--", "s"),
        ("ece_internal", "ECE (internal)", C["calibration"], "--", "^"),
        ("hedge_rate", "Hedge rate", C["marker"], ":", "v"),
        ("probe_auroc", "Probe AUROC", C["probe"], "-.", "D"),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for key, label, colour, ls, mk in series:
        if key not in curves:
            continue
        xs, ys = _clean(gens, curves[key])
        if not xs:
            continue
        ax1.plot(xs, ys, label=label, color=colour, linestyle=ls, marker=mk, ms=4)
        base = ys[0] if ys and ys[0] not in (0, None) else None
        if base:
            ax2.plot(xs, [y / base for y in ys], label=label, color=colour,
                     linestyle=ls, marker=mk, ms=4)

    ax1.set_xlabel("generation"); ax1.set_ylabel("value"); ax1.set_title("Raw")
    ax2.set_xlabel("generation"); ax2.set_ylabel("relative to g0")
    ax2.set_title("Normalised to generation 0")
    ax2.axhline(1.0, color="black", lw=0.6, alpha=0.4)
    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.set_xticks(gens)
    ax1.legend(fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_ivd(curves: dict, out_path: str | Path, title: str = "") -> Path:
    """Main figure 2 (H3): internal signal survives, verbalised one decouples."""
    gens = list(range(len(curves.get("ivd_l1", []))))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    for key, label, colour in [("probe_auroc", "Probe AUROC (internal)", C["probe"]),
                               ("auroc_marker", "Marker AUROC (verbalised)", C["marker"]),
                               ("auroc_internal", "Token-prob AUROC", C["calibration"])]:
        if key in curves:
            xs, ys = _clean(gens, curves[key])
            if xs:
                ax1.plot(xs, ys, marker="o", ms=4, label=label, color=colour)
    ax1.axhline(0.5, color="black", lw=0.6, ls=":", alpha=0.6)
    ax1.set_xlabel("generation"); ax1.set_ylabel("AUROC")
    ax1.set_title("Discrimination by signal"); ax1.legend(fontsize=8, frameon=False)

    for key, label, colour in [("ivd_l1", "IVD (L1)", C["calibration"]),
                               ("ivd_auroc_gap", "IVD (AUROC gap)", C["probe"])]:
        if key in curves:
            xs, ys = _clean(gens, curves[key])
            if xs:
                ax2.plot(xs, ys, marker="s", ms=4, label=label, color=colour)
    ax2.set_xlabel("generation"); ax2.set_ylabel("divergence")
    ax2.set_title("Internal-verbal divergence"); ax2.legend(fontsize=8, frameon=False)

    for ax in (ax1, ax2):
        ax.grid(alpha=0.25)
        ax.set_xticks(gens)
    if title:
        fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, out_path)


def plot_filter_decay(per_gen_quality: list[dict], out_path: str | Path,
                      title: str = "") -> Path:
    """H4: the mitigation degrades exactly where it is needed."""
    gens = list(range(len(per_gen_quality)))
    prec = [q.get("precision") for q in per_gen_quality]
    base = [q.get("base_rate") for q in per_gen_quality]
    lift = [q.get("lift") for q in per_gen_quality]

    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs, ys = _clean(gens, prec)
    if xs:
        ax.plot(xs, ys, marker="o", ms=5, color=C["calibration"],
                label="filter precision  P(correct | kept)")
    xs, ys = _clean(gens, base)
    if xs:
        ax.plot(xs, ys, marker="s", ms=5, color=C["diversity"], ls="--",
                label="base rate  P(correct)")
    xs, ys = _clean(gens, lift)
    if xs:
        ax.plot(xs, ys, marker="^", ms=5, color=C["probe"], ls=":",
                label="lift (precision - base rate)")
    ax.axhline(0.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("generation"); ax.set_ylabel("rate")
    ax.set_xticks(gens); ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title(title or "Self-verification filter quality across generations")
    fig.tight_layout()
    return _save(fig, out_path)


def plot_language_comparison(curves_by_lang: dict, metric: str,
                             out_path: str | Path, title: str = "") -> Path:
    """H2: normalised degradation, both languages on one axis."""
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for lang, curves in curves_by_lang.items():
        c = curves.get(metric)
        if not c:
            continue
        gens = list(range(len(c)))
        xs, ys = _clean(gens, c)
        if not xs or not ys[0]:
            continue
        ax.plot(xs, [y / ys[0] for y in ys], marker="o", ms=5,
                color=C.get(lang, None), label=lang)
    ax.axhline(1.0, color="black", lw=0.6, alpha=0.4)
    ax.set_xlabel("generation"); ax.set_ylabel(f"{metric} relative to g0")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    ax.set_title(title or f"{metric}: cross-lingual degradation rate")
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig, out_path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    LOG.info("figure -> %s", out_path)
    return out_path


def plot_all(summary: dict, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    made = []
    for name, curves in summary.get("curves", {}).items():
        made.append(plot_ordering(curves, out_dir / f"fig1_ordering_{name}.png",
                                  title=f"Onset ordering -- {name}"))
        made.append(plot_ivd(curves, out_dir / f"fig2_ivd_{name}.png",
                             title=f"Internal vs verbalised -- {name}"))
    return made
