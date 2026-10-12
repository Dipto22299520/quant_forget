r"""Mechanism table and figures for the recursive self-training study (English, Qwen3-1.7B).

    .venv\Scripts\python.exe scripts\mechanism_assets.py

Every 1.7B English Qwen3 arm at a valid fingerprint is summarised per condition
(mean +- sd over seeds):
  kept/round      geometric-mean ratio of the model's own entropy on its
                  self-generated training targets, round to round
  calib-start     over(+)/under(-)confidence at g5 under a temperature fitted
                  ONCE on labelled g0 items (2-fold cross-fitted). This is drift
                  measured from a calibrated starting point, free of the raw-ECE
                  ceiling of the overconfident base model.
  fixed/tracked/oracle   ECE at g5 with that stale T0, with a label-free
                  entropy-matched temperature, and with a per-round labelled T.
Writes paper/tables/mechanism.tex, paper/figures/mech_*.pdf, paper/mechanism.json.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from collapse.utils import read_jsonl                                         # noqa: E402
from label_free import ece_at, fit_t, log_conf, match_t, mean_entropy, round_logits, softmax_t  # noqa: E402
from paper_assets import FIG, TAB, auroc_np, discover, setup_mpl              # noqa: E402

RUNS = ROOT / "runs"

# Display order and labels. Protocol knobs first, then the loss baselines.
ORDER = [
    ("r100", "Standard (2 epochs, top-p 0.95, 1 sample)"),
    ("ancestral", "Exact sampling (top-p off)"),
    ("epochs1", "One epoch per round"),
    ("e1anc", "One epoch + exact sampling"),
    ("k2anc", "2 samples/prompt, exact"),
    ("k4anc", "4 samples/prompt, exact"),
    ("k4", "4 samples/prompt, top-p"),
    ("ensampler", "Entropy-neutral sampler"),
    ("selfverify", "Half data, 2 epochs (step-matched control)"),
    ("r50", "50% reference data"),
    ("r75", "25% reference data"),
    ("cal", "Focal loss (ForTIFAI CAL family)"),
    ("cal_g3", "Focal loss, gamma = 3"),
    ("cal_anchor", "Focal loss + 5% fresh anchor"),
    ("tce90", "TCE, tau 0.9 (ForTIFAI default)"),
    ("tce", "TCE, tau 0.95"),
    ("selfverify_top", "Self-verify, top 50%"),
    ("ls", "Label smoothing"),
    ("ent", "Entropy bonus"),
    ("fullft", "Full fine-tuning (no LoRA)"),
    ("model2", "Standard loop, Qwen2.5-1.5B"),
]
# Conditions run on a different base model: in the tables, kept out of the per-protocol figure.
OTHER_MODEL = {"model2": "Qwen/Qwen2.5-1.5B"}
LAB = dict(ORDER)
# Shorter labels for the per-protocol figure, matching the paper's names.
SHORT = {
    "r100": "Standard (CE)", "ancestral": "Exact sampling (top-$p$ off)", "epochs1": "One epoch per round",
    "e1anc": "One epoch + exact sampling", "k2anc": "2 exact samples/prompt", "k4anc": "4 exact samples/prompt",
    "k4": "4 samples/prompt, top-$p$", "ensampler": "Entropy-matched sampler", "selfverify": "Half data, 2 epochs",
    "r50": "50% reference data", "r75": "25% reference data", "cal": "Focal loss (CAL family)",
    "cal_g3": r"Focal loss, $\gamma$=3", "cal_anchor": "Focal loss + 5% fresh reference",
    "tce90": r"TCE, $\tau$=0.9", "tce": r"TCE, $\tau$=0.95", "selfverify_top": "Self-verification (top 50%)",
    "ls": "Label smoothing", "ent": "Entropy bonus", "fullft": "Full fine-tuning (no LoRA)",
    "model2": "Standard loop, Qwen2.5-1.5B",
}
ENT_LAB = {"r100": "Standard", "ancestral": "Exact sampling", "epochs1": "One epoch",
           "e1anc": "One epoch + exact"}
# Family colours shared with the two-axes figure; single-seed arms are grey.
FAMILY = {"r100": "#2a78d6", "selfverify": "#1baf7a", "epochs1": "#1baf7a", "tce90": "#1baf7a",
          "ent": "#1baf7a", "cal": "#eb6834", "k2anc": "#eb6834", "r50": "#4a3aa7"}


def avg_precision(score: np.ndarray, pos: np.ndarray) -> float:
    """Average precision of `score` for the positives in `pos` (no interpolation)."""
    p = pos[np.argsort(-score, kind="mergesort")]
    prec = np.cumsum(p) / np.arange(1, len(p) + 1)
    return float((prec * p).sum() / max(p.sum(), 1))


def margin(L: np.ndarray) -> np.ndarray:
    """Gap between the two highest option log-probabilities (a label-free confidence)."""
    z = np.sort(L, axis=1)
    return z[:, -1] - z[:, -2]


def top_acc(score: np.ndarray, correct: np.ndarray, frac: float = 0.2) -> float:
    """Accuracy among the `frac` of items the model is most confident about."""
    k = int(frac * len(score))
    return float(correct[np.argsort(-score, kind="mergesort")[:k]].mean())


def arm_stats(a: dict) -> dict:
    n = a["name"]
    ent = []
    for g in range(5):
        f = RUNS / n / f"gen{g}" / "synthetic.jsonl"
        v = [r["mean_entropy"] for r in read_jsonl(f) if r.get("mean_entropy") is not None]
        ent.append(float(np.mean(v)))
    L0, y0, _, _ = round_logits(a, 0)
    L5, y5, _, _ = round_logits(a, 5)
    curves = a["summary"]["summary"]["curves"]["global_mmlu_en"]
    k = len(y0)
    perm = np.random.default_rng(0).permutation(k)
    # AUROC after label-free recalibration; a temperature need not preserve the
    # ranking of max-probabilities across items, so this is measured, not assumed.
    Z = L5 / match_t(L5, mean_entropy(L0, fit_t(L0, y0)))
    zmax = Z.max(1)
    lconf_tr = zmax - (np.log(np.exp(Z - zmax[:, None]).sum(1)) + zmax)
    oc = fx = tr = orc = gap0 = 0.0
    for fi, ev in ((perm[: k // 2], perm[k // 2:]), (perm[k // 2:], perm[: k // 2])):
        T0 = fit_t(L0[fi], y0[fi])
        P0 = softmax_t(L0[ev], T0)  # the same held-out gap at the calibrated start
        gap0 += (P0.max(1).mean() - (P0.argmax(1) == y0[ev]).mean()) / 2
        P = softmax_t(L5[ev], T0)
        oc += (P.max(1).mean() - (P.argmax(1) == y5[ev]).mean()) / 2
        fx += ece_at(L5[ev], y5[ev], T0) / 2
        tr += ece_at(L5[ev], y5[ev], match_t(L5, mean_entropy(L0, T0))) / 2
        orc += ece_at(L5[ev], y5[ev], fit_t(L5[fi], y5[fi])) / 2
    return {
        "name": n, "seed": a["seed"], "cond": a["cond"],
        "entropy": ent, "kept": float(np.exp(np.mean(np.diff(np.log(ent))))),
        "acc5": curves["accuracy_scored"][5],
        "raw_decE": curves["ece_internal"][5] - curves["ece_internal"][0],
        "auroc5": auroc_np(log_conf(L5), (L5.argmax(1) == y5).astype(float)),
        "auroc5_tracked": auroc_np(lconf_tr, (L5.argmax(1) == y5).astype(float)),
        "margin5": auroc_np(margin(L5), (L5.argmax(1) == y5).astype(float)),
        "ap5": avg_precision(log_conf(L5), (L5.argmax(1) == y5).astype(float)),
        "top20_5": top_acc(log_conf(L5), (L5.argmax(1) == y5).astype(float)),
        "ties5": int(len(L5) - len(np.unique(log_conf(L5)))),
        "calib_start": float(oc), "gap0": float(gap0),
        "ece_fixed": float(fx), "ece_tracked": float(tr), "ece_oracle": float(orc),
    }


def pm(v, nd=3, sign=False):
    v = np.asarray(v, float)
    f = (lambda x: f"{x:+.{nd}f}") if sign else (lambda x: f"{x:.{nd}f}")
    return f(v.mean()) + (f" $\\pm$ {v.std(ddof=1):.{nd}f}" if len(v) > 1 else "")


def main():
    arms = [a for a in discover() if a["valid"] and a["lang"] == "en" and a["size"] == "1.7B"
            and a["model"] == OTHER_MODEL.get(a["cond"], "Qwen/Qwen3-1.7B") and a["cond"] in LAB]
    by = defaultdict(list)
    for a in arms:
        s = arm_stats(a)
        by[s["cond"]].append(s)
        print(f"{s['name']:22} kept {s['kept']:.3f} acc {s['acc5']:.3f} AUROC {s['auroc5']:.3f} calib-start {s['calib_start']:+.3f}")
    (ROOT / "paper" / "mechanism.json").write_text(json.dumps(by, indent=1), encoding="utf-8")

    # ---------------- table
    rows = []
    for cond, lab in ORDER:
        S = by.get(cond)
        if not S:
            continue
        rows.append(" & ".join([
            lab, str(len(S)), pm([s["kept"] for s in S]), pm([s["acc5"] for s in S]),
            pm([s["auroc5"] for s in S]), pm([s["calib_start"] for s in S], sign=True),
            pm([s["ece_fixed"] for s in S]), pm([s["ece_tracked"] for s in S])]) + r"\\")
    tex = (r"""\begin{tabular}{@{}lclllrll@{}}
\toprule
Self-training protocol & $n$ & Entropy kept/round & Acc.\ $g_5$ & AUROC $g_5$ & Calibrated-start drift & ECE, stale $T_0$ & ECE, tracked \\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")
    (TAB / "mechanism.tex").write_text(tex, encoding="utf-8")

    plt = setup_mpl()
    # ---------------- figure 1: model entropy on its own targets, by round
    fig, ax = plt.subplots(figsize=(3.2, 2.6))
    # colours follow the family key of the other figures; single-seed arms are grey
    show = [("r100", "#2a78d6", "o"), ("ancestral", "#8a8a86", "s"),
            ("epochs1", "#1baf7a", "^"), ("e1anc", "#3d3d3a", "D")]
    xs = np.arange(5)
    for cond, col, mk in show:
        S = by.get(cond)
        if not S:
            continue
        E = np.array([s["entropy"] for s in S]); E = E / E[:, :1]
        for row in E:
            ax.plot(xs, row, color=col, alpha=0.3, lw=0.8)
        ax.plot(xs, E.mean(0), color=col, marker=mk, ms=3.5, mfc="white",
                label=f"{ENT_LAB[cond]} (n={len(S)})")
    ax.axhline(1.0, color="#8a8a86", lw=0.6, ls=":")
    ax.set_xticks(xs); ax.set_xticklabels([f"$g_{i}$" for i in xs])
    ax.set_xlabel("Generation that produced the training targets")
    ax.set_ylabel("Model entropy on own targets\n(relative to $g_0$)")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "mech_entropy.pdf", bbox_inches="tight"); plt.close(fig)

    # ---------------- figure 2: every protocol, (a) drift from a calibrated start, (b) ECE after recalibration
    items = [(c, by[c]) for c, _ in ORDER if c in by and c not in OTHER_MODEL]
    items.sort(key=lambda it: np.mean([s["calib_start"] for s in it[1]]))
    gap0 = float(np.mean([s["gap0"] for c, S in items for s in S]))
    fig, (axd, axe) = plt.subplots(1, 2, figsize=(6.3, 0.17 * len(items) + 1.0), sharey=True)
    axd.axvspan(-0.02, 0.02, color="#1baf7a", alpha=0.10, lw=0)
    axd.axvline(0, color="#8a8a86", lw=0.7)
    axd.axvline(gap0, color="#2b2b2a", lw=0.8, ls=":")
    axd.text(gap0 + 0.004, len(items) - 0.75, "$g_0$", fontsize=6.5, color="#2b2b2a", va="center")
    for i, (cond, S) in enumerate(items):
        col = FAMILY.get(cond, "#8a8a86")
        v = np.array([s["calib_start"] for s in S])
        axd.scatter(v, [i] * len(v), s=11, facecolor="white", edgecolor=col, lw=0.9, zorder=3)
        axd.plot([v.mean()], [i], marker="|", ms=9, color=col, mew=1.6, zorder=4)
        m = {k: np.mean([s[k] for s in S]) for k in ("ece_fixed", "ece_tracked", "ece_oracle")}
        axe.plot([m["ece_fixed"], m["ece_tracked"]], [i, i], color="#d5d4ce", lw=1.0, zorder=1)
        axe.scatter([m["ece_fixed"]], [i], s=14, facecolor="white", edgecolor="#8a8a86", lw=0.9, zorder=3,
                    label="stale $T_0$ (labels at $g_0$ only)" if i == 0 else None)
        axe.scatter([m["ece_oracle"]], [i], s=40, marker="|", color="#2b2b2a", lw=1.2, zorder=4,
                    label="oracle (labels every round)" if i == 0 else None)
        axe.scatter([m["ece_tracked"]], [i], s=14, color="#2a78d6", zorder=5,
                    label="label-free (entropy matching)" if i == 0 else None)
    axd.set_yticks(range(len(items)))
    axd.set_yticklabels([SHORT[c] for c, _ in items], fontsize=6)
    for t, (cond, S) in zip(axd.get_yticklabels(), items):
        t.set_color("#2b2b2a" if len(S) > 1 else "#8a8a86")
    axd.set_ylim(-0.7, len(items) - 0.3)
    axd.set_xlabel("Gap at $g_5$ under the $g_0$ temperature ($+$ over-, $-$ under-confident)")
    axe.set_xlabel("ECE at $g_5$")
    axd.set_title("(a) Calibration gap under $T_0$", fontsize=7, loc="left")
    axe.set_title("(b) ECE after recalibration", fontsize=7, loc="left")
    axe.legend(frameon=False, fontsize=5.5, loc="upper right")
    for ax in (axd, axe):
        ax.grid(axis="y", visible=False)
    fig.tight_layout(w_pad=1.0); fig.savefig(FIG / "mech_protocols.pdf", bbox_inches="tight"); plt.close(fig)
    print("wrote", TAB / "mechanism.tex", FIG / "mech_entropy.pdf", FIG / "mech_protocols.pdf")


if __name__ == "__main__":
    main()
