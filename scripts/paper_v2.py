r"""Numbers, tables and the headline figure for the analysis paper.

    .venv\Scripts\python.exe scripts\mechanism_assets.py   # per-arm stats -> paper/mechanism.json
    .venv\Scripts\python.exe scripts\label_free.py         # tracking at 1.7B and 8B -> paper/label_free.json
    .venv\Scripts\python.exe scripts\paper_v2.py           # this file

Writes
  paper/numbers.tex              \newcommand for every number the prose quotes
  paper/tables/protocols.tex     the seven replicated protocols (3 seeds each)
  paper/tables/all_arms.tex      every arm, one row per condition (appendix)
  paper/tables/tracking.tex      stale / label-free / oracle recalibration, 1.7B and 8B
  paper/figures/two_axes.pdf     calibration drift vs discrimination (headline)
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from label_free import ece_at, fit_t, log_conf, round_logits, softmax_t   # noqa: E402
from paper_assets import FIG, TAB, auroc_np, discover, setup_mpl           # noqa: E402
from mechanism_assets import avg_precision, margin, top_acc                # noqa: E402

P = ROOT / "paper"
MECH = json.load(io.open(P / "mechanism.json", encoding="utf-8"))
LF = json.load(io.open(P / "label_free.json", encoding="utf-8"))
PROBE = json.load(io.open(P / "sampler_probe.json", encoding="utf-8")) if (P / "sampler_probe.json").exists() else {}

# condition key -> (macro stem, display label, family)
REPL = [
    ("r100", "ce", "Standard self-training (CE)", "base"),
    ("selfverify", "half", "Half data, 2 epochs", "hold"),
    ("epochs1", "oneep", "One epoch per round", "hold"),
    ("tce90", "tce", r"TCE, $\tau{=}0.9$ \citep{shabgahi2025fortifai}", "hold"),
    ("ent", "ent", r"Entropy bonus \citep{pereyra2017regularizing}", "hold"),
    ("cal", "focal", r"Focal loss (CAL family) \citep{shabgahi2025fortifai}", "disc"),
    ("k2anc", "ktwo", "Two exact samples per prompt", "disc"),
    ("r50", "mix", r"50\% reference data \citep{gerstgrasser2024}", "ref"),
]
SINGLE = [
    ("ancestral", "anc", "Exact sampling (top-$p$ off)"),
    ("e1anc", "eoneanc", "One epoch + exact sampling"),
    ("k4anc", "kfouranc", "Four exact samples per prompt"),
    ("k4", "kfour", "Four samples per prompt, top-$p$"),
    ("ensampler", "ens", "Entropy-matched sampler"),
    ("tce", "tcehi", r"TCE, $\tau{=}0.95$"),
    ("cal_g3", "focalthree", r"Focal loss, $\gamma{=}3$"),
    ("cal_anchor", "focalanchor", "Focal loss + 5\\% fresh reference"),
    ("ls", "ls", "Label smoothing"),
    ("r75", "mixquarter", "25\\% reference data"),
    ("selfverify_top", "svtop", "Self-verification (top 50\\%)"),
    ("fullft", "fullft", "Full fine-tuning (no LoRA)"),
    ("model2", "mtwo", "Standard loop, Qwen2.5-1.5B (own $g_0$)"),
]

macros: list[str] = []


def mac(name: str, value: str) -> None:
    macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def f3(x, sign=False):
    return f"{x:+.3f}".replace("-", "$-$") if sign else f"{x:.3f}"


def fp(p):
    return f"{p:.3f}" if p >= 0.001 else f"{p:.4f}" if p >= 0.0001 else f"{p:.0e}"


def arr(cond, key):
    return np.array([s[key] for s in MECH.get(cond, [])], float)


def main():
    # ---------------- baseline g0 (shared by every arm)
    base_arm = next(a for a in discover() if a["name"] == "main_en_r100_s1")
    L0, y0, _, _ = round_logits(base_arm, 0)
    P0 = softmax_t(L0, 1.0)
    k = len(y0); perm = np.random.default_rng(0).permutation(k)
    T0s, ece_cal0, gap0 = [], 0.0, 0.0
    for fi, ev in ((perm[: k // 2], perm[k // 2:]), (perm[k // 2:], perm[: k // 2])):
        T0 = fit_t(L0[fi], y0[fi]); T0s.append(T0); ece_cal0 += ece_at(L0[ev], y0[ev], T0) / 2
        Pe = softmax_t(L0[ev], T0)
        gap0 += (Pe.max(1).mean() - (Pe.argmax(1) == y0[ev]).mean()) / 2
    # The calibration gap is measured under T0; at g0 itself it is this (held-out) value, not zero.
    mac("baseGap", f3(gap0, sign=True))
    mac("baseAcc", f3((P0.argmax(1) == y0).mean()))
    mac("baseConf", f3(P0.max(1).mean()))
    mac("baseAUROC", f3(auroc_np(log_conf(L0), (L0.argmax(1) == y0).astype(float))))
    mac("baseECEraw", f3(ece_at(L0, y0, 1.0)))
    mac("baseECEcal", f3(ece_cal0))
    mac("baseTzero", f"{np.mean(T0s):.1f}")
    mac("nItems", str(k))
    sat = int(np.sum(np.array([r["p_pred"] for r in __import__("collapse.utils", fromlist=["read_jsonl"]).read_jsonl(
        ROOT / "runs" / "main_en_r100_s1" / "gen0" / "eval_global_mmlu_en.jsonl")]) >= 0.999999))
    mac("baseSaturated", str(sat))
    c0 = (L0.argmax(1) == y0).astype(float)
    mac("baseTies", str(int(len(L0) - len(np.unique(log_conf(L0))))))
    mac("baseAP", f3(avg_precision(log_conf(L0), c0)))
    mac("baseMargin", f3(auroc_np(margin(L0), c0)))
    mac("baseTopTwenty", f3(top_acc(log_conf(L0), c0)))

    # Standard loop, g0 -> g5, paired item bootstrap per seed: AP falls in every seed while
    # the AUROC interval includes zero in every seed. The prose states both; fail loudly if not.
    rng = np.random.default_rng(0)
    ap_drop, auc_ci, ap_ci = [], [], []
    for a in sorted((a for a in discover() if a["valid"] and a["lang"] == "en" and a["size"] == "1.7B"
                     and a["model"] == "Qwen/Qwen3-1.7B" and a["cond"] == "r100"), key=lambda a: a["name"]):
        A0, b0, _, _ = round_logits(a, 0)
        A5, b5, _, _ = round_logits(a, 5)
        s0, s5 = log_conf(A0), log_conf(A5)
        k0, k5 = (A0.argmax(1) == b0).astype(float), (A5.argmax(1) == b5).astype(float)
        ap_drop.append(avg_precision(s5, k5) - avg_precision(s0, k0))
        bs_auc, bs_ap = [], []
        for _ in range(2000):
            i = rng.integers(0, len(k0), len(k0))
            bs_auc.append(auroc_np(s5[i], k5[i]) - auroc_np(s0[i], k0[i]))
            bs_ap.append(avg_precision(s5[i], k5[i]) - avg_precision(s0[i], k0[i]))
        auc_ci.append(np.percentile(bs_auc, [2.5, 97.5])); ap_ci.append(np.percentile(bs_ap, [2.5, 97.5]))
    assert all(hi < 0 for _, hi in ap_ci), ap_ci
    assert all(lo < 0 < hi for lo, hi in auc_ci), auc_ci
    mac("ceAPdropMin", f3(-max(ap_drop))); mac("ceAPdropMax", f3(-min(ap_drop)))

    # ---------------- replicated protocols
    ce_auc, ce_drift = arr("r100", "auroc5"), arr("r100", "calib_start")
    for cond, stem, _, _ in REPL:
        a, d = arr(cond, "auroc5"), arr(cond, "calib_start")
        mac(stem + "N", str(len(a)))
        mac(stem + "AUROC", f3(a.mean())); mac(stem + "AUROCsd", f3(a.std(ddof=1)))
        mac(stem + "Drift", f3(d.mean(), sign=True)); mac(stem + "Driftsd", f3(d.std(ddof=1)))
        mac(stem + "Shift", f3(d.mean() - gap0, sign=True))  # change of the gap from g0 to g5
        mac(stem + "Acc", f3(arr(cond, "acc5").mean()))
        mac(stem + "Kept", f3(arr(cond, "kept").mean()))
        mac(stem + "Tracked", f3(arr(cond, "ece_tracked").mean()))
        mac(stem + "Stale", f3(arr(cond, "ece_fixed").mean()))
        mac(stem + "Oracle", f3(arr(cond, "ece_oracle").mean()))
        mac(stem + "RawdECE", f3(arr(cond, "raw_decE").mean(), sign=True))
        mac(stem + "AUROCtracked", f3(arr(cond, "auroc5_tracked").mean()))
        mac(stem + "AP", f3(arr(cond, "ap5").mean()))
        mac(stem + "Margin", f3(arr(cond, "margin5").mean()))
        mac(stem + "TopTwenty", f3(arr(cond, "top20_5").mean()))
        assert arr(cond, "ties5").sum() == 0, (cond, "tied confidences")
        if cond != "r100":
            for key, nm in (("ap5", "AP"), ("margin5", "Margin")):
                v, w = arr(cond, key), arr("r100", key)
                mac(stem + "d" + nm, f3(v.mean() - w.mean(), sign=True))
                mac(stem + "p" + nm, fp(ttest_ind(v, w, equal_var=False).pvalue))
            mac(stem + "dAUROC", f3(a.mean() - ce_auc.mean(), sign=True))
            mac(stem + "pAUROC", fp(ttest_ind(a, ce_auc, equal_var=False).pvalue))
            mac(stem + "pDrift", fp(ttest_ind(d, ce_drift, equal_var=False).pvalue))
    hold_p = [ttest_ind(arr(c, "calib_start"), ce_drift, equal_var=False).pvalue for c in ("tce90", "ent", "epochs1")]
    mac("holdMaxpDrift", fp(max(hold_p)))
    hold_pa = [ttest_ind(arr(c, "auroc5"), ce_auc, equal_var=False).pvalue for c in ("tce90", "ent", "epochs1")]
    mac("holdMinpAUROC", fp(min(hold_pa)))
    mac("baseOverconf", f3(P0.max(1).mean() - (P0.argmax(1) == y0).mean()))
    fa, ha = arr("cal", "auroc5"), arr("selfverify", "auroc5")
    mac("focalVsHalfdAUROC", f3(fa.mean() - ha.mean(), sign=True))
    mac("focalVsHalfp", fp(ttest_ind(fa, ha, equal_var=False).pvalue))

    # ---------------- single-seed arms
    for cond, stem, _ in SINGLE:
        S = MECH.get(cond)
        if not S:
            continue
        mac(stem + "Kept", f3(np.mean([s["kept"] for s in S])))
        mac(stem + "Drift", f3(np.mean([s["calib_start"] for s in S]), sign=True))
        mac(stem + "AUROC", f3(np.mean([s["auroc5"] for s in S])))
        mac(stem + "Acc", f3(np.mean([s["acc5"] for s in S])))
        mac(stem + "Tracked", f3(np.mean([s["ece_tracked"] for s in S])))
        mac(stem + "Oracle", f3(np.mean([s["ece_oracle"] for s in S])))
        mac(stem + "Stale", f3(np.mean([s["ece_fixed"] for s in S])))
    # label-free vs oracle ECE gap at g5, every protocol except label smoothing (the one exception)
    gaps = {c: abs(arr(c, "ece_tracked").mean() - arr(c, "ece_oracle").mean()) for c in MECH}
    # a bound, so round up: "within X" must hold for every protocol it covers
    mac("trackMaxGap", f"{np.ceil(1000 * max(v for c, v in gaps.items() if c != 'ls')) / 1000:.3f}")
    mac("trackNprot", str(len(gaps)))
    fa_tr, ce_tr = arr("cal", "auroc5_tracked"), arr("r100", "auroc5_tracked")
    mac("focaldAUROCtracked", f3(fa_tr.mean() - ce_tr.mean(), sign=True))
    mac("focalpAUROCtracked", fp(ttest_ind(fa_tr, ce_tr, equal_var=False).pvalue))
    e = MECH["e1anc"][0]["entropy"]
    mac("eoneancEntropyFirst", f3(e[0])); mac("eoneancEntropyLast", f3(e[-1]))
    ce_e = np.array([s["entropy"] for s in MECH["r100"]]).mean(0)
    mac("ceEntropyFirst", f3(ce_e[0])); mac("ceEntropyLast", f3(ce_e[-1]))
    mac("ceEntropyLost", f"{100 * (1 - ce_e[-1] / ce_e[0]):.0f}")

    # ---------------- sampler probe
    if PROBE:
        mac("probeKeepGzero", f"{100 * PROBE['g0']['ratio']:.0f}")
        mac("probeKeepGfive", f"{100 * PROBE['g5']['ratio']:.0f}")
        mac("probeGreedyGzero", f"{100 * PROBE['g0']['greedy_frac']:.0f}")
        mac("probeGreedyGfive", f"{100 * PROBE['g5']['greedy_frac']:.0f}")

    # ---------------- 8B recalibration (label_free.json, round 5)
    rows8 = [r for r in LF["arms"] if r["size"] == "8B"]
    for r in rows8:
        stem = {"r100": "eightce", "r50": "eighthalf", "r0": "eightreal"}.get(r["cond"])
        if stem and r["seed"] == 1:
            mac(stem + "Stale", f3(r["ece"]["fixed"][5])); mac(stem + "Tracked", f3(r["ece"]["tracked"][5]))
            mac(stem + "Oracle", f3(r["ece"]["oracle"][5]))

    (P / "numbers.tex").write_text("% generated by scripts/paper_v2.py -- do not edit\n" + "\n".join(macros) + "\n",
                                   encoding="utf-8")

    # ---------------- table: replicated protocols
    def pm(v, sign=False):
        v = np.asarray(v, float)
        return f"{f3(v.mean(), sign)} {{\\scriptsize$\\pm${v.std(ddof=1):.3f}}}"
    lines = []
    fam_rule = {"base": None, "hold": r"\multicolumn{7}{@{}l}{\textit{Stay calibrated under the starting temperature}}\\",
                "disc": r"\multicolumn{7}{@{}l}{\textit{Improve discrimination}}\\",
                "ref": r"\multicolumn{7}{@{}l}{\textit{Mix in reference data}}\\"}
    seen = set()
    for cond, stem, lab, fam in REPL:
        if fam not in seen and fam_rule[fam]:
            lines.append(r"\midrule" + "\n" + fam_rule[fam]); seen.add(fam)
        elif fam == "base":
            seen.add(fam)
        S = MECH[cond]
        lines.append(" & ".join([lab, pm([s["kept"] for s in S]), pm([s["acc5"] for s in S]),
                                 pm([s["auroc5"] for s in S]), pm([s["calib_start"] for s in S], True),
                                 pm([s["ece_fixed"] for s in S]), pm([s["ece_tracked"] for s in S])]) + r"\\")
    (TAB / "protocols.tex").write_text(r"""\begin{tabular}{@{}lllllll@{}}
\toprule
 & Entropy kept & Accuracy & AUROC & Calibration gap & \multicolumn{2}{c}{ECE at $g_5$}\\
\cmidrule(l){6-7}
Protocol (3 seeds each) & per round & at $g_5$ & at $g_5$ & at $g_5$ under $T_0$ & stale $T_0$ & label-free\\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}""", encoding="utf-8")

    # ---------------- table: discrimination under three metrics (appendix)
    lines = [" & ".join([r"$g_0$ (shared start)", f3(auroc_np(log_conf(L0), c0)), f3(auroc_np(margin(L0), c0)),
                         f3(avg_precision(log_conf(L0), c0)), f3(top_acc(log_conf(L0), c0))]) + r"\\", r"\midrule"]
    for cond, _, lab, _ in REPL:
        lines.append(" & ".join([lab] + [pm(arr(cond, k)) for k in ("auroc5", "margin5", "ap5", "top20_5")]) + r"\\")
    (TAB / "discrimination.tex").write_text(r"""\begin{tabular}{@{}lllll@{}}
\toprule
Protocol ($g_5$; 3 seeds each) & AUROC & Margin AUROC & Average precision & Acc.\ of top 20\%\\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}""", encoding="utf-8")

    # ---------------- table: all arms (appendix)
    lines = []
    for cond, _, lab, _ in REPL:
        S = MECH[cond]
        lines.append(" & ".join([lab, str(len(S)), f3(np.mean([s["kept"] for s in S])),
                                 f3(np.mean([s["acc5"] for s in S])), f3(np.mean([s["auroc5"] for s in S])),
                                 f3(np.mean([s["calib_start"] for s in S]), True),
                                 f3(np.mean([s["raw_decE"] for s in S]), True)]) + r"\\")
    lines.append(r"\midrule")
    for cond, _, lab in SINGLE:
        S = MECH.get(cond)
        if not S:
            continue
        lines.append(" & ".join([lab, str(len(S)), f3(np.mean([s["kept"] for s in S])),
                                 f3(np.mean([s["acc5"] for s in S])), f3(np.mean([s["auroc5"] for s in S])),
                                 f3(np.mean([s["calib_start"] for s in S]), True),
                                 f3(np.mean([s["raw_decE"] for s in S]), True)]) + r"\\")
    (TAB / "all_arms.tex").write_text(r"""\begin{tabular}{@{}lcccccc@{}}
\toprule
Protocol & Seeds & Entropy kept/round & Acc.\ $g_5$ & AUROC $g_5$ & Gap under $T_0$ & Raw $\Delta$ECE\\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}""", encoding="utf-8")

    # ---------------- table: recalibration, 1.7B protocols + 8B arms
    lines = []
    for cond, _, lab, _ in REPL:
        S = MECH[cond]
        lines.append(" & ".join(["1.7B", lab] + [f3(np.mean([s[k] for s in S]))
                                                 for k in ("ece_fixed", "ece_tracked", "ece_oracle")]) + r"\\")
    lines.append(r"\midrule")
    lab8 = {"r100": "Standard (CE)", "r50": "50\\% reference data", "r0": "Reference data only"}
    for r in sorted(rows8, key=lambda r: (r["cond"] != "r100", r["cond"], r["seed"])):
        lines.append(" & ".join(["8B", f"{lab8.get(r['cond'], r['cond'])}, seed {r['seed']}"] +
                                [f3(r["ece"][k][5]) for k in ("fixed", "tracked", "oracle")]) + r"\\")
    (TAB / "tracking.tex").write_text(r"""\begin{tabular}{@{}llccc@{}}
\toprule
Model & Protocol & Stale $T_0$ & Label-free & Oracle\\
\midrule
""" + "\n".join(lines) + "\n" + r"""\bottomrule
\end{tabular}""", encoding="utf-8")

    # ---------------- figure: the two axes
    plt = setup_mpl()
    col = {"base": "#2a78d6", "hold": "#1baf7a", "disc": "#eb6834", "ref": "#4a3aa7"}
    short = {"r100": "standard (CE)", "selfverify": "half data", "epochs1": "one epoch", "tce90": "TCE 0.9",
             "ent": "entropy bonus", "cal": "focal (CAL)", "k2anc": "2 samples/prompt", "r50": "50% reference"}
    fig, ax = plt.subplots(figsize=(3.2, 2.7))
    ax.axvspan(-0.02, 0.02, color="#1baf7a", alpha=0.08, lw=0)
    ax.axvline(0, color="#8a8a86", lw=0.6)
    base_auc = auroc_np(log_conf(L0), (L0.argmax(1) == y0).astype(float))
    ax.axhline(base_auc, color="#8a8a86", lw=0.6, ls=":")
    ax.text(-0.21, base_auc + 0.0006, "$g_0$", fontsize=6, color="#52514e", va="bottom")
    # Label positions in data coordinates, with leader lines for the cluster near
    # zero drift, so no two labels overprint.
    lab_at = {"r100": (0.058, 0.698), "selfverify": (0.032, 0.728), "ent": (0.032, 0.717),
              "tce90": (0.032, 0.706), "epochs1": (-0.105, 0.733), "cal": (-0.195, 0.752),
              "k2anc": (-0.085, 0.752), "r50": (-0.205, 0.708)}
    for cond, _, _, fam in REPL:
        S = MECH[cond]
        x, y = np.array([s["calib_start"] for s in S]), np.array([s["auroc5"] for s in S])
        ax.errorbar(x.mean(), y.mean(), xerr=x.std(ddof=1), yerr=y.std(ddof=1), fmt="o", ms=4.5,
                    mfc="white", mec=col[fam], ecolor=col[fam], elinewidth=0.8, capsize=0, zorder=3)
        ax.annotate(short[cond], xy=(x.mean(), y.mean()), xytext=lab_at[cond], textcoords="data",
                    fontsize=6, color="#2b2b2a", va="center",
                    arrowprops=dict(arrowstyle="-", color="#b5b4ad", lw=0.5, shrinkA=0, shrinkB=3))
    # the shared starting model, at its own held-out gap under T0
    ax.plot([gap0], [base_auc], marker="*", ms=8, color="#2b2b2a", zorder=5, lw=0)
    ax.annotate("start ($g_0$)", xy=(gap0, base_auc), xytext=(0.062, 0.7125), textcoords="data",
                fontsize=6, color="#2b2b2a", va="center",
                arrowprops=dict(arrowstyle="-", color="#b5b4ad", lw=0.5, shrinkA=0, shrinkB=3))
    ax.set_xlabel("Calibration gap at $g_5$ under the $g_0$ temperature\n($+$ over-, $-$ under-confident)")
    ax.set_ylabel("AUROC at $g_5$ (discrimination)")
    ax.text(0.0, 0.691, "calibrated\nunder $T_0$", ha="center", va="bottom", fontsize=5.5, color="#1b7f5a")
    ax.set_xlim(-0.215, 0.105)
    ax.set_ylim(0.688, 0.762)
    fig.tight_layout(); fig.savefig(FIG / "two_axes.pdf", bbox_inches="tight"); plt.close(fig)
    print(f"wrote numbers.tex ({len(macros)} macros), protocols/all_arms/tracking tables, two_axes.pdf")


if __name__ == "__main__":
    main()
