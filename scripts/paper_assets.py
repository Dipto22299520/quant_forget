r"""Build every number, table and figure in the paper from the raw run records.

    .venv\Scripts\python.exe scripts\paper_assets.py            # all
    .venv\Scripts\python.exe scripts\paper_assets.py --no-boot  # skip bootstraps (fast preview)

Nothing in paper/ is typed by hand from a summary: tables are written to
paper/tables/*.tex, figures to paper/figures/*.pdf, and every scalar the prose
quotes is in paper/results.json. Re-run after any arm finishes.

Validity gate: Bangla arms are admitted ONLY at the corrected code fingerprint
(README finding #5). English arms are admitted at any study fingerprint because
no English prompt reaches the code path that changed.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from collapse.utils import read_jsonl                              # noqa: E402
from collapse.eval.calibration import ece as ece_fn, auroc as auroc_fn  # noqa: E402
from collapse.eval.calibration import adaptive_ece, brier as brier_fn   # noqa: E402
from collapse.analysis.inference import h1_ordering, MDE            # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "paper"
FIG = OUT / "figures"
TAB = OUT / "tables"

# ------------------------------------------------------------------ validity
BN_VALID_FP = {"fe7f45b17922", "2a7be7a589bc"}   # corrected pipeline; 2a7b adds only the quantile filter
EN_VALID_FP = {"a96c1f931adc", "10c791652a94", "fe7f45b17922", "2a7be7a589bc", "ab8a87ca8c78",
               "4cd0e20a5c4f", "e278137385fe", "15a77122795f"}   # 4cd0 exact logits; e278 samplers; 15a7 K samples
STUDY_MODEL = "Qwen/Qwen3-1.7B"
N_BOOT = 2000
SEED = 0

# Categorical palette (validated with the dataviz validator, all-pairs, light).
COL = {"r100": "#2a78d6", "r75": "#eb6834", "r50": "#1baf7a", "r0": "#4a3aa7",
       "fullft": "#eb6834", "model2": "#1baf7a", "cal": "#eb6834", "selfverify": "#1baf7a",
       "selfverify_top": "#4a3aa7", "tce": "#4a3aa7", "tce90": "#4a3aa7", "cal_g3": "#2a78d6",
       "cal_anchor": "#1baf7a", "ls": "#eb6834", "ent": "#1baf7a",
       "ensampler": "#1baf7a", "ancestral": "#eb6834", "epochs1": "#4a3aa7",
       "k4": "#2a78d6", "k2anc": "#eb6834", "k4anc": "#1baf7a", "e1anc": "#1baf7a"}
MARK = {"r100": "o", "r75": "s", "r50": "^", "r0": "D",
        "fullft": "s", "model2": "^", "cal": "s", "selfverify": "^", "selfverify_top": "D",
        "tce": "v", "tce90": "v", "cal_g3": "P", "cal_anchor": "X", "ls": "h", "ent": "p",
        "ensampler": "^", "ancestral": "s", "epochs1": "D", "k4": "o", "k2anc": "s", "k4anc": "^", "e1anc": "P"}
LABEL = {"r100": "100% synthetic", "r75": "75% synthetic", "r50": "50% synthetic",
         "r0": "0% synthetic (real only)", "fullft": "full fine-tune",
         "model2": "Qwen2.5-1.5B", "cal": "CAL loss", "selfverify": "self-verify filter (p>0.5)",
         "selfverify_top": "self-verify filter (top 50%)", "tce": "TCE",
         "cal_g3": "CAL gamma=3", "tce90": "TCE (tau 0.9, ForTIFAI default)", "cal_anchor": "CAL + 5% fresh anchor",
         "ls": "label smoothing", "ent": "entropy bonus",
         "ensampler": "entropy-neutral sampler", "ancestral": "ancestral sampling (top-p 1.0)",
         "epochs1": "one epoch per round", "k4": "4 samples/prompt",
         "k2anc": "2 samples/prompt, ancestral", "k4anc": "4 samples/prompt, ancestral",
         "e1anc": "one epoch + ancestral"}
LANG_NAME = {"en": "English", "bn": "Bangla"}
GROUPS = [("en", "1.7B"), ("en", "8B"), ("bn", "1.7B")]   # (language, model size) blocks
def group_name(lang, size):
    return f"{LANG_NAME[lang]}, Qwen3-{size}"


# ------------------------------------------------------------------ discovery
def arm_identity(run_dir: Path) -> dict | None:
    cfg_p = run_dir / "config.yaml"
    sum_p = run_dir / "summary.json"
    if not cfg_p.exists() or not sum_p.exists():
        return None
    cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8"))
    ev = cfg["data"]["eval_sets"]
    if cfg["loop"]["generations"] != 6 or not ev or not ev[0].startswith("global_mmlu"):
        return None
    summ = json.load(io.open(sum_p, encoding="utf-8"))
    fps = {m.get("code_fingerprint") for m in summ["metrics"]}
    lang = cfg["data"]["lang"]
    ok = fps <= (BN_VALID_FP if lang == "bn" else EN_VALID_FP)
    model = cfg["model"]["name"]
    ratio = float(cfg["loop"]["ratio"])
    if model == "Qwen/Qwen2.5-1.5B":
        cond = "model2"
    elif cfg["train"]["mode"] == "full":
        cond = "fullft"
    elif cfg["train"]["loss"] == "cal":
        if float(cfg["loop"]["ratio"]) < 1.0 and cfg["loop"].get("real_fresh", False):
            cond = "cal_anchor"
        elif abs(float(cfg["train"].get("cal_gamma", 1.0)) - 1.0) > 1e-9:
            cond = f"cal_g{int(round(float(cfg['train']['cal_gamma'])))}"
        else:
            cond = "cal"
    elif cfg["train"]["loss"] == "tce":
        thr = float(cfg["train"].get("tce_threshold", 0.95))
        cond = "tce" if abs(thr - 0.95) < 1e-9 else f"tce{int(round(thr * 100))}"
    elif cfg["train"]["loss"] in ("ls", "ent"):
        cond = cfg["train"]["loss"]
    elif cfg["loop"]["filter"] in ("selfverify", "selfverify_top"):
        cond = cfg["loop"]["filter"]
    elif int(cfg["train"]["epochs"]) == 1 and cfg["gen"].get("sampler", "standard") == "ancestral":
        cond = "e1anc"
    elif int(cfg["loop"].get("samples_per_prompt", 1)) > 1:
        k = int(cfg["loop"]["samples_per_prompt"])
        cond = f"k{k}" + ("anc" if cfg["gen"].get("sampler", "standard") == "ancestral" else "")
    elif cfg["gen"].get("sampler", "standard") != "standard":
        cond = {"entropy_neutral": "ensampler", "ancestral": "ancestral"}[cfg["gen"]["sampler"]]
    elif int(cfg["train"]["epochs"]) == 1:
        cond = "epochs1"
    else:
        cond = f"r{int(round(ratio * 100))}"
    size = {"Qwen/Qwen3-1.7B": "1.7B", "Qwen/Qwen3-4B": "4B", "Qwen/Qwen3-8B": "8B",
            "Qwen/Qwen2.5-1.5B": "1.7B"}.get(model, model)
    return {"name": run_dir.name, "lang": lang, "cond": cond, "seed": int(cfg["seed"]),
            "size": size,
            "ratio": ratio, "model": model, "mode": cfg["train"]["mode"],
            "loss": cfg["train"]["loss"], "filter": cfg["loop"]["filter"],
            "sft_size": cfg["loop"]["sft_size"], "max_len": cfg["model"]["max_len"],
            "train_max_len": cfg["train"]["max_len"],
            "bs": cfg["train"]["per_device_bs"], "accum": cfg["train"]["grad_accum"],
            "fingerprints": sorted(fps), "valid": ok, "eval_set": ev[0],
            "summary": summ}


def discover() -> list[dict]:
    arms = []
    for d in sorted(RUNS.iterdir()):
        a = arm_identity(d)
        if a is None:
            continue
        # A run that carries a mixed set of fingerprints or a non-Qwen3 model
        # that is not the designated second-model control is not a study arm.
        arms.append(a)
    return arms


# ------------------------------------------------------------------ item stats
def item_arrays(recs: list[dict], raw_file: Path | None = None) -> dict:
    """Per-item arrays for the primary and sensitivity signals.

    Undebiased option scores were not stored by the loop; scripts/rescore_raw.py
    recovers them for g0 and g5 into paper/raw_scores/, keyed by uid.
    """
    pp = np.array([r["p_pred"] for r in recs], dtype=float)
    cs = np.array([r["correct_scored"] for r in recs], dtype=float)
    raw = [r.get("choice_probs_raw") for r in recs]
    if raw_file is not None and raw_file.exists():
        by_uid = {x["uid"]: x["choice_probs_raw"] for x in read_jsonl(raw_file)}
        raw = [by_uid.get(r["uid"]) for r in recs]
    if all(x for x in raw):
        pr = np.array([max(x) for x in raw])
        cr = np.array([int(int(np.argmax(x)) == r["answer_idx"]) for x, r in zip(raw, recs)],
                      dtype=float)
    else:
        pr = np.full(len(recs), np.nan); cr = np.full(len(recs), np.nan)
    cg = np.array([int(r.get("correct_generated") or 0) for r in recs], dtype=float)
    hb = np.array([bool(r.get("hit_budget", (r.get("n_gen_tokens") or 0) >= 32)) for r in recs])
    return {"p": pp, "y": cs, "p_raw": pr, "y_raw": cr, "y_gen": cg, "hit_budget": hb,
            "uid": [r["uid"] for r in recs]}


def ece_np(p, y, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins).astype(float)
    sp = np.bincount(idx, weights=p, minlength=n_bins)
    sy = np.bincount(idx, weights=y, minlength=n_bins)
    m = cnt > 0
    return float(np.sum(np.abs(sp[m] - sy[m])) / len(p))


def auroc_np(p, y):
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


STATS = {
    "acc": lambda p, y: float(y.mean()),
    "ece": lambda p, y: ece_np(p, y, 15),
    "overconf": lambda p, y: float(p.mean() - y.mean()),
    "brier": lambda p, y: float(np.mean((p - y) ** 2)),
    "auroc": auroc_np,
}


def paired_boot(a: dict, b: dict, key: str, n_boot=N_BOOT, seed=SEED, sig="p") -> dict:
    """Δ = stat(b) − stat(a) on shared items, item-level paired bootstrap."""
    pk, yk = (sig, "y") if sig == "p" else ("p_raw", "y_raw")
    pa, ya, pb, yb = a[pk], a[yk], b[pk], b[yk]
    f = STATS[key]
    point = f(pb, yb) - f(pa, ya)
    rng = np.random.default_rng(seed)
    n = len(pa)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = f(pb[idx], yb[idx]) - f(pa[idx], ya[idx])
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    pval = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
    return {"delta": point, "lo": float(lo), "hi": float(hi), "p": float(min(1, pval))}


# ------------------------------------------------------------------ per-arm
def analyse_arm(a: dict, do_boot: bool) -> dict:
    d = RUNS / a["name"]
    es = a["eval_set"]
    gens = []
    arrays = []
    for g in range(6):
        recs = read_jsonl(d / f"gen{g}" / f"eval_{es}.jsonl")
        arr = item_arrays(recs, OUT / "raw_scores" / f"{a['name']}_g{g}.jsonl")
        arrays.append(arr)
        p, y = arr["p"], arr["y"]
        gens.append({
            "g": g, "acc": float(y.mean()), "acc_gen": float(arr["y_gen"].mean()),
            "conf": float(p.mean()), "ece": ece_np(p, y), "ece10": ece_np(p, y, 10),
            "ece20": ece_np(p, y, 20), "aece": float(adaptive_ece(p, y)),
            "brier": float(np.mean((p - y) ** 2)), "auroc": auroc_np(p, y),
            "overconf": float(p.mean() - y.mean()),
            "ece_raw": ece_np(arr["p_raw"], arr["y_raw"]) if np.isfinite(arr["p_raw"]).all() else float("nan"),
            "acc_raw": float(arr["y_raw"].mean()) if np.isfinite(arr["y_raw"]).all() else float("nan"),
            "budget_hit": float(arr["hit_budget"].mean()),
        })
    # synthetic-data signal the NEXT model trains on (gens 0..4) + training loss
    syn = []
    for g in range(5):
        f = d / f"gen{g}" / "synthetic.jsonl"
        done = d / "ckpt" / f"model_{g + 1}" / "_done.json"
        row = {"g": g}
        if f.exists():
            R = read_jsonl(f)
            ent = [r["mean_entropy"] for r in R if r.get("mean_entropy") is not None]
            mtp = [r["mean_token_prob"] for r in R if r.get("mean_token_prob") is not None]
            row.update(entropy=float(np.mean(ent)), token_prob=float(np.mean(mtp)),
                       n=len(R), correct_generated=float(np.mean([r.get("correct_generated") or 0 for r in R])),
                       budget_hit=float(np.mean([bool(r.get("hit_budget", (r.get("n_gen_tokens") or 0) >= 32)) for r in R])))
        if done.exists():
            info = json.load(io.open(done, encoding="utf-8"))
            h = info["train_history"]
            row.update(loss_first=h[0]["loss"], loss_last=h[-1]["loss"],
                       n_syn=info["n_syn_pairs"], n_real=info["n_real_pairs"],
                       mix=info["mix"])
        syn.append(row)

    out = {**{k: v for k, v in a.items() if k != "summary"}, "gens": gens, "synthetic": syn}
    if do_boot:
        out["delta05"] = {k: paired_boot(arrays[0], arrays[5], k) for k in STATS}
        out["delta15"] = {k: paired_boot(arrays[1], arrays[5], k) for k in ("ece", "acc", "overconf")}
        out["delta01"] = {k: paired_boot(arrays[0], arrays[1], k) for k in ("ece", "acc", "overconf")}
        if np.isfinite(arrays[0]["p_raw"]).all():
            out["delta05_raw"] = {k: paired_boot(arrays[0], arrays[5], k, sig="raw")
                                  for k in ("ece", "acc")}
        # Pre-declared H1 ordering test (only meaningful for the primary arms,
        # but cheap and reported for every arm in the appendix).
        recs_by_gen = [read_jsonl(d / f"gen{g}" / f"eval_{es}.jsonl") for g in range(6)]
        h1 = h1_ordering(recs_by_gen, seed=SEED)
        out["h1"] = {k: v for k, v in h1.items()
                     if k in ("testable", "reason", "calibration_onset", "accuracy_onset",
                              "h1_supported", "verdict", "accuracy_g0", "floor", "chance_level")}
        out["h1"]["per_metric"] = {
            m: {"onset": h1["calibration"][m]["onset"],
                "p_min": h1["calibration"][m]["p_min"],
                "holm_significant": h1["holm"].get(m, {}).get("significant")}
            for m in ("ece", "overconfidence", "auroc")} if h1.get("testable") else {}
        out["h1"]["accuracy_p_min"] = h1["accuracy"]["p_min"] if h1.get("testable") else None
    return out


# ------------------------------------------------------------------ tables
def fmt(x, nd=3, sign=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def ci(dd, nd=3):
    if not dd:
        return "--"
    return f"{dd['delta']:+.{nd}f} [{dd['lo']:+.{nd}f}, {dd['hi']:+.{nd}f}]"


COND_ORDER = ["r100", "r75", "r50", "r0", "fullft", "model2", "ensampler", "ancestral", "epochs1",
              "e1anc", "k2anc", "k4anc", "k4",
              "cal", "cal_g3", "cal_anchor", "tce90", "tce", "selfverify", "selfverify_top", "ls", "ent"]
TEXLABEL = {k: v.replace("%", chr(92) + "%") for k, v in LABEL.items()}


def table_main(arms: list[dict]) -> str:
    rows = []
    for lang, size in GROUPS:
        if not any(a["lang"] == lang and a["size"] == size for a in arms):
            continue
        rows.append(r"\midrule" + "\n" + r"\multicolumn{8}{l}{\textit{" + group_name(lang, size) + r"}}\\")
        for cond in COND_ORDER:
            sel = sorted([a for a in arms if a["lang"] == lang and a["size"] == size and a["cond"] == cond],
                         key=lambda a: a["seed"])
            for a in sel:
                g0, g5 = a["gens"][0], a["gens"][5]
                d = a.get("delta05", {})
                rows.append(" & ".join([
                    TEXLABEL[cond], str(a["seed"]),
                    fmt(g0["acc"]), fmt(g5["acc"]), ci(d.get("acc")),
                    fmt(g0["ece"]), fmt(g5["ece"]), ci(d.get("ece"))]) + r"\\")
    body = "\n".join(rows)
    return (r"""\begin{tabular}{@{}llcclccl@{}}
\toprule
 & & \multicolumn{3}{c}{Accuracy (option-scored)} & \multicolumn{3}{c}{ECE ($p_\mathrm{pred}$, 15 bins)}\\
\cmidrule(lr){3-5}\cmidrule(lr){6-8}
Condition & Seed & $g_0$ & $g_5$ & $\Delta$ [95\% CI] & $g_0$ & $g_5$ & $\Delta$ [95\% CI]\\
""" + body + "\n" + r"""\bottomrule
\end{tabular}""")


def table_secondary(arms: list[dict]) -> str:
    rows = []
    for lang, size in GROUPS:
        if not any(a["lang"] == lang and a["size"] == size for a in arms):
            continue
        rows.append(r"\midrule" + "\n" + r"\multicolumn{7}{l}{\textit{" + group_name(lang, size) + r"}}\\")
        for cond in COND_ORDER:
            for a in sorted([a for a in arms if a["lang"] == lang and a["size"] == size and a["cond"] == cond],
                            key=lambda a: a["seed"]):
                g0, g5 = a["gens"][0], a["gens"][5]
                d = a.get("delta05", {})
                rows.append(" & ".join([
                    TEXLABEL[cond], str(a["seed"]),
                    f"{g0['conf']:.3f}$\\to${g5['conf']:.3f}",
                    ci(d.get("overconf")), ci(d.get("auroc")), ci(d.get("brier")),
                    ci(a.get("delta15", {}).get("ece"))]) + r"\\")
    return (r"""\begin{tabular}{@{}llclll l@{}}
\toprule
Condition & Seed & Mean conf.\ $g_0\to g_5$ & $\Delta$Overconf.\ & $\Delta$AUROC & $\Delta$Brier & $\Delta$ECE $g_1\to g_5$\\
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")


def table_h1(arms: list[dict]) -> str:
    rows = []
    for lang, size in GROUPS:
        for a in sorted([a for a in arms if a["lang"] == lang and a["size"] == size and a["cond"] == "r100"],
                        key=lambda a: a["seed"]):
            h = a.get("h1")
            if not h:
                continue
            if not h["testable"]:
                rows.append(f"{lang.upper()}, {size} & {a['seed']} & \\multicolumn{{5}}{{l}}{{untestable: {h['reason']}}}\\\\")
                continue
            pm = h["per_metric"]
            def cell(m):
                o = pm[m]["onset"]
                s = pm[m]["holm_significant"]
                return "--" if o is None else f"$g_{o}$" + (" $^\\dagger$" if s else "")
            sup = {True: "supported", False: "refuted", None: "undetermined"}[h["h1_supported"]]
            acc_on = "--" if h["accuracy_onset"] is None else f"$g_{h['accuracy_onset']}$"
            rows.append(" & ".join([f"{lang.upper()}, {size}", str(a["seed"]), cell("ece"),
                                    cell("overconfidence"), cell("auroc"), acc_on, sup]) + r"\\")
    return (r"""\begin{tabular}{@{}llccccl@{}}
\toprule
 & & \multicolumn{3}{c}{Calibration onset} & & \\
\cmidrule(lr){3-5}
Language & Seed & ECE & Overconf. & AUROC & Accuracy onset & H1\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")


def table_sensitivity(arms: list[dict]) -> str:
    rows = []
    for lang, size in GROUPS:
        for a in sorted([a for a in arms if a["lang"] == lang and a["size"] == size and a["cond"] in ("r100", "r50")],
                        key=lambda a: (a["cond"], a["seed"])):
            g0, g5 = a["gens"][0], a["gens"][5]
            dr = a.get("delta05_raw", {}).get("ece")
            rows.append(" & ".join([
                group_name(lang, size), TEXLABEL[a["cond"]], str(a["seed"]),
                fmt(g5["ece10"] - g0["ece10"], sign=True),
                fmt(g5["ece"] - g0["ece"], sign=True),
                fmt(g5["ece20"] - g0["ece20"], sign=True),
                fmt(g5["aece"] - g0["aece"], sign=True),
                ci(dr) if dr else "--",
                fmt(g5["acc_raw"] - g0["acc_raw"], sign=True)]) + r"\\")
    return (r"""\begin{tabular}{@{}lllccccll@{}}
\toprule
 & & & \multicolumn{4}{c}{$\Delta$ECE, debiased $p_\mathrm{pred}$} & \multicolumn{2}{c}{Undebiased option scores}\\
\cmidrule(lr){4-7}\cmidrule(lr){8-9}
Lang.\ & Condition & Seed & 10 bins & 15 bins & 20 bins & adaptive & $\Delta$ECE [95\% CI] & $\Delta$Acc.\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")


def table_provenance(arms: list[dict]) -> str:
    rows = []
    for lang, size in GROUPS:
        for cond in COND_ORDER:
            for a in sorted([a for a in arms if a["lang"] == lang and a["size"] == size and a["cond"] == cond],
                            key=lambda a: a["seed"]):
                syn = [s for s in a["synthetic"] if "n_syn" in s]
                ns = f"{min(s['n_syn'] for s in syn)}--{max(s['n_syn'] for s in syn)}" if syn else "--"
                nr = f"{min(s['n_real'] for s in syn)}--{max(s['n_real'] for s in syn)}" if syn else "--"
                rows.append(" & ".join([
                    group_name(lang, size), TEXLABEL[cond], str(a["seed"]),
                    f"{a['bs']}$\\times${a['accum']}", str(a["max_len"]),
                    ns, nr, "/".join(a["fingerprints"])]) + r"\\")
    return (r"""\begin{tabular}{@{}lllccccl@{}}
\toprule
Lang.\ & Condition & Seed & Micro-batch$\times$accum.\ & Window & Synthetic pairs & Real pairs & Code fingerprint\\
\midrule
""" + "\n".join(rows) + "\n" + r"""\bottomrule
\end{tabular}""")


# ------------------------------------------------------------------ figures
def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 8.5, "legend.fontsize": 7, "xtick.labelsize": 7,
        "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.5,
        "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.6, "lines.linewidth": 1.4,
        "pdf.fonttype": 42, "figure.dpi": 150,
    })
    return plt


def by_cond(arms, lang, conds, size="1.7B"):
    out = defaultdict(list)
    for a in arms:
        if a["lang"] == lang and a["cond"] in conds and a.get("size", "1.7B") == size:
            out[a["cond"]].append(a)
    return out


def _plot_traj(ax, groups, metric, conds, seeds_thin=True, rel=False):
    xs = np.arange(6)
    handles = []
    for cond in conds:
        arms = groups.get(cond, [])
        if not arms:
            continue
        Y = np.array([[g[metric] for g in a["gens"]] for a in arms])
        if rel:
            Y = Y - Y[:, :1]
        if seeds_thin and len(arms) > 1:
            for row in Y:
                ax.plot(xs, row, color=COL[cond], alpha=0.35, lw=0.8)
        h, = ax.plot(xs, Y.mean(0), color=COL[cond], marker=MARK[cond], ms=3.5,
                     mfc="white", mew=1.0,
                     label=f"{LABEL[cond]} (n={len(arms)})")
        handles.append(h)
    ax.set_xticks(xs); ax.set_xticklabels([f"$g_{i}$" for i in xs])
    return handles


def _equalise_span(axes_row, span):
    """Same y-span in every panel of a row, so slopes are comparable across languages."""
    for ax in axes_row:
        lo, hi = ax.get_ylim()
        mid = (lo + hi) / 2
        ax.set_ylim(mid - span / 2, mid + span / 2)


def fig_trajectories(arms, plt):
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 4.2), sharex=True)
    conds = ["r100", "r75", "r50", "r0"]
    for j, lang in enumerate(("en", "bn")):
        groups = by_cond(arms, lang, conds)
        _plot_traj(axes[0, j], groups, "ece", conds)
        _plot_traj(axes[1, j], groups, "acc", conds)
        axes[0, j].set_title(LANG_NAME[lang] + " -- Global-MMLU (600 items)")
        if not groups:
            axes[0, j].text(2.5, 0.5, "corrected arms pending", ha="center", color="#52514e")
    # Equal y-spans per row: a 0.03 change must look the same size in both languages.
    _equalise_span(axes[0], 0.16)
    _equalise_span(axes[1], 0.16)
    for j in range(2):
        lo, hi = axes[1, j].get_ylim()
        if lo <= 0.25 <= hi:
            axes[1, j].axhline(0.25, color="#8a8a86", lw=0.6, ls=":")
            axes[1, j].text(5.05, 0.252, "chance", fontsize=6, color="#52514e", ha="right", va="bottom")
        else:
            axes[1, j].text(5.05, lo + 0.004, "chance = 0.25 (off-scale)", fontsize=6,
                            color="#52514e", ha="right", va="bottom")
    axes[0, 0].set_ylabel("ECE ($p_\\mathrm{pred}$)")
    axes[1, 0].set_ylabel("Accuracy (option-scored)")
    axes[1, 0].set_xlabel("Generation"); axes[1, 1].set_xlabel("Generation")
    h, l = axes[0, 0].get_legend_handles_labels()
    if not h:
        h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "trajectories.pdf", bbox_inches="tight"); plt.close(fig)


def fig_tradeoff(arms, plt):
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
    for j, lang in enumerate(("en", "bn")):
        ax = axes[j]
        ax.axhline(0, color="#8a8a86", lw=0.6); ax.axvline(0, color="#8a8a86", lw=0.6)
        seen = set()
        for a in [a for a in arms if a["lang"] == lang and a["size"] == "1.7B" and "delta05" in a]:
            d = a["delta05"]
            ax.errorbar(d["acc"]["delta"], d["ece"]["delta"],
                        xerr=[[d["acc"]["delta"] - d["acc"]["lo"]], [d["acc"]["hi"] - d["acc"]["delta"]]],
                        yerr=[[d["ece"]["delta"] - d["ece"]["lo"]], [d["ece"]["hi"] - d["ece"]["delta"]]],
                        fmt=MARK[a["cond"]], color=COL[a["cond"]], mfc="white", ms=5,
                        elinewidth=0.6, capsize=0,
                        label=LABEL[a["cond"]] if a["cond"] not in seen else None)
            seen.add(a["cond"])
        ax.set_title(LANG_NAME[lang])
        ax.set_xlabel("$\\Delta$ accuracy ($g_5 - g_0$)")
        ax.text(0.98, 0.02, "better calibrated $\\downarrow$", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, color="#52514e")
        ax.legend(frameon=False, loc="upper left", ncol=1)
    axes[0].set_ylabel("$\\Delta$ ECE ($g_5 - g_0$)")
    fig.tight_layout()
    fig.savefig(FIG / "tradeoff.pdf", bbox_inches="tight"); plt.close(fig)


def fig_controls(arms, plt):
    fig, axes = plt.subplots(2, 1, figsize=(3.1, 4.0), sharex=True)
    conds = ["r100", "fullft", "model2"]
    for j, lang in enumerate(("en", "bn")):
        groups = by_cond(arms, lang, conds)
        _plot_traj(axes[j], groups, "ece", conds, rel=True)
        axes[j].axhline(0, color="#8a8a86", lw=0.6)
        axes[j].set_title(LANG_NAME[lang])
        axes[j].set_ylabel("ECE change from $g_0$")
        if groups:
            axes[j].legend(frameon=False, loc="best")
        else:
            axes[j].text(2.5, 0, "corrected arms pending", ha="center", color="#52514e")
    axes[1].set_xlabel("Generation")
    fig.tight_layout()
    fig.savefig(FIG / "controls.pdf", bbox_inches="tight"); plt.close(fig)


def fig_mitigation(arms, plt):
    fig, axes = plt.subplots(2, 1, figsize=(3.1, 4.0), sharex=True)
    conds = ["r100", "cal", "selfverify", "selfverify_top"]
    for j, lang in enumerate(("en", "bn")):
        groups = by_cond(arms, lang, conds)
        _plot_traj(axes[j], groups, "ece", conds, rel=True)
        axes[j].axhline(0, color="#8a8a86", lw=0.6)
        axes[j].set_title(LANG_NAME[lang])
        axes[j].set_ylabel("ECE change from $g_0$")
        if groups:
            axes[j].legend(frameon=False, loc="best")
        else:
            axes[j].text(2.5, 0, "corrected arms pending", ha="center", color="#52514e")
    axes[1].set_xlabel("Generation")
    fig.tight_layout()
    fig.savefig(FIG / "mitigation.pdf", bbox_inches="tight"); plt.close(fig)


def fig_mechanism(arms, plt):
    fig, axes = plt.subplots(2, 1, figsize=(3.1, 3.5), sharey=True, sharex=True)
    conds = ["r100", "r50", "r0"]
    xs = np.arange(5)
    for j, lang in enumerate(("en", "bn")):
        groups = by_cond(arms, lang, conds)
        ax = axes[j]
        for cond in conds:
            sel = groups.get(cond, [])
            if not sel:
                continue
            E = np.array([[s.get("entropy", np.nan) for s in a["synthetic"]] for a in sel])
            for row in E:
                ax.plot(xs, row, color=COL[cond], alpha=0.35, lw=0.8)
            ax.plot(xs, np.nanmean(E, 0), color=COL[cond], marker=MARK[cond], ms=3.5,
                    mfc="white", label=f"{LABEL[cond]} (n={len(sel)})")
        ax.set_title(LANG_NAME[lang])
        ax.set_xticks(xs); ax.set_xticklabels([f"$g_{i}$" for i in xs])
        ax.set_yscale("log"); ax.legend(frameon=False, loc="best")
        ax.set_ylabel("Entropy of targets (nats)")
        from matplotlib.ticker import FuncFormatter, LogLocator
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    axes[1].set_xlabel("Generation that produced the data")
    fig.tight_layout()
    fig.savefig(FIG / "mechanism.pdf", bbox_inches="tight"); plt.close(fig)


def fig_reliability(arms, plt):
    """Reliability diagrams at g0 and g5, items pooled over r100 seeds."""
    fig, axes = plt.subplots(2, 1, figsize=(3.1, 5.2))
    for j, lang in enumerate(("en", "bn")):
        ax = axes[j]
        sel = [a for a in arms if a["lang"] == lang and a["cond"] == "r100" and a["size"] == "1.7B"]
        ax.plot([0, 1], [0, 1], color="#8a8a86", lw=0.6, ls="--")
        if not sel:
            ax.text(0.5, 0.5, "corrected arms pending", ha="center", color="#52514e")
        for g, col, lab in ((0, "#8a8a86", "$g_0$"), (5, COL["r100"], "$g_5$")):
            P, Y = [], []
            # g0 is the same base model in every seed: one copy, not three.
            for a in (sel[:1] if g == 0 else sel):
                es = a["eval_set"]
                R = read_jsonl(RUNS / a["name"] / f"gen{g}" / f"eval_{es}.jsonl")
                P += [r["p_pred"] for r in R]; Y += [r["correct_scored"] for r in R]
            if not P:
                continue
            P, Y = np.array(P), np.array(Y, dtype=float)
            edges = np.linspace(0, 1, 11)
            idx = np.clip(np.digitize(P, edges[1:-1], right=True), 0, 9)
            cnt = np.bincount(idx, minlength=10)
            accb = np.array([Y[idx == b].mean() if cnt[b] else np.nan for b in range(10)])
            confb = np.array([P[idx == b].mean() if cnt[b] else np.nan for b in range(10)])
            m = cnt >= 5
            top = float((P > 0.9).mean())
            ax.plot(confb[m], accb[m], color=col, lw=1.2,
                    label=f"{lab}: ECE {ece_np(P, Y):.3f}, {top:.0%} of items at $p>0.9$")
            # marker area follows bin count, so sparse bins read as sparse
            ax.scatter(confb[m], accb[m], s=8 + 120 * cnt[m] / cnt.max(), facecolor="white",
                       edgecolor=col, linewidth=1.0, zorder=3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_title(LANG_NAME[lang] + ", 100% synthetic")
        ax.set_ylabel("Empirical accuracy")
        ax.legend(frameon=False, loc="upper left", fontsize=6)
    axes[1].set_xlabel("Confidence ($p_\\mathrm{pred}$, bin mean)")
    fig.tight_layout()
    fig.savefig(FIG / "reliability.pdf", bbox_inches="tight"); plt.close(fig)


def fig_scale(arms, plt):
    """1.7B vs 8B, English: ECE change from g0 and mean confidence, r100 and r50."""
    sizes = [("1.7B", "-"), ("8B", "--")]
    conds = ["r100", "r50", "r0"]
    if not any(a["size"] == "8B" for a in arms):
        return
    fig, axes = plt.subplots(2, 1, figsize=(3.1, 4.6), sharex=True)
    xs = np.arange(6)
    for size, ls in sizes:
        groups = by_cond(arms, "en", conds, size=size)
        for cond in conds:
            sel = groups.get(cond, [])
            if not sel:
                continue
            E = np.array([[g["ece"] for g in a["gens"]] for a in sel]); E = E - E[:, :1]
            C = np.array([[g["conf"] for g in a["gens"]] for a in sel])
            lab = f"{LABEL[cond]}, {size} (n={len(sel)})"
            axes[0].plot(xs, E.mean(0), color=COL[cond], ls=ls, marker=MARK[cond], ms=3.5, mfc="white", label=lab)
            axes[1].plot(xs, C.mean(0), color=COL[cond], ls=ls, marker=MARK[cond], ms=3.5, mfc="white", label=lab)
    for ax, yl in ((axes[0], "ECE change from $g_0$"), (axes[1], "Mean confidence ($p_\\mathrm{pred}$)")):
        ax.set_xticks(xs); ax.set_xticklabels([f"$g_{i}$" for i in xs]); ax.set_xlabel("Generation"); ax.set_ylabel(yl)
    axes[0].axhline(0, color="#8a8a86", lw=0.6)
    axes[0].set_xlabel("")
    axes[0].legend(frameon=False, loc="best", fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "scale.pdf", bbox_inches="tight"); plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-boot", action="store_true")
    ap.add_argument("--reuse", action="store_true",
                    help="re-render tables/figures from paper/results.json without recomputing")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

    found = discover()
    valid = [a for a in found if a["valid"]]
    excluded = [(a["name"], a["fingerprints"]) for a in found if not a["valid"]]
    print(f"study arms found: {len(found)}, valid: {len(valid)}, excluded (invalid fingerprint): {len(excluded)}")
    for n, f in excluded:
        print(f"   excluded {n} @ {'/'.join(f)}")

    arms = []
    cached = {}
    if args.reuse and (OUT / "results.json").exists():
        cached = {a["name"]: a for a in json.load(io.open(OUT / "results.json", encoding="utf-8"))["arms"]}
    for a in valid:
        if a["name"] in cached and "delta05" in cached[a["name"]]:
            # keep the cached analysis, refresh identity fields (e.g. a new key)
            c = cached[a["name"]]; c.update({k: v for k, v in a.items() if k != "summary"})
            arms.append(c); continue
        print(f"analysing {a['name']:32} {a['lang']} {a['cond']:10} seed {a['seed']}")
        arms.append(analyse_arm(a, do_boot=not args.no_boot))

    # ---- summary scalars the prose quotes
    def mean_delta(lang, size, cond, key):
        v = [a["delta05"][key]["delta"] for a in arms
             if a["lang"] == lang and a["size"] == size and a["cond"] == cond and "delta05" in a]
        return {"mean": float(np.mean(v)) if v else None, "n": len(v), "values": v}
    results = {
        "arms": arms,
        "excluded": excluded,
        "headline": {f"{lang}_{size}_{cond}_{key}": mean_delta(lang, size, cond, key)
                     for lang, size in GROUPS for cond in COND_ORDER
                     for key in ("ece", "acc", "auroc", "brier", "overconf")},
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")

    (TAB / "main.tex").write_text(table_main(arms), encoding="utf-8")
    (TAB / "secondary.tex").write_text(table_secondary(arms), encoding="utf-8")
    (TAB / "h1.tex").write_text(table_h1(arms), encoding="utf-8")
    (TAB / "sensitivity.tex").write_text(table_sensitivity(arms), encoding="utf-8")
    (TAB / "provenance.tex").write_text(table_provenance(arms), encoding="utf-8")

    plt = setup_mpl()
    fig_trajectories(arms, plt); fig_tradeoff(arms, plt); fig_controls(arms, plt)
    fig_mitigation(arms, plt); fig_mechanism(arms, plt); fig_reliability(arms, plt)
    fig_scale(arms, plt)
    print("wrote", OUT / "results.json", TAB, FIG)


if __name__ == "__main__":
    main()
