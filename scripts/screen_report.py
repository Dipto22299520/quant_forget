r"""Score the Stage M method screen against its pre-declared rule.

    .venv\Scripts\python.exe scripts\screen_report.py

Rule (declared in scripts/run_all.ps1 before any Stage M arm ran):
    promote a method if its g5 AUROC beats the CE-baseline mean by >= 0.03
    and its g5 accuracy is >= the CE-baseline mean - 0.01.

Why AUROC: scripts/label_free.py shows calibration drift is a temperature shift
that is removable post hoc without labels, so a training-time method must earn
its place on discrimination. Raw and label-free-tracked ECE are reported too.

Each method arm is also compared with the CE arm of the SAME seed on the same
600 items (paired item bootstrap): same base model, same g0, same first
synthetic set, so the difference is the method.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8")

from collapse.utils import read_jsonl                                 # noqa: E402
from label_free import ece_at, fit_t, log_conf, match_t, mean_entropy, round_logits  # noqa: E402
from paper_assets import auroc_np, discover, ece_np                   # noqa: E402

MARGIN_AUROC = 0.03
MARGIN_ACC = 0.01
N_BOOT = 2000


def endpoint(arm, g):
    L, y, uids, exact = round_logits(arm, g)
    lc = log_conf(L)                         # rank in log space: no saturation ties
    corr = (L.argmax(1) == y).astype(float)
    return {"L": L, "y": y, "uids": uids, "conf": lc, "corr": corr, "exact": exact,
            "auroc": auroc_np(lc, corr), "acc": float(corr.mean()),
            "ece": ece_np(np.exp(lc), corr)}


def tracked_ece(arm):
    """Label-free tracked ECE at g5 (two-fold cross-fitted, as in label_free.py)."""
    L0, y0, _, _ = round_logits(arm, 0)
    L5, y5, _, _ = round_logits(arm, 5)
    n = len(y0); perm = np.random.default_rng(0).permutation(n)
    out = 0.0
    for fi, ev in ((perm[: n // 2], perm[n // 2:]), (perm[n // 2:], perm[: n // 2])):
        T0 = fit_t(L0[fi], y0[fi])
        out += ece_at(L5[ev], y5[ev], match_t(L5, mean_entropy(L0, T0))) / 2
    return out


def paired_auroc_diff(a, b, seed=0):
    """AUROC(a) - AUROC(b) on shared items, item bootstrap."""
    assert a["uids"] == b["uids"]
    point = a["auroc"] - b["auroc"]
    rng = np.random.default_rng(seed); n = len(a["corr"]); boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        va, vb = auroc_np(a["conf"][i], a["corr"][i]), auroc_np(b["conf"][i], b["corr"][i])
        if np.isfinite(va) and np.isfinite(vb):
            boots.append(va - vb)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def main():
    arms = [a for a in discover() if a["valid"] and a["lang"] == "en" and a["size"] == "1.7B"
            and a["model"] == "Qwen/Qwen3-1.7B"]
    ce = sorted([a for a in arms if a["cond"] == "r100"], key=lambda a: a["seed"])
    ce_end = {a["seed"]: endpoint(a, 5) for a in ce}
    base_auroc = float(np.mean([e["auroc"] for e in ce_end.values()]))
    base_acc = float(np.mean([e["acc"] for e in ce_end.values()]))
    bar_auroc, bar_acc = base_auroc + MARGIN_AUROC, base_acc - MARGIN_ACC
    print(f"CE baseline (n={len(ce)} seeds): g5 AUROC {base_auroc:.3f}, accuracy {base_acc:.3f}")
    print(f"Promotion bar: AUROC >= {bar_auroc:.3f} and accuracy >= {bar_acc:.3f}\n")
    order = ["r100", "ensampler", "ancestral", "epochs1", "e1anc", "k2anc", "k4anc", "k4", "cal", "cal_g3", "cal_anchor", "tce90", "tce",
             "selfverify", "selfverify_top", "ls", "ent", "r75", "r50", "fullft"]
    print(f"{'arm':22} {'AUROC g0->g5':>14} {'acc g5':>7} {'ECE raw':>8} {'ECE tracked':>12}"
          f"   {'dAUROC vs CE same seed [95% CI]':>34}  verdict")
    for cond in order:
        for a in sorted([a for a in arms if a["cond"] == cond], key=lambda a: a["seed"]):
            e0, e5 = endpoint(a, 0), endpoint(a, 5)
            tr = tracked_ece(a)
            if cond == "r100":
                cmp_txt, verdict = "baseline", ""
            else:
                ref = ce_end.get(a["seed"])
                if ref is not None:
                    d, lo, hi = paired_auroc_diff(e5, ref)
                    cmp_txt = f"{d:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                else:
                    cmp_txt = "no same-seed CE arm"
                is_method = cond not in ("r75", "r50", "fullft", "ancestral", "epochs1", "selfverify")
                ok = e5["auroc"] >= bar_auroc and e5["acc"] >= bar_acc
                verdict = ("PROMOTE" if ok else "stop") if is_method else "(reference)"
            flag = "" if (e0["exact"] and e5["exact"]) else "  [SATURATED: re-score g0/g5]"
            print(f"{a['name']:22} {e0['auroc']:.3f}->{e5['auroc']:.3f}  {e5['acc']:7.3f} {e5['ece']:8.3f}"
                  f" {tr:12.3f}   {cmp_txt:>34}  {verdict}{flag}")


if __name__ == "__main__":
    main()
