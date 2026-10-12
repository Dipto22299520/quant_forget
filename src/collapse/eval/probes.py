"""Linear/logistic confidence probes on the residual stream (H3).

The question H3 asks: does the *internal* correctness signal survive longer than
the *verbalised* one? So we need, at every generation, an honest cross-validated
estimate of how decodable correctness is from hidden states.

Two failure modes this code guards against:
  * Probe overfitting inflating AUROC late in the loop -> always cross-validate,
    never report train AUROC.
  * Class collapse (a late-generation model that is right/wrong almost always)
    making AUROC undefined -> return NaN explicitly rather than a fake 0.5.
"""
from __future__ import annotations

import numpy as np

from ..utils import LOG
from .calibration import auroc


def _make_clf(kind: str, C: float):
    from sklearn.linear_model import LogisticRegression, RidgeClassifier
    if kind == "linear":
        return RidgeClassifier(alpha=1.0 / max(C, 1e-6))
    return LogisticRegression(C=C, max_iter=2000, solver="lbfgs")


def _scores(clf, X):
    if hasattr(clf, "predict_proba"):
        return clf.predict_proba(X)[:, 1]
    return clf.decision_function(X)


def fit_probes(X: np.ndarray, y: np.ndarray, layers: list[int],
               kind: str = "logistic", C: float = 1.0, n_splits: int = 5,
               seed: int = 0) -> dict:
    """X: [N, L, H], y: [N] binary correctness.

    Returns per-layer cross-validated AUROC and out-of-fold probe scores for the
    best layer (those feed IVD).
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).astype(int)
    n = len(y)
    out: dict = {"n": n, "layers": layers, "per_layer": {}, "kind": kind}

    if n < 40 or len(np.unique(y)) < 2:
        out["error"] = (f"probe not fittable: n={n}, classes={np.unique(y).tolist()}. "
                        f"Need >=40 items and both classes present.")
        LOG.warning(out["error"])
        out["best_layer"] = None
        out["best_auroc"] = float("nan")
        out["oof_scores"] = None
        return out

    n_splits = int(min(n_splits, np.bincount(y).min()))
    if n_splits < 2:
        out["error"] = f"minority class too small for CV: {np.bincount(y).tolist()}"
        out["best_layer"] = None
        out["best_auroc"] = float("nan")
        out["oof_scores"] = None
        return out

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_layer, best_auc, best_oof = None, -np.inf, None

    for li, layer in enumerate(layers):
        Xl = X[:, li, :]
        oof = np.full(n, np.nan)
        for tr, te in skf.split(Xl, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(Xl[tr])
            Xte = sc.transform(Xl[te])
            clf = _make_clf(kind, C)
            clf.fit(Xtr, y[tr])
            oof[te] = _scores(clf, Xte)
        a = auroc(oof, y)
        out["per_layer"][int(layer)] = {"auroc": a}
        if not np.isnan(a) and a > best_auc:
            best_layer, best_auc, best_oof = int(layer), float(a), oof

    out["best_layer"] = best_layer
    out["best_auroc"] = float(best_auc) if best_layer is not None else float("nan")
    out["oof_scores"] = best_oof.tolist() if best_oof is not None else None
    LOG.info("probe: best layer %s auroc %.3f (n=%d, %d folds)",
             best_layer, out["best_auroc"], n, n_splits)
    return out


def fit_probes_nested(X: np.ndarray, y: np.ndarray, layers: list[int],
                      kind: str = "logistic", C: float = 1.0, n_splits: int = 5,
                      seed: int = 0) -> dict:
    """Layer selection INSIDE the cross-validation (audit finding #11).

    `fit_probes` picks the layer with the highest CV AUROC and then reports that
    same AUROC. Taking a maximum over ~8 correlated layers and reporting it as an
    out-of-sample estimate is selection bias: it is optimistic by construction,
    and the bias grows with the number of layers.

    Here the layer is chosen on inner folds and scored on a held-out outer fold,
    so the reported number is an honest estimate of "fit a probe, choosing the
    layer by CV". The biased per-layer numbers are still returned, clearly
    labelled, because they are useful for seeing WHERE correctness is decodable.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    y = np.asarray(y).astype(int)
    n = len(y)
    if n < 60 or len(np.unique(y)) < 2:
        return {"error": f"nested probe needs >=60 items and both classes; "
                         f"got n={n}, classes={np.unique(y).tolist()}",
                "auroc_nested": float("nan"), "n": n}

    minority = int(np.bincount(y).min())
    n_out = int(min(n_splits, minority))
    n_in = max(2, n_out - 1)
    # NESTED splitting needs the minority class to survive TWO levels of folding.
    # Checking only `minority >= 2` passes a 2-positives-in-600 case, and then an
    # inner training fold ends up single-class and sklearn raises mid-run.
    # Observed on ctrl_bn_model2_s1 gen1: correct_generated was 2/600.
    need = max(2 * n_in, 8)
    if n_out < 2 or minority < need:
        return {"error": (f"minority class too small for nested CV: "
                          f"{np.bincount(y).tolist()} (need >= {need} to survive "
                          f"{n_out} outer x {n_in} inner folds)"),
                "auroc_nested": float("nan"), "n": n,
                "class_counts": np.bincount(y).tolist()}

    outer = StratifiedKFold(n_splits=n_out, shuffle=True, random_state=seed)
    oof = np.full(n, np.nan)
    chosen = []

    for tr, te in outer.split(X[:, 0, :], y):
        # ---- inner: pick the layer using ONLY the outer-training data --------
        best_layer_i, best_inner = 0, -np.inf
        inner = StratifiedKFold(n_splits=n_in, shuffle=True, random_state=seed + 1)
        for li in range(len(layers)):
            Xl = X[tr][:, li, :]
            inner_oof = np.full(len(tr), np.nan)
            for itr, ite in inner.split(Xl, y[tr]):
                sc = StandardScaler()
                clf = _make_clf(kind, C)
                clf.fit(sc.fit_transform(Xl[itr]), y[tr][itr])
                inner_oof[ite] = _scores(clf, sc.transform(Xl[ite]))
            a = auroc(inner_oof, y[tr])
            if not np.isnan(a) and a > best_inner:
                best_inner, best_layer_i = a, li
        chosen.append(int(layers[best_layer_i]))

        # ---- outer: fit the chosen layer, score the untouched fold ----------
        sc = StandardScaler()
        clf = _make_clf(kind, C)
        clf.fit(sc.fit_transform(X[tr][:, best_layer_i, :]), y[tr])
        oof[te] = _scores(clf, sc.transform(X[te][:, best_layer_i, :]))

    return {"auroc_nested": float(auroc(oof, y)), "n": int(n),
            "layers_chosen_per_fold": chosen, "oof_scores": oof.tolist(),
            "n_outer": n_out, "n_inner": n_in}


def probe_scores_to_prob(scores: np.ndarray) -> np.ndarray:
    """Map probe scores to [0,1]. Logistic probes already are; ridge needs squashing."""
    s = np.asarray(scores, dtype=float)
    finite = s[np.isfinite(s)]
    if len(finite) == 0:
        return s
    if finite.min() >= 0.0 and finite.max() <= 1.0:
        return s
    return 1.0 / (1.0 + np.exp(-s))


def probe_report(model, tokenizer, records: list[dict], cfg) -> dict:
    """Extract hidden states for graded records and fit probes."""
    from ..modeling.hidden_states import extract_hidden_states

    recs = records[: cfg.probe.max_items]
    if not recs:
        return {"error": "no records"}
    # Hidden states are read at the last token of the GENERATED answer, so the
    # probe target must be whether that generated answer was right -- not the
    # letter-scored answer, which came from a different forward pass.
    y = np.array([int(r.get("correct_generated", r.get("correct", 0)) or 0)
                  for r in recs])
    X, layers = extract_hidden_states(
        model, tokenizer, recs, layer_stride=cfg.probe.layer_stride,
        batch_size=max(1, cfg.gen.batch_size // 2), max_len=cfg.model.max_len)
    if X.shape[0] != len(y):
        y = y[: X.shape[0]]
    # A probe failure must never kill a multi-generation arm. When the target is
    # almost all one class the probe is simply not estimable, and the honest
    # output is an error field, not a crash six hours into a run.
    try:
        res = fit_probes(X, y, layers, kind=cfg.probe.kind, C=cfg.probe.C,
                         n_splits=cfg.probe.n_splits, seed=cfg.seed)
    except Exception as e:
        LOG.warning("probe fit failed (%s: %s) -- reporting as not estimable",
                    type(e).__name__, e)
        return {"error": f"{type(e).__name__}: {e}", "best_auroc": float("nan"),
                "best_layer": None, "oof_scores": None, "n": int(len(y)),
                "class_counts": np.bincount(y).tolist() if len(y) else []}
    # The layer-selected AUROC is optimistically biased; keep it, but rename it
    # so it cannot be mistaken for an out-of-sample estimate.
    res["best_auroc_selected_biased"] = res.get("best_auroc")
    try:
        nested = fit_probes_nested(X, y, layers, kind=cfg.probe.kind, C=cfg.probe.C,
                                   n_splits=cfg.probe.n_splits, seed=cfg.seed)
    except Exception as e:
        LOG.warning("nested probe failed (%s: %s) -- keeping the selected estimate",
                    type(e).__name__, e)
        nested = {"error": f"{type(e).__name__}: {e}", "auroc_nested": float("nan")}
    res["nested"] = {k: v for k, v in nested.items() if k != "oof_scores"}
    # PRIMARY probe number = the nested, selection-corrected estimate.
    if np.isfinite(nested.get("auroc_nested", np.nan)):
        res["best_auroc"] = nested["auroc_nested"]
        res["oof_scores"] = nested["oof_scores"]
    res["uids"] = [r["uid"] for r in recs[: X.shape[0]]]
    return res
