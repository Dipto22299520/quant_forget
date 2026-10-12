"""Regression tests for the defects in docs/TECHNICAL_AUDIT.md.

Each test names the audit finding it locks down. These are the failures that
already produced confident, plausible, WRONG results once, so they get explicit
coverage rather than relying on the metric tests.

    python tests/test_audit_fixes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


# ---------------------------------------------------------------- finding #1
def test_ratio_fails_closed():
    print("\n[#1 synthetic ratio must not silently change]")
    from collapse.filters import RatioShortfall, build_sft_mix

    syn = [{"uid": f"s{i}"} for i in range(3000)]
    real = [{"uid": f"r{i}"} for i in range(20000)]

    # The exact historical bug: r100 asking for 20k pairs from a 3k prompt pool
    # trained on 15% synthetic and was reported as 100%.
    try:
        build_sft_mix(syn, real, ratio=1.0, total=20000)
        check("r100 shortfall raises instead of topping up", False)
    except RatioShortfall:
        check("r100 shortfall raises instead of topping up", True)

    s, r, info = build_sft_mix(syn, real, ratio=1.0, total=3000)
    check("honest r100 is all synthetic", len(s) == 3000 and len(r) == 0)
    check("r100 marked controlled", info["controlled_condition"] is True)

    s, r, info = build_sft_mix(syn, real, ratio=0.5, total=6000)
    check("r50 splits evenly", len(s) == 3000 and len(r) == 3000)
    check("realized ratio recorded", abs(info["effective_ratio"] - 0.5) < 1e-9)

    # Legacy mode must still mark the run as NOT a controlled condition.
    s, r, info = build_sft_mix(syn, real, ratio=1.0, total=20000, strict=False)
    check("legacy top-up reproduces the historical 0.150",
          abs(info["effective_ratio"] - 0.15) < 1e-9, f"got {info['effective_ratio']}")
    check("legacy top-up flagged uncontrolled",
          info["controlled_condition"] is False)


# ---------------------------------------------------------------- finding #2
def test_context_and_prompt_preserved():
    print("\n[#2 context must survive generation -> training]")
    from collapse.modeling.prompts import build_prompt

    ex = {"uid": "x1", "lang": "bn", "task_type": "open_qa",
          "question": "রাজধানী কোথায়?", "answers": ["ঢাকা"],
          "context": "বাংলাদেশের রাজধানী ঢাকা শহর।"}

    p_with = build_prompt(ex)
    check("context appears in the generation prompt", ex["context"] in p_with)

    # A record returned from generate_batch carries the exact prompt; rebuilding
    # must reuse it verbatim rather than re-rendering without context.
    rec = {**ex, "prompt": p_with, "response": "ঢাকা"}
    check("preserved prompt is reused verbatim", build_prompt(rec) == p_with)

    # The failure mode: a record that lost context re-renders WITHOUT it, so the
    # answer would be trained against a different prompt than it was generated
    # from. This is language-correlated because bn pools carry context and en
    # pools do not.
    lost = {k: v for k, v in ex.items() if k != "context"}
    check("dropping context really does change the prompt",
          build_prompt(lost) != p_with)


# ---------------------------------------------------------------- finding #3
def test_kept_token_span():
    print("\n[#3 token stats must follow the KEPT span, not the first N tokens]")
    import torch
    from transformers import AutoTokenizer

    from collapse.modeling.generate import _kept_token_mask, truncate_response

    tok = AutoTokenizer.from_pretrained("gpt2")

    def span(raw: str):
        ids = torch.tensor(tok.encode(raw))
        valid = torch.ones(len(ids), dtype=torch.bool)
        kept = truncate_response(raw)
        mask, ok = _kept_token_mask(tok, ids, valid, raw, kept)
        got = tok.decode(ids[mask]) if mask.any() else ""
        return kept, got.strip(), ok

    # PREFIX removal is the case the old code got wrong: it kept the FIRST n
    # tokens, i.e. the reasoning preamble, and reported its log probabilities as
    # if they were the answer's.
    kept, got, ok = span("<think>Let me recall the capital.</think>Paris is correct.")
    check("closed think block: kept text is the answer", kept == "Paris is correct.")
    check("closed think block: tokens map to the ANSWER, not the preamble",
          ok and "Paris" in got and "recall" not in got, f"got {got!r}")

    kept, got, ok = span("Answer: Berlin")
    check("echoed cue: kept text drops the cue", kept == "Berlin")
    check("echoed cue: tokens exclude the cue", ok and "Berlin" in got, f"got {got!r}")

    # SUFFIX removal still works.
    kept, got, ok = span("Paris.\nQuestion: what else?")
    check("suffix cut: kept text stops at the boundary", kept == "Paris.")
    check("suffix cut: tokens exclude the hallucinated turn",
          ok and "Question" not in got, f"got {got!r}")

    # An unclosed think block means there is NO answer -> no tokens at all.
    ids = torch.tensor(tok.encode("<think>Okay, I need to work out"))
    valid = torch.ones(len(ids), dtype=torch.bool)
    mask, ok = _kept_token_mask(tok, ids, valid, "<think>Okay, I need to work out", "")
    check("unclosed think block retains zero tokens", int(mask.sum()) == 0)

    # Untouched text keeps everything.
    kept, got, ok = span("It is definitely Paris.")
    check("clean answer keeps all its tokens", ok and got == "It is definitely Paris.")


def test_empty_answer_has_no_confidence():
    print("\n[#7 an empty answer must not carry confidence]")
    from collapse.eval.calibration import calibration_report

    # n_gen_tokens == 0 with None confidences: the signal must be absent, not
    # silently defaulted to something high.
    recs = [{"uid": f"u{i}", "correct_generated": 0, "correct": 0,
             "n_gen_tokens": 0, "mean_token_prob": None, "min_token_prob": None,
             "marker_conf": 0.85, "hedged": False, "refused": False,
             "n_markers": 0, "exact_match": 0, "token_f1": 0.0}
            for i in range(50)]
    rep = calibration_report(recs, n_boot=0)
    check("all-None internal signal is dropped, not defaulted",
          "mean_token_prob" not in rep["signals"])



# ---------------------------------------------------------------- finding #4
def test_inference_is_predeclared():
    print("\n[#4 onset must use CIs, Holm correction and a chance-relative floor]")
    import numpy as np

    from collapse.analysis.inference import (FLOOR_MARGIN, chance_level,
                                             h1_ordering, holm_bonferroni)

    def gen(n, acc, conf, seed):
        r = np.random.default_rng(seed)
        return [{"uid": f"u{i}", "task_type": "mcq", "choices": ["a", "b", "c", "d"],
                 "correct_scored": int(r.random() < acc),
                 "p_pred": float(np.clip(r.normal(conf, 0.08), 0, 1))}
                for i in range(400)]

    check("chance level derived from option count",
          abs(chance_level(gen(10, .5, .8, 0)) - 0.25) < 1e-9)

    # No movement must NOT produce an onset. The old code compared against the
    # g0 POINT estimate, so noise alone crossed the threshold.
    flat = [gen(400, 0.55, 0.80, 100 + g) for g in range(6)]
    r = h1_ordering(flat)
    check("no false onset under pure seed noise",
          r["calibration_onset"] is None and r["accuracy_onset"] is None,
          f"cal={r['calibration_onset']} acc={r['accuracy_onset']}")

    # True H1: calibration degrades at g2, accuracy at g4.
    confs = [0.80, 0.80, 0.90, 0.93, 0.95, 0.96]
    accs = [0.55, 0.55, 0.55, 0.55, 0.45, 0.40]
    runs = [gen(400, accs[g], confs[g], 200 + g) for g in range(6)]
    r = h1_ordering(runs)
    check("detects a real H1 ordering", r["h1_supported"] is True, r["verdict"])
    check("calibration onset precedes accuracy onset",
          r["calibration_onset"] < r["accuracy_onset"],
          f"{r['calibration_onset']} vs {r['accuracy_onset']}")

    # A four-option MCQ at 0.26 is AT CHANCE. The old absolute 0.10 floor let
    # this through and called H1 testable.
    floor = [gen(400, 0.26, 0.80, 300 + g) for g in range(6)]
    r = h1_ordering(floor)
    check("chance-relative floor blocks an untestable claim",
          r["testable"] is False, str(r.get("accuracy_g0")))
    check("floor is chance + margin, not an absolute 0.10",
          abs(r["floor"] - (0.25 + FLOOR_MARGIN)) < 1e-9, str(r.get("floor")))

    # Improvement must never be recorded as an onset.
    improving = [gen(400, 0.55, 0.90 - 0.05 * g, 400 + g) for g in range(6)]
    r = h1_ordering(improving)
    check("improving calibration is not an onset",
          r["calibration_onset"] is None, str(r["calibration_onset"]))

    h = holm_bonferroni({"a": 0.001, "b": 0.04, "c": 0.9})
    check("Holm rejects only the smallest here",
          h["a"]["significant"] and not h["b"]["significant"]
          and not h["c"]["significant"])


# ---------------------------------------------------------------- finding #15
def test_overlong_prompt_keeps_question():
    print("\n[#15 an over-long prompt must lose PASSAGE, never the question]")
    from transformers import AutoTokenizer

    from collapse.modeling.generate import truncate_response
    from collapse.modeling.loader import load_tokenizer
    from collapse.modeling.prompts import build_prompt, fit_prompt

    # Measured on squad_bn: 75% of prompts overflowed 768 tokens, the tokenizer
    # dropped the END (question + assistant marker), and 75% of Bangla synthetic
    # targets were passage continuation. English never overflowed. That is a
    # language-correlated corruption of the treatment, so it is locked down.
    tok = AutoTokenizer.from_pretrained("gpt2")
    passage = " ".join(f"sentence {i} of a long passage." for i in range(200))
    ex = {"uid": "x", "lang": "en", "task_type": "open_qa",
          "question": "What is the capital of France?", "answers": ["Paris"],
          "context": passage}
    full = build_prompt(ex, tokenizer=tok, use_chat_template=False)
    n_full = len(tok.encode(full))
    check("fixture really overflows the window", n_full > 200, f"n={n_full}")

    fitted, trimmed = fit_prompt(ex, tokenizer=tok, max_len=200,
                                 use_chat_template=False)
    n_fit = len(tok.encode(fitted))
    check("fitted prompt fits the window", n_fit <= 200, f"n={n_fit}")
    check("fit is reported", trimmed is True)
    check("question survives", "capital of France" in fitted)
    check("answer marker survives", fitted.rstrip().endswith("Answer:"), repr(fitted[-40:]))
    check("system prompt survives", fitted.startswith("You are a helpful assistant"))
    check("passage was what got shortened",
          "sentence 0 of" in fitted and "sentence 199 of" not in fitted)

    short, trimmed = fit_prompt({**ex, "context": "Paris is in France."},
                                tokenizer=tok, max_len=200, use_chat_template=False)
    check("short prompt is untouched", trimmed is False and "Paris is in France." in short)

    kept, trimmed = fit_prompt({**ex, "prompt": full}, tokenizer=tok, max_len=200,
                               use_chat_template=False)
    check("preserved generation prompt is never re-fitted", kept == full and trimmed is False)

    # Safety net for anything that still overflows: keep the END, where the
    # question and the assistant marker are.
    real = load_tokenizer("gpt2")
    check("tokenizer truncates from the left", real.truncation_side == "left")
    ids = real(full, truncation=True, max_length=50)["input_ids"]
    check("left truncation keeps the answer marker",
          real.decode(ids).rstrip().endswith("Answer:"))

    # A budget cut inside a multi-byte character decodes to U+FFFD; left in, it
    # becomes a token in the SFT target. Bangla hit this on 33% of targets.
    check("trailing partial character is dropped",
          truncate_response("ঢাকা শহ�") == "ঢাকা শহ")
    check("leading partial character is dropped",
          truncate_response("�াকা শহর") == "াকা শহর")
    check("a clean answer is untouched",
          truncate_response("ঢাকা শহর।") == "ঢাকা শহর।")



# ---------------------------------------------------------------- method screen
def test_method_screen_losses():
    print("\n[#16 method-screen losses, fresh anchor and quantile filter]")
    import torch

    from collapse.config import load_config, parse_overrides
    from collapse.filters import RatioShortfall, apply_filter, fresh_real_slice
    from collapse.modeling.train_sft import IGNORE, weighted_lm_loss

    torch.manual_seed(0)
    B, T, V = 2, 6, 11
    logits = torch.randn(B, T, V) * 3
    labels = torch.randint(0, V, (B, T))
    labels[:, :2] = IGNORE                      # prompt tokens are masked

    ce, _ = weighted_lm_loss(logits, labels, "ce")
    ls0, _ = weighted_lm_loss(logits, labels, "ls", ls_eps=0.0)
    en0, _ = weighted_lm_loss(logits, labels, "ent", ent_beta=0.0)
    check("label smoothing with eps=0 is plain CE", torch.allclose(ce, ls0, atol=1e-6))
    check("entropy bonus with beta=0 is plain CE", torch.allclose(ce, en0, atol=1e-6))

    sharp = torch.zeros(B, T, V); sharp[..., 0] = 12.0
    lab0 = torch.zeros(B, T, dtype=torch.long); lab0[:, :2] = IGNORE
    ce_s, _ = weighted_lm_loss(sharp, lab0, "ce")
    ls_s, _ = weighted_lm_loss(sharp, lab0, "ls", ls_eps=0.1)
    check("label smoothing penalises a confidently-right prediction", float(ls_s) > float(ce_s) + 0.1)

    def entropy_after_step(loss_type, **kw):
        x = logits.clone().requires_grad_(True)
        loss, _ = weighted_lm_loss(x, labels, loss_type, **kw)
        loss.backward()
        with torch.no_grad():
            y = x - 0.5 * x.grad
            lp = torch.log_softmax(y[:, :-1], -1)
            return float(-(lp.exp() * lp).sum(-1)[labels[:, 1:] != IGNORE].mean())
    check("entropy bonus raises predictive entropy more than CE does",
          entropy_after_step("ent", ent_beta=1.0) > entropy_after_step("ce"))

    cal, _ = weighted_lm_loss(logits, labels, "cal", cal_gamma=1.0)
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    tgt = labels[:, 1:].clone(); m = tgt != IGNORE; tgt[~m] = 0
    tok = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    w = (1 - tok.exp()) * m
    check("CAL is unchanged by the refactor",
          torch.allclose(cal, ((-tok) * w).sum() / w.sum(), atol=1e-6))

    pool = [{"uid": f"r{i}"} for i in range(700)]
    slices = [fresh_real_slice(pool, 150, g, seed=1)[0] for g in range(4)]
    uids = [set(x["uid"] for x in sl) for sl in slices]
    check("fresh anchor: every round gets n_real items", all(len(u) == 150 for u in uids))
    check("fresh anchor: no item is reused across rounds",
          all(uids[i].isdisjoint(uids[j]) for i in range(4) for j in range(i + 1, 4)))
    check("fresh anchor: deterministic for a seed",
          [x["uid"] for x in fresh_real_slice(pool, 150, 2, seed=1)[0]] == [x["uid"] for x in slices[2]])
    try:
        fresh_real_slice(pool, 150, 4, seed=1)          # needs [600, 750) of 700
        check("fresh anchor fails closed when the pool is exhausted", False)
    except RatioShortfall:
        check("fresh anchor fails closed when the pool is exhausted", True)

    recs = [{"uid": str(i), "mean_token_prob": i / 100, "correct_generated": 0} for i in range(100)]
    kept, rep = apply_filter(recs, kind="selfverify_top", threshold=0.5)
    check("top-50% self-verify keeps exactly the most confident half",
          len(kept) == 50 and min(r["mean_token_prob"] for r in kept) >= 0.5)

    cfg = load_config("configs/main_en.yaml", parse_overrides(
        ["loop.real_fresh=true", "train.loss=ent", "train.ent_beta=0.2", "train.ls_eps=0.05"]))
    check("new config fields parse with the right types",
          cfg.loop.real_fresh is True and cfg.train.loss == "ent"
          and abs(cfg.train.ent_beta - 0.2) < 1e-9 and abs(cfg.train.ls_eps - 0.05) < 1e-9)



# ---------------------------------------------------------------- sampler
def test_entropy_neutral_sampler():
    print("\n[#17 entropy-neutral sampler: truncated sampler carries the model's entropy]")
    import torch

    from collapse.modeling.generate import EntropyNeutralTopP

    torch.manual_seed(0)
    V = 500
    # rows from very peaked (top token p > 0.99) to flat
    z = torch.randn(6, V) * torch.tensor([[0.5], [1.0], [2.0], [4.0], [8.0], [1.0]])
    z[4, 0] += 12.0                                    # a confident step: top-p would be greedy
    proc = EntropyNeutralTopP(top_p=0.95)
    out = proc(None, z.clone())
    lp = torch.log_softmax(z, -1)
    H_model = -(lp.exp() * lp).sum(-1)
    q = torch.softmax(out, -1)
    H_q = -(q * torch.log(q.clamp_min(1e-30))).sum(-1)
    # Within 1%, not exactly: H(TopP(softmax(z/T))) jumps whenever a token enters
    # the nucleus, so the bisection can land on a discontinuity.
    check("sampler entropy matches model entropy on every row (within 1%)",
          bool(((H_q - H_model).abs() <= 0.01 * H_model + 1e-4).all()),
          f"{H_q.tolist()} vs {H_model.tolist()}")

    lp1 = torch.log_softmax(z, -1); p1 = lp1.exp()
    ps, _ = p1.sort(-1, descending=True)
    greedy_std = ((ps.cumsum(-1) - ps) < 0.95).sum(-1)
    check("fixture has a step that standard top-p samples greedily", int(greedy_std[4]) == 1)
    check("entropy-neutral sampler keeps more than one token there",
          int(torch.isfinite(out[4]).sum()) > 1)
    check("tokens outside the nucleus are impossible",
          bool((q[~torch.isfinite(out)] == 0).all()))
    check("temperature is only ever raised (T* >= 1)", proc.t_sum / proc.t_n >= 1.0)



# ---------------------------------------------------------------- K samples
def test_samples_per_prompt():
    print("\n[#18 K samples per prompt at a fixed budget]")
    from collapse.filters import expand_prompts
    pool = [{"uid": f"q{i}", "question": f"Q{i}"} for i in range(10)]
    check("K=1 returns the pool unchanged", expand_prompts(pool, 1) is pool)
    e = expand_prompts(pool, 4)
    check("K=4 keeps the budget within one K of the pool size", len(e) == 8)
    check("K=4 uses only the first len//K prompts", {x["question"] for x in e} == {"Q0", "Q1"})
    check("every copy has a unique uid", len({x["uid"] for x in e}) == len(e))
    check("the source records are not mutated", all("#" not in x["uid"] for x in pool))



def main():
    print("=" * 66)
    print("AUDIT REGRESSION TESTS")
    print("=" * 66)
    test_ratio_fails_closed()
    test_context_and_prompt_preserved()
    test_kept_token_span()
    test_empty_answer_has_no_confidence()
    test_inference_is_predeclared()
    test_overlong_prompt_keeps_question()
    test_method_screen_losses()
    test_entropy_neutral_sampler()
    test_samples_per_prompt()
    print("\n" + "=" * 66)
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
