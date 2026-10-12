"""Unit tests for the parts where a silent bug would corrupt the paper's claims.

Run:  python -m pytest tests -q     (or: python tests/test_units.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from collapse.analysis.matched import (degradation_rate, matched_baseline_subsample,
                                       normalize_curve)
from collapse.analysis.onset import onset_generation, ordering_test, onset_table
from collapse.eval.calibration import (adaptive_ece, auroc, brier, ece,
                                       overconfidence, calibration_report)
from collapse.eval.diversity import distinct_n, ngram_overconcentration, vocab_size
from collapse.eval.grade import (contains_match, exact_match, normalize_answer,
                                 parse_mcq_choice, token_f1)
from collapse.eval.ivd import internal_verbal_divergence
from collapse.filters import apply_filter, build_sft_mix, filter_quality
from collapse.markers import (extract_verbalized_confidence,
                              marker_implied_confidence)

FAILED = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILED.append(name)


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------- calibration
def test_calibration():
    print("\n[calibration]")
    # A perfectly calibrated signal: confidence == empirical accuracy in each bin.
    rng = np.random.default_rng(0)
    conf = rng.uniform(0, 1, 20000)
    correct = (rng.uniform(0, 1, 20000) < conf).astype(int)
    check("ECE ~0 for a calibrated signal", ece(conf, correct) < 0.02,
          f"got {ece(conf, correct):.4f}")

    # Maximally overconfident: always says 1.0, right half the time.
    conf2 = np.ones(1000)
    correct2 = np.zeros(1000)
    correct2[:500] = 1
    check("ECE = 0.5 when always-certain and 50% right",
          approx(ece(conf2, correct2), 0.5, 1e-9), f"got {ece(conf2, correct2)}")
    check("overconfidence is signed +0.5",
          approx(overconfidence(conf2, correct2), 0.5, 1e-9))

    # AUROC against a known-good reference.
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    check("AUROC matches hand-computed 0.75", approx(auroc(s, y), 0.75, 1e-9),
          f"got {auroc(s, y)}")
    check("AUROC = 1.0 for perfect ranking", approx(auroc(np.array([0., 1.]), np.array([0, 1])), 1.0))
    check("AUROC handles all-ties as 0.5",
          approx(auroc(np.array([.5, .5, .5, .5]), np.array([0, 1, 0, 1])), 0.5))
    check("AUROC is NaN with one class", np.isnan(auroc(np.array([.1, .9]), np.array([1, 1]))))

    check("Brier = 0 for a perfect predictor",
          approx(brier(np.array([1.0, 0.0]), np.array([1, 0])), 0.0))
    check("adaptive ECE ~0 for calibrated", adaptive_ece(conf, correct) < 0.02)


# --------------------------------------------------------------------- grading
def test_grading():
    print("\n[grading]")
    check("EN article stripping", normalize_answer("The Paris") == "paris")
    check("EN punctuation stripping", normalize_answer("Paris!") == "paris")
    check("Bangla digits normalised", normalize_answer("১৯৪৫") == "1945")
    check("Bangla dari stripped", normalize_answer("প্যারিস।") == "প্যারিস")
    check("answer prefix stripped (en)", normalize_answer("Answer: Paris") == "paris")
    check("answer prefix stripped (bn)", normalize_answer("উত্তর: প্যারিস") == "প্যারিস")

    check("exact match", exact_match("Paris", ["paris"]) == 1)
    check("exact match rejects a hedged sentence",
          exact_match("I think it is Paris", ["Paris"]) == 0)
    check("contains_match accepts a hedged sentence",
          contains_match("I think it is Paris", ["Paris"]) == 1)
    check("contains_match rejects a wrong answer",
          contains_match("Maybe it is Berlin", ["Paris"]) == 0)
    check("token F1 partial credit",
          0.0 < token_f1("William Shakespeare the poet", ["William Shakespeare"]) < 1.0)

    # Bangla suffix stripping should let an inflected form match its lemma.
    check("Bangla suffix stripping helps matching",
          contains_match("রাজধানী ঢাকায় অবস্থিত", ["ঢাকা"]) == 1,
          f"norm={normalize_answer('রাজধানী ঢাকায় অবস্থিত')!r}")

    check("MCQ letter parsed", parse_mcq_choice("B. Jupiter", ["Mars", "Jupiter"]) == 1)
    check("MCQ text parsed", parse_mcq_choice("Jupiter", ["Mars", "Jupiter"]) == 1)


# --------------------------------------------------------------------- markers
def test_markers():
    print("\n[markers]")
    hi = marker_implied_confidence("It is definitely Paris.", "en")
    lo = marker_implied_confidence("I think it might be Paris.", "en")
    bare = marker_implied_confidence("Paris.", "en")
    check("strengthener -> high confidence", hi["marker_conf"] > 0.9)
    check("strengthener is not flagged as hedged", hi["hedged"] is False)
    check("weakener -> low confidence", lo["marker_conf"] < 0.5)
    check("weakener flagged as hedged", lo["hedged"] is True)
    check("bare assertion -> default 0.85", approx(bare["marker_conf"], 0.85))
    check("bare assertion is not hedged", bare["hedged"] is False)

    # The negation trap: "not sure" must never score as "sure".
    neg = marker_implied_confidence("I'm not sure, but Paris.", "en")
    check("'not sure' beats 'sure' (longest match)", neg["marker_conf"] < 0.4,
          f"got {neg['marker_conf']} markers={neg['markers']}")

    bn_hi = marker_implied_confidence("এটি নিশ্চিতভাবে প্যারিস।", "bn")
    bn_lo = marker_implied_confidence("হয়তো এটি প্যারিস।", "bn")
    bn_neg = marker_implied_confidence("আমি নিশ্চিত নই, তবে প্যারিস।", "bn")
    check("bn strengthener high", bn_hi["marker_conf"] > 0.9, f"got {bn_hi}")
    check("bn weakener low", bn_lo["marker_conf"] < 0.5, f"got {bn_lo}")
    check("bn negation trap handled", bn_neg["marker_conf"] < 0.4,
          f"got {bn_neg['marker_conf']} markers={bn_neg['markers']}")
    check("bn lexicon is independent of en",
          marker_implied_confidence("definitely", "bn")["n_markers"] == 0)

    # Refusals must not be scored as confident bare assertions.
    ref = marker_implied_confidence("I'm sorry, but I cannot answer this question.", "en")
    check("refusal detected", ref["refused"] is True)
    check("refusal gets low implied confidence, not the 0.85 bare default",
          ref["marker_conf"] < 0.2, f"got {ref['marker_conf']}")
    check("refusal is not counted as a hedge", ref["hedged"] is False)
    check("plain answer is not a refusal",
          marker_implied_confidence("Paris.", "en")["refused"] is False)
    check("'as an AI language model' is a refusal",
          marker_implied_confidence("As an AI language model, I lack data.", "en")["refused"] is True)
    bn_ref = marker_implied_confidence("আমার কাছে এই বিষয়ে কোনো তথ্য নেই।", "bn")
    check("bn refusal detected", bn_ref["refused"] is True, f"got {bn_ref}")

    check("verbalized % parsed", approx(extract_verbalized_confidence("Paris. Confidence: 80%"), 0.8))
    check("verbalized decimal parsed", approx(extract_verbalized_confidence("Confidence: 0.8"), 0.8))
    check("verbalized bangla parsed",
          approx(extract_verbalized_confidence("প্যারিস। আত্মবিশ্বাস: ৮০%"), 0.8))
    check("no verbalized -> None", extract_verbalized_confidence("Paris.") is None)


# ----------------------------------------------------------------------- onset
def test_onset():
    print("\n[onset]")
    # Accuracy falls at g3; ECE rises at g1 -> H1 supported.
    curves = {
        "accuracy":   [0.50, 0.50, 0.50, 0.40, 0.35],
        "ece_marker": [0.10, 0.20, 0.30, 0.40, 0.50],
    }
    check("accuracy onset detected at g3",
          onset_generation(curves["accuracy"], "accuracy", g0_ci=(0.48, 0.52)) == 3)
    check("ECE onset detected at g1",
          onset_generation(curves["ece_marker"], "ece_marker", g0_ci=(0.08, 0.12)) == 1)

    t = onset_table(curves, cis={"accuracy": (0.48, 0.52), "ece_marker": (0.08, 0.12)})
    res = ordering_test(t)
    check("H1 supported when calibration moves first", res["h1_supported"] is True,
          res["verdict"])
    check("gap computed", res["gap"] == 2)

    # Direction sanity: a RISING accuracy must not count as degradation.
    check("rising accuracy is not an onset",
          onset_generation([0.5, 0.6, 0.7], "accuracy", g0_ci=(0.48, 0.52)) is None)
    # A FALLING ECE must not count either.
    check("falling ECE is not an onset",
          onset_generation([0.5, 0.4, 0.3], "ece_marker", g0_ci=(0.48, 0.52)) is None)
    check("flat curve has no onset",
          onset_generation([0.5, 0.5, 0.5], "accuracy", g0_ci=(0.48, 0.52)) is None)

    # Unknown metric must raise, not guess.
    try:
        onset_generation([1, 2], "not_a_real_metric")
        check("unknown metric raises", False)
    except KeyError:
        check("unknown metric raises", True)


# --------------------------------------------------------------------- matched
def test_matched():
    print("\n[matched baseline]")
    strong = [{"uid": f"a{i}", "correct": int(i < 80)} for i in range(100)]  # 0.80
    weak = [{"uid": f"b{i}", "correct": int(i < 40)} for i in range(100)]    # 0.40
    m = matched_baseline_subsample(strong, weak, tol=0.02, seed=0)
    check("subsampling matched the accuracies", m["matched"] is True,
          f"achieved {m.get('achieved_accuracy')}")
    check("dropped only correct items", m["dropped"] > 0)
    check("weak set untouched", m["n_b"] == 100)

    check("normalise divides by g0",
          normalize_curve([0.5, 0.25])[1] == 0.5)
    check("degradation slope is negative for a falling curve",
          degradation_rate([1.0, 0.8, 0.6]) < 0)


# --------------------------------------------------------------------- filters
def test_filters():
    print("\n[filters]")
    recs = [{"uid": f"u{i}", "correct": int(i % 2 == 0),
             "mean_token_prob": 0.9 if i % 2 == 0 else 0.1,
             "marker_conf": 0.85} for i in range(100)]
    kept, q = apply_filter(recs, "selfverify", threshold=0.5)
    check("perfect filter keeps exactly the correct half", len(kept) == 50)
    check("precision 1.0 for a perfect signal", approx(q["precision"], 1.0))
    check("recall 1.0 for a perfect signal", approx(q["recall"], 1.0))
    check("lift = precision - base rate", approx(q["lift"], 0.5))

    # A useless signal must show zero lift -- this is the H4 null.
    recs2 = [{"uid": f"u{i}", "correct": int(i % 2 == 0), "mean_token_prob": 0.9}
             for i in range(100)]
    _, q2 = apply_filter(recs2, "selfverify", threshold=0.5)
    check("uninformative filter has ~0 lift", approx(q2["lift"], 0.0, 1e-9),
          f"got {q2['lift']}")

    none_kept, q3 = apply_filter(recs, "none")
    check("filter=none keeps everything", len(none_kept) == 100)

    syn = [{"uid": f"s{i}"} for i in range(100)]
    real = [{"uid": f"r{i}"} for i in range(100)]
    s, r, info = build_sft_mix(syn, real, ratio=0.5, total=40, seed=0)
    check("50:50 mix splits evenly", len(s) == 20 and len(r) == 20)
    check("effective ratio reported", approx(info["effective_ratio"], 0.5))
    s2, r2, info2 = build_sft_mix(syn, real, ratio=1.0, total=40, seed=0)
    check("ratio 1.0 -> all synthetic", len(s2) == 40 and len(r2) == 0)


# ------------------------------------------------------------------------- IVD
def test_ivd():
    print("\n[IVD]")
    # Internal signal tracks correctness; verbal signal is constant (decoupled).
    recs = [{"uid": f"u{i}", "correct": int(i % 2 == 0),
             "mean_token_prob": 0.9 if i % 2 == 0 else 0.2,
             "marker_conf": 0.85} for i in range(200)]
    r = internal_verbal_divergence(recs)
    check("internal AUROC high", r["auroc_internal"] > 0.99, f"got {r['auroc_internal']}")
    # A constant verbalised signal carries no discriminative information, so the
    # all-ties Mann-Whitney value 0.5 is correct (chance), NOT NaN. This is the
    # exact signature of a collapsed model that has stopped hedging entirely, so
    # it must read as "chance-level discrimination" rather than "missing".
    check("verbal AUROC is chance (0.5) when constant",
          approx(r["auroc_verbal"], 0.5, 1e-9), f"got {r['auroc_verbal']}")
    check("IVD AUROC gap is large when verbal signal is dead",
          r["ivd_auroc_gap"] > 0.45, f"got {r['ivd_auroc_gap']}")
    check("IVD L1 computed", r["ivd_l1"] > 0)

    # Both informative -> small AUROC gap.
    recs2 = [{"uid": f"u{i}", "correct": int(i % 2 == 0),
              "mean_token_prob": 0.9 if i % 2 == 0 else 0.2,
              "marker_conf": 0.9 if i % 2 == 0 else 0.2} for i in range(200)]
    r2 = internal_verbal_divergence(recs2)
    check("AUROC gap ~0 when both signals track correctness",
          abs(r2["ivd_auroc_gap"]) < 1e-6, f"got {r2['ivd_auroc_gap']}")
    check("IVD L1 ~0 when signals agree", r2["ivd_l1"] < 1e-6)


# ------------------------------------------------------------------- diversity
def test_truncation():
    print("\n[response truncation]")
    from collapse.modeling.generate import truncate_response

    check("cuts hallucinated system turn",
          truncate_response("Paris.\nsystem\nYou are a helpful assistant.") == "Paris.")
    check("cuts next fabricated question",
          truncate_response("Paris.\nQuestion: What is X?") == "Paris.")
    check("cuts chat special tokens",
          truncate_response("Paris.<|im_end|>") == "Paris.")
    check("cuts bangla question marker",
          truncate_response("প্যারিস।\nপ্রশ্ন: কী?") == "প্যারিস।")
    check("leaves a clean answer untouched",
          truncate_response("It is definitely Paris.") == "It is definitely Paris.")
    check("empty stays empty", truncate_response("") == "")
    # A hedged multi-sentence answer must survive intact -- over-eager truncation
    # would delete the very markers H1 measures.
    long_ans = "I think it might be Paris, but I am not certain about that."
    check("hedged sentence survives", truncate_response(long_ans) == long_ans)


def test_diversity():
    print("\n[diversity]")
    varied = ["the cat sat on the mat", "a dog ran through the park",
              "birds fly over tall mountains"]
    same = ["the cat sat on the mat"] * 3
    check("distinct-2 higher for varied text",
          distinct_n(varied, 2) > distinct_n(same, 2))
    check("vocab size higher for varied text", vocab_size(varied) > vocab_size(same))
    check("n-gram concentration higher for repeated text",
          ngram_overconcentration(same, 3, 10) > ngram_overconcentration(varied, 3, 10))


# ---------------------------------------------------------------------- report
def test_report():
    print("\n[calibration_report]")
    recs = [{"uid": f"u{i}", "correct": int(i % 2 == 0), "exact_match": int(i % 2 == 0),
             "token_f1": 1.0 * (i % 2 == 0), "n_gen_tokens": 10,
             "mean_token_prob": 0.9, "marker_conf": 0.85,
             "hedged": i % 4 == 0, "n_markers": 1, "verbalized_conf": None}
            for i in range(200)]
    rep = calibration_report(recs, n_boot=0)
    check("accuracy computed", approx(rep["accuracy"], 0.5))
    check("hedge rate computed", approx(rep["hedge_rate"], 0.25))
    check("internal signal present", "mean_token_prob" in rep["signals"])
    check("all-None signal is dropped", "verbalized_conf" not in rep["signals"])
    check("overconfidence is +0.4 for conf .9 / acc .5",
          approx(rep["signals"]["mean_token_prob"]["overconfidence"], 0.4, 1e-9))



def test_format_independent_accuracy():
    print("\n[format-independent accuracy]")
    from collapse.eval.calibration import calibration_report, SIGNAL_TARGET

    # The exact confound this exists to catch: a model that KNOWS the answers
    # (letter-scored accuracy 0.60) but cannot emit a parseable one yet
    # (generated accuracy 0.02). A format-only improvement must not read as a
    # knowledge gain.
    recs = []
    for i in range(200):
        knows = i < 120                     # scored correct 60% of the time
        recs.append({
            "uid": f"u{i}",
            "correct_generated": int(i < 4),      # 2% -- format failure
            "correct_scored": int(knows),        # 60% -- real knowledge
            "correct": int(knows),
            "p_correct": 0.62 if knows else 0.38,
            "marker_conf": 0.85,
            "mean_token_prob": 0.9,
            "hedged": False, "refused": False, "n_markers": 0,
            "exact_match": 0, "token_f1": 0.0, "n_gen_tokens": 10,
        })
    rep = calibration_report(recs, n_boot=0)
    check("primary accuracy uses letter scoring for MCQ",
          approx(rep["accuracy"], 0.60), f"got {rep['accuracy']}")
    check("generated accuracy reported separately",
          approx(rep["accuracy_generated"], 0.02), f"got {rep['accuracy_generated']}")
    check("scored accuracy reported separately",
          approx(rep["accuracy_scored"], 0.60))
    check("format_gap exposes the format failure",
          approx(rep["format_gap"], 0.58), f"got {rep['format_gap']}")

    # Signal->target pairing: marker_conf describes the GENERATED text.
    check("marker_conf calibrated against generated correctness",
          rep["signals"]["marker_conf"]["calibrated_against"] == "correct_generated")
    check("p_correct calibrated against scored correctness",
          rep["signals"]["p_correct"]["calibrated_against"] == "correct_scored")

    # marker says 0.85 while the generated text is right 2% of the time
    check("marker overconfidence measured vs generated text",
          approx(rep["signals"]["marker_conf"]["overconfidence"], 0.83, 1e-6),
          f"got {rep['signals']['marker_conf']['overconfidence']}")
    # p_correct averages 0.524 vs scored accuracy 0.60 -> slightly UNDER-confident
    check("p_correct overconfidence measured vs scored correctness",
          rep["signals"]["p_correct"]["overconfidence"] < 0,
          f"got {rep['signals']['p_correct']['overconfidence']}")

    # Open QA (no scored field) must still work and fall back cleanly.
    open_recs = [{"uid": f"o{i}", "correct_generated": int(i % 2 == 0),
                  "correct": int(i % 2 == 0), "marker_conf": 0.85,
                  "mean_token_prob": 0.9, "hedged": False, "refused": False,
                  "n_markers": 0, "exact_match": 0, "token_f1": 0.0,
                  "n_gen_tokens": 10} for i in range(100)]
    rep2 = calibration_report(open_recs, n_boot=0)
    check("open QA: scored accuracy is None", rep2["accuracy_scored"] is None)
    check("open QA: format_gap is None", rep2["format_gap"] is None)
    check("open QA: accuracy still computed", approx(rep2["accuracy"], 0.5))


def test_onset_format_family():
    print("\n[onset uses format-independent accuracy]")
    from collapse.analysis.onset import ACCURACY_METRICS, ordering_test, onset_table

    check("accuracy_scored leads the accuracy family",
          ACCURACY_METRICS[0] == "accuracy_scored")

    # Format acquisition: generated accuracy rockets, scored accuracy decays.
    # H1 must be judged on the scored curve.
    curves = {
        "accuracy_generated": [0.02, 0.19, 0.24, 0.24],   # format learning
        "accuracy_scored":    [0.60, 0.59, 0.52, 0.48],   # real decay at g2
        "ece_marker":         [0.10, 0.22, 0.30, 0.40],   # calibration at g1
    }
    cis = {"accuracy_scored": (0.57, 0.63), "ece_marker": (0.08, 0.12),
           "accuracy_generated": (0.00, 0.04)}
    t = onset_table(curves, cis=cis)
    check("scored accuracy onset detected at g2", t["accuracy_scored"]["onset"] == 2,
          f"got {t['accuracy_scored']['onset']}")
    res = ordering_test(t)
    check("H1 judged on the format-independent curve",
          res["earliest_accuracy"]["metric"] == "accuracy_scored",
          f"got {res['earliest_accuracy']}")
    check("H1 supported (calibration g1 < accuracy g2)", res["h1_supported"] is True,
          res["verdict"])



def test_reasoning_strip():
    print("\n[reasoning-model handling]")
    from collapse.modeling.generate import strip_reasoning, truncate_response

    # Closed block: drop the thinking, keep the answer.
    closed = "<think>Let me recall. Paris is in France.</think>Paris."
    check("closed think block removed, answer kept",
          strip_reasoning(closed) == "Paris.", repr(strip_reasoning(closed)))

    # UNCLOSED block: the budget ran out mid-thought, so there is NO answer.
    # Returning the reasoning text would make every metric score the model's
    # musings instead of its answer -- exactly the bug that flattened the pilot.
    open_blk = "<think>Okay, the user is asking about George Rankin. Let me think"
    check("unclosed think block yields empty, not the reasoning text",
          strip_reasoning(open_blk) == "", repr(strip_reasoning(open_blk)))
    check("truncate_response also yields empty for unclosed reasoning",
          truncate_response(open_blk) == "")

    check("text without reasoning is untouched",
          strip_reasoning("It is definitely Paris.") == "It is definitely Paris.")
    check("empty stays empty", strip_reasoning("") == "")
    check("case-insensitive tag", strip_reasoning("<THINK>x</THINK>Paris") == "Paris")

    # A hedge INSIDE reasoning must not be counted; only the answer counts.
    from collapse.eval.grade import grade_record
    rec = {"uid": "t1", "lang": "en", "task_type": "open_qa",
           "answers": ["Paris"], "response": truncate_response(
               "<think>I am not sure, maybe Berlin?</think>It is Paris.")}
    g = grade_record(rec)
    check("hedge inside reasoning is not counted as hedging",
          g["hedged"] is False, f"markers={g['markers']}")
    check("answer after reasoning still grades correct", g["correct"] == 1)


def main():
    print("=" * 66)
    print("UNIT TESTS -- metric correctness")
    print("=" * 66)
    test_calibration()
    test_grading()
    test_markers()
    test_onset()
    test_matched()
    test_filters()
    test_ivd()
    test_truncation()
    test_reasoning_strip()
    test_diversity()
    test_report()
    test_format_independent_accuracy()
    test_onset_format_family()
    print("\n" + "=" * 66)
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        return 1
    print("ALL PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
