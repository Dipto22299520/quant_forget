# Datasets — status

Every id below was checked against the Hub API on **18 Aug 2026**, and everything
marked DOWNLOADED was actually fetched, mapped, and row-counted on this machine.

## ✅ Already on disk — you do not need to download these

They are in `data/raw/<key>.jsonl`:

| key | rows | key | rows |
|---|---|---|---|
| `triviaqa` | 138,384 | `squad_bn` | 68,671 |
| `nq_open` | 87,925 | `bnmmlu` | 14,750 |
| `dolly` | 15,011 | `mmmlu_bn` | 14,042 |
| `popqa` | 14,267 | `global_mmlu_bn` | 14,042 |
| `mmlu_en` | 14,042 | `banglarqa` | 7,990 |
| `global_mmlu_en` | 14,042 | | |

Every one mapped with **0 rows dropped** except `squad_bn` (see §3) and
`banglarqa` (see §3).

## ⚠️ What you actually need to get — three items, all manual

Jump to §5. **BLUCK**, **BEnQA**, and **BanglaMedQA**. Nothing else is
outstanding.

---

## 0. Re-fetching

```powershell
cd "c:\Users\User\Desktop\Qantization forgets"
$env:PYTHONPATH="src"
.\.venv\Scripts\python.exe scripts\download_data.py          # skips existing
.\.venv\Scripts\python.exe -m collapse.cli check-data        # reachability only
```

---

## 1. English — training pools

| key | HF id | status | licence | why |
|---|---|---|---|---|
| `triviaqa` | `mandarjoshi/trivia_qa` (config `rc.nocontext`) | VERIFIED | Apache-2.0 | Rich alias lists → lenient EM works properly |
| `nq_open` | `google-research-datasets/nq_open` | VERIFIED | CC BY-SA 3.0 | Standard open-QA |
| `dolly` | `databricks/databricks-dolly-15k` | VERIFIED | CC BY-SA 3.0 | **Human-written** — this is the "real" half of the SFT mix |
| `alpaca` | `tatsu-lab/alpaca` | VERIFIED | CC BY-NC 4.0 | ⚠️ **Alpaca is itself synthetic** (self-instruct). Do **not** use it as real anchoring data — it would contaminate generation 0. Registered only for a deliberately-contaminated control arm. |

## 2. English — evaluation (never trained on)

| key | HF id | status | why |
|---|---|---|---|
| `popqa` | `akariasai/PopQA` | VERIFIED (MIT) | Carries `s_pop` popularity → **tail-knowledge buckets**, your direct link to collapse theory. Only a `test` split exists upstream. |
| `mmlu_en` | `cais/mmlu` (config `all`) | VERIFIED (MIT) | English side of the parallel-knowledge axis |

## 3. Bangla — training pools

| key | HF id | status | licence | note |
|---|---|---|---|---|
| `squad_bn` | `csebuetnlp/squad_bn` | DOWNLOADED 68,671 | CC BY-NC-SA 4.0 | **NC licence** — fine for research, check before releasing derived annotations. Needed the parquet-branch fallback (§6). **42% of rows dropped** (118,117 → 68,671) for empty answer spans — do not quote the upstream count in the paper. |
| `banglarqa` | `sartajekram/BanglaRQA` | DOWNLOADED 7,990 | — | 3,922 dropped: unanswerable items, **plus all `confirmation` (yes/no) questions**. Those are coin-flips that would inflate baseline accuracy and make confidence meaningless. `is_answerable` ships as the string `'1'`, not a bool — the mapper handles it. |
| `indicqa_bn` | `ai4bharat/IndicQA` | ❌ BROKEN | — | Loading script unsupported in `datasets` 5, and the auto-parquet branch collapses all languages into one `default` config. Not worth fixing — `squad_bn` alone gives 68k. Left registered as a documented dead end. |

## 4. Bangla — evaluation (never trained on)

| key | HF id | status | note |
|---|---|---|---|
| `mmmlu_bn` | `openai/MMMLU` (config `BN_BD`) | DOWNLOADED 14,042 | Professionally translated MMLU; pairs with `mmlu_en` |
| `bnmmlu` | `hishab/bangla-mmlu` | DOWNLOADED 14,750 | ⚠️ The id in the plan (`BanglaLLM/bangla-mmlu`) **does not exist**. This is the real one. |
| `global_mmlu_bn` | `CohereLabs/Global-MMLU` (config `bn`) | DOWNLOADED 14,042 | **Better than the plan expected — see below** |
| `mmlu_bn_alt` | `rishiraj/MMLU-BN` | VERIFIED | Backup, not downloaded |
| `flores_bn` | `openlanguagedata/flores_plus` | ⚠️ **GATED** | Needs `huggingface-cli login` + accepting terms. Only used for held-out perplexity, so it is optional. Ungated substitute: hold out a slice of `squad_bn` contexts. |

### Global-MMLU is worth more than its row count suggests

`global_mmlu_bn` and `global_mmlu_en` are **both exactly 14,042 rows — the same
items in both languages.** That gives you a second item-matched parallel
instrument for H2, independent of BEnQA, which means the matched-baseline
argument does not rest on a single dataset.

It also carries `cultural_sensitivity_label` (CS = culturally sensitive, CA =
culturally agnostic), preserved into `meta.cultural_sensitivity`. That splits the
eval along exactly the axis the tail-knowledge claim needs: an English-centric
generator should damage CS items faster than CA items. That is a clean,
pre-registered-looking prediction you can state before running it.

---

## 5. ⚠️ The two that need manual work — start these first

These are the two most important Bangla items and **neither is on Hugging Face.**
Both need a human action, so start them before anything else.

### BLUCK — highest priority
- **Not on HF under any id.** I searched the Hub API and the arXiv abstract page;
  the abstract states no release URL.
- Paper: [arXiv 2505.21092](https://arxiv.org/abs/2505.21092) ·
  [ACL Anthology 2025.banglalp-1.14](https://aclanthology.org/2025.banglalp-1.14/) (BLP-2025)
- 2,366 MCQs, 23 categories of Bangladeshi culture/history + Bengali linguistics.
- **Action:** open the PDF's data-availability statement, else email the authors.
  This is *the* set that measures the cultural tail an English-centric generator
  won't preserve — it is worth the email.
- Drop it at `data/raw/bluck.jsonl`, then add `bluck` to `eval_sets` in
  `configs/bn_g5.yaml`.

### BEnQA — highest value for H2
- **Not on HF** (`sheikhshafayat/BEnQA` returns 401/not-found).
- Distributed on GitHub: <https://github.com/sheikhshafayat/BEnQA>
  ([ACL Findings 2024](https://aclanthology.org/2024.findings-acl.68/), arXiv 2403.10900)
- ~5K **parallel bn/en** exam questions.
- **Why it matters more than its size suggests:** the *same item* exists in both
  languages. That gives you a matched-baseline comparison for H2 by construction,
  instead of relying on the subsampling correction. If any single dataset can
  defuse the "you measured the accuracy gap, not the collapse rate" reviewer
  attack, it is this one.
- Convert to `data/raw/benqa.jsonl` with fields
  `question`, `choices`, `answer_idx`.

### BanglaMedQA (local file, not used in the paper)
- Path: `data/raw/banglamedqa.jsonl` (fields: `question`, `choices`, `answer_idx`).
- Held out entirely. The contamination guard enforces this.
- Two public datasets with this name exist on the Hub
  (`ajwad-abrar/BanglaMedQA`, `Oni279/BanglaMedQA`); any overlap with a local copy
  should be declared.

---

## 6. `datasets` 5.0 broke script-based loaders

`datasets` 5.0 removed support for repos that ship a `.py` loading script, which
kills `csebuetnlp/squad_bn`, `sartajekram/BanglaRQA`, and `ai4bharat/IndicQA`
with:

```
Dataset scripts are no longer supported, but found squad_bn.py
```

The loader now detects this and automatically retries against the Hub's
auto-converted `refs/convert/parquet` branch, which recovers squad_bn and
BanglaRQA. IndicQA is not recoverable that way (the conversion flattens its
per-language configs into one `default`).

If you hit this on a *new* dataset, that fallback is already in
`load_dataset_key` — you should see a `retrying via refs/convert/parquet` warning
rather than a failure.

---

## 7. Not needed

The raw-text corpora in plan §4a (Bangla2B+, CulturaX-bn, Sangraha, FineWeb2-bn)
are only for the optional continued-pretraining arm, which §5 rules out. **Skip
them** — that is ~30 GB you don't need to download.

---

## 8. Model weights

| model | gated? | note |
|---|---|---|
| `Qwen/Qwen3-1.7B` | **No** (Apache-2.0) | Primary. Downloads without a token. |
| `meta-llama/Llama-3.2-1B` | **YES** | Needs a HF account, licence acceptance, and `huggingface-cli login`. You have **no token configured** — do this before the second arm. |
| `Qwen/Qwen1.5-0.5B` | No | Already cached; used by the smoke test |

If you would rather avoid the Llama gate entirely, `Qwen2.5-1.5B` and
`HuggingFaceTB/SmolLM2-1.7B` are ungated second families.
