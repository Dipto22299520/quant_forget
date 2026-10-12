# Self-training and model confidence

Code for a study of what recursive self-training, and the remedies proposed against model
collapse, do to a language model's confidence. A model answers a fixed pool of questions, is
fine-tuned on its own answers, and the fine-tuned model starts the next round. Every round is
evaluated on the same held-out multiple-choice items, and confidence is measured as two
properties: *temperature*, the gap between confidence and accuracy under a temperature calibrated
at the start, and *discrimination*, the AUROC of confidence as a predictor of correctness.

## What is here

| Path | Contents |
|---|---|
| `src/collapse/loop.py` | the self-training loop: generate, choose training answers, fine-tune with LoRA, merge, evaluate |
| `src/collapse/modeling/` | model loading, generation (top-p, exact and entropy-matched sampling), option scoring, training losses (cross-entropy, TCE, focal, entropy bonus, label smoothing) |
| `src/collapse/filters.py` | choosing training answers: random subsets, confidence-based self-verification, reference mixing, several samples per prompt |
| `src/collapse/eval/` | calibration, diversity and perplexity metrics |
| `src/collapse/data/` | dataset registry, loaders and preprocessing |
| `src/collapse/cli.py` | command line: `run`, `rescore`, `analyze`, `check-data`, `check-env` |
| `configs/` | experiment configurations (`main_en.yaml` is the main English setup) |
| `scripts/run_all.ps1`, `scripts/run_all_linux.py` | run the experiment stages end to end |
| `scripts/mechanism_assets.py`, `scripts/label_free.py`, `scripts/paper_v2.py` | analysis: calibration gap, label-free recalibration, discrimination measures, tables and figures |
| `tests/` | unit tests |

Model weights and per-run records are not included; together they take several hundred gigabytes.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt     # Windows: .venv\Scripts\pip install -r requirements.txt
python scripts/download_data.py               # datasets are described in DATASETS.md
PYTHONPATH=src python -m collapse.cli check-env
```

## Running

One protocol and one seed:

```bash
PYTHONPATH=src python -m collapse.cli run --config configs/main_en.yaml --run-name main_en_r100_s1
```

Settings can be overridden with `--set`, for example `--set loop.ratio=0.5` to replace half of each
round's answers with reference answers. `scripts/run_all.ps1` (Windows) and
`scripts/run_all_linux.py` run the full set of stages.

Once the runs have finished, the analysis scripts regenerate every number, table and figure from
the saved per-item records (they write to `paper/`):

```bash
PYTHONPATH=src python scripts/mechanism_assets.py
PYTHONPATH=src python scripts/label_free.py
PYTHONPATH=src python scripts/paper_v2.py
```

## Tests

```bash
PYTHONPATH=src python -m pytest tests
```

## Note on Bangla runs

Bangla runs made before the prompt-fitting fix (`fit_prompt` in `src/collapse/modeling/prompts.py`)
are invalid: a right-truncation default could cut the question from long passage-bearing prompts.
The English results are unaffected.
