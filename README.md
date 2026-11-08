# Reproducing USAD on SMD, and testing it on new data

COMP8240 Applications of Data Science, Macquarie University, Session 2 2026. Vishnu Prasad (48750301).

This repository holds everything behind my project report on **USAD: UnSupervised Anomaly
Detection on Multivariate Time Series** (Audibert et al., KDD 2020):

1. **Replication** of USAD on all 28 machines of the Server Machine Dataset (SMD) with the official code.
2. **Existing new dataset**: SKAB (Skoltech Anomaly Benchmark), not used in the paper.
3. **Constructed dataset**: UCI household power data with 80 injected anomalies of known type, timing and severity.
4. **LLM-as-judge**: an LLM reviews USAD's alerts on the constructed dataset, scored against the injection log.

The USAD model code (`src/usad.py`, `src/utils.py`) is copied **unchanged** from the official
repository [manigalati/usad](https://github.com/manigalati/usad) (BSD-3, see `src/USAD_LICENSE`). Report source: [`report/main.tex`](report/main.tex) (the compiled PDF is submitted on iLearn).
Everything else is my own.

## Layout

```
notebooks/
  Step1_USAD_SMD.ipynb             replication on SMD (Colab, T4 GPU)
  Step2_USAD_SKAB.ipynb            USAD on SKAB (Colab)
  Step3_Constructed_Dataset.ipynb  builds the UCI + injected-anomaly dataset, first USAD run (Colab)
src/
  usad.py, utils.py                official USAD implementation (unmodified)
  common.py                        windowing, point adjustment, best-F1 threshold search
  build_constructed_dataset.py     script version of Step 3 data construction (same seed, same output)
  run_usad_uci.py                  trains USAD on the constructed dataset, saves scores + per-segment detection
  llm_judge_prepare.py             turns USAD alerts into blind judge prompts + a hidden answer key
  llm_judge_run.py                 sends prompts to an LLM through the Anthropic API
  llm_judge_evaluate.py            scores verdicts against the key (accuracy, precision, recall, Cohen's kappa)
  make_figures.py                  summary figures for the report
scripts/get_data.sh                downloads SMD, SKAB and the UCI file into data/
results/                           every number in the report, as CSV/JSON
figures/                           every figure in the report
```

## How to run

```bash
pip install -r requirements.txt
bash scripts/get_data.sh

# Constructed dataset + USAD (CPU is fine, ~5 min)
python src/build_constructed_dataset.py --raw data/household_power_consumption.txt --out data/uci
python src/run_usad_uci.py --data data/uci --out results/uci

# LLM-as-judge
python src/llm_judge_prepare.py                      # writes results/llm_judge/items.jsonl + items_key.csv
export ANTHROPIC_API_KEY=...                         # optional: rerun the judge yourself
python src/llm_judge_run.py --out results/llm_judge/judgments_api.jsonl
python src/llm_judge_evaluate.py --judgments results/llm_judge/judgments.jsonl

python src/make_figures.py
```

SMD and SKAB: open the notebooks in Google Colab (Runtime → T4 GPU → Run all). They clone the
official USAD code and the datasets themselves and write `smd_results.csv`, `smd_summary.csv`,
`skab_results.csv` and `skab_summary.csv` (copies are in `results/`).

## Settings (same everywhere)

Window 12, latent size 12 × 100, 100 epochs, Adam defaults, α = β = 0.5, min-max scaling fitted on
training data only, 80/20 train/validation split of the normal data, seed 0. Thresholds are chosen by
searching 300 score quantiles for the best F1 (the OmniAnomaly benchmark protocol), with and without
point adjustment. F1* is the F1 of the precision and recall averaged over entities (machines or files).

## Headline results

| Dataset | Setting | Precision | Recall | F1* |
|---|---|---|---|---|
| SMD, paper (Audibert et al., 2020) | with PA | 0.931 | 0.962 | 0.946 |
| SMD, this replication (28 machines) | with PA | 0.792 | 0.736 | 0.763 |
| SMD, this replication | without PA | 0.591 | 0.476 | 0.528 |
| SKAB (34 files) | with PA | 0.947 | 1.000 | 0.973 |
| SKAB | without PA | 0.602 | 0.929 | 0.730 |
| UCI constructed | with PA | 0.538 | 0.433 | 0.480 (F1) |
| UCI constructed | without PA | 0.497 | 0.188 | 0.273 (F1) |

LLM judge on 110 USAD alerts: accuracy 0.95, Cohen's κ = 0.88 against the injection log; filtering
alerts through the judge raises alert precision from 0.25 to 0.92.

## Data licences

SMD: MIT (OmniAnomaly repository). SKAB: GPL-3.0. UCI household power: CC BY 4.0. No personal data
is included; the UCI file describes one anonymous household and is not redistributed here.

## Reproducibility notes

- `build_constructed_dataset.py` reproduces the Colab injection log byte for byte (checked).
- `run_usad_uci.py` on CPU (PyTorch 2.15) gives F1 0.273 / 0.480 (without / with PA) against 0.271 / 0.480 from the
  Colab GPU run; per-type detection rates are identical.
- The verdicts in `results/llm_judge/judgments.jsonl` were produced by Claude Opus 5.5 from the exact
  prompts in `items.jsonl`; see the report (Section 6) for how blinding was handled and its limits.
