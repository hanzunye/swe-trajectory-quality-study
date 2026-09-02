# Data Quantity Dominates Quality in LoRA Fine-Tuning for Code Agents

[![DOI](https://img.shields.io/badge/DOI-10.1007%2F978--981--92--3492--9__4-0b6bcb)](https://doi.org/10.1007/978-981-92-3492-9_4)
[![arXiv](https://img.shields.io/badge/arXiv-2607.17205-b31b1b)](https://arxiv.org/abs/2607.17205)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-swe--trajectory--subsets-ffcc4d)](https://huggingface.co/datasets/davongluck/swe-trajectory-subsets)
[![License](https://img.shields.io/badge/Code%20License-MIT-green)](LICENSE)

Official code and experimental artefacts for the ICIC 2026 paper
**"A Systematic Evaluation of Trajectory Data Curation for LoRA Fine-Tuning of
Code Agents"** — a systematic empirical study of data quality filtering
strategies for LoRA fine-tuning of code agent LLMs on the SWE-trajectory dataset.

> **Publication.** Yunze Han. *A Systematic Evaluation of Trajectory Data
> Curation for LoRA Fine-Tuning of Code Agents.* In **Advanced Intelligent
> Computing Technology and Applications** (Proceedings of ICIC 2026, the 22nd
> International Conference on Intelligent Computing, Toronto, ON, Canada,
> July 22–26, 2026), Lecture Notes in Computer Science, Springer Nature
> Singapore, pp. 39–50, 2027 (published online 14 July 2026). Oral presentation.
> DOI: [10.1007/978-981-92-3492-9_4](https://doi.org/10.1007/978-981-92-3492-9_4)
> · Preprint (author's accepted manuscript): [arXiv:2607.17205](https://arxiv.org/abs/2607.17205)

## Abstract

Supervised fine-tuning (SFT) of open-weight LLMs on expert agent trajectories has
emerged as a prominent approach to building capable code agents without reliance
on proprietary models. A central yet underexplored question is how trajectory
quality and quantity jointly shape model performance. We present a systematic
empirical study of trajectory data filtering for LoRA fine-tuning of
Qwen2.5-Coder-7B-Instruct on the SWE-trajectory dataset (67,074 trajectories, of
which 32,161 are resolved). We propose a two-axis quality scoring framework —
Efficiency and Style — and evaluate it through 16 controlled experiments spanning
strategy, scale, and ablation analyses. Since 7B-scale models attain near-zero
SWE-bench resolve rates, we adopt cross-entropy (CE) loss on held-out
trajectories as the primary metric, validated via first-action generation: CE
loss and ROUGE-L are perfectly rank-correlated (Spearman ρ = −1.00), with
limited-sample evidence supporting but not conclusively establishing this proxy.
Our results reveal a scale-dependent quality–quantity trade-off: at small scales,
doubling the dataset (500 → 1,000) yields ~12.7% CE-loss reduction whereas the
TopQ–Random gap stays < 1% (Mann–Whitney p > 0.10); at 2,000 trajectories this
same gap widens to 3.6% (p = 0.016). Ablation further identifies error-retry rate
as the dominant sub-dimension, performing comparably to the full composite
(Δ < 0.2%). Together, these findings establish trajectory-level quality scoring
as a viable but scale-sensitive lever for code-agent SFT and offer a
proxy-validated evaluation protocol for the regime where end-to-end resolve rate
is statistically infeasible.

## Key Finding

In the 500–2000 trajectory regime, **data quantity effects (~12.7% relative loss reduction)** substantially exceed **data quality filtering effects (~0.7%, p > 0.10)**. However, the quality gap widens at 2000 samples (3.6% vs 0.7% at 500), suggesting a crossover point may exist at larger scales where quality filtering becomes the dominant factor.

> **Note.** Three supplementary experiments were added during peer review and are part of the published version: (A) first-action evaluation for proxy metric validation, (B) scaling extension to 2000 trajectories, and (C) B2-only baseline comparison. See [Supplementary Validation](#supplementary-validation) below.

<p align="center">
  <img src="figures/fig2_loss_comparison.png" width="90%" alt="Loss Comparison">
</p>

## Paper & Resources

| Resource | Link | Description |
|----------|------|-------------|
| Published paper | [doi.org/10.1007/978-981-92-3492-9_4](https://doi.org/10.1007/978-981-92-3492-9_4) | ICIC 2026 proceedings version (Springer LNCS, pp. 39–50) |
| Preprint | [arxiv.org/abs/2607.17205](https://arxiv.org/abs/2607.17205) | Author's accepted manuscript (12 pages, open access) |
| Quality-Scored Subsets | [huggingface.co/datasets/davongluck/swe-trajectory-subsets](https://huggingface.co/datasets/davongluck/swe-trajectory-subsets) | 16 curated training subsets with quality scores |


## Experiment Overview

### Scoring Framework

Trajectory quality is decomposed along two axes with four sub-metrics:

| Dimension | Sub-metric | Description | Discriminability (σ) |
|-----------|-----------|-------------|---------------------|
| **Efficiency** | B2: Error-Retry Rate | Action-error-retry loops | 0.286 (highest) |
| **Efficiency** | B3: Step-Count Ratio | Trajectory length vs. median | 0.063 |
| **Style** | C2: Action Diversity | Shannon entropy of action types | 0.046 |
| **Style** | C3: Obs. Utilization | Reference to observation entities | 0.118 |

### 16 Controlled Experiments

```
Block 1 — Strategy Comparison (7 experiments)
  Baseline (no SFT), Random-{500,1000}, TopQ-{500,1000},
  ResolvedOnly-{500,1000}, BottomQ-500

Block 2 — High-Level Ablation (2 experiments)
  NoEfficiency-500, NoStyle-500

Block 3 — Sub-Dimension Ablation (4 experiments)
  NoB2-500, NoB3-500, NoC2-500, NoC3-500

Block 4 — Supplementary (3 experiments, added in revision)
  Random-2000, TopQ-2000, B2Only-Top500
```

### Main Results

| Hypothesis | Comparison | Result | p-value |
|-----------|-----------|--------|---------|
| H1: Gate effect | ResolvedOnly vs Random | Not supported | 0.783 |
| H2: Score effect | TopQ vs ResolvedOnly | Directionally correct | 0.105 |
| H3: Scaling | 500 → 1000 | **Strongly supported** (~12.7%) | < 0.001 |
| H4: Sanity check | TopQ vs BottomQ | **Significant** | 0.007 |
| H5: Efficiency > Style | NoEff vs NoStyle | Directionally correct | 0.154 |

| Supplementary Finding | Evidence | Significance |
|----------------------|----------|-------------|
| CE Loss is a valid proxy | Spearman ρ = −1.00 (CE Loss vs ROUGE-L) | p < 0.001 |
| Scaling continues to 2000 | Random −18.3%, TopQ −20.7% (500→2000) | Diminishing but sustained |
| Quality-quantity crossover | TopQ−Random gap: 0.7%@500 → 3.6%@2000 | 4× widening (p = 0.016 @2000) |
| B2 ≈ Composite @500 | CE Loss 0.4714 vs 0.4704 (Δ = 0.2%) | Not significant |

<p align="center">
  <img src="figures/fig3_hypothesis_validation.png" width="90%" alt="Hypothesis Validation">
  <br><em>Hypothesis validation across three test sets</em>
</p>

<p align="center">
  <img src="figures/fig4_ablation_impact.png" width="80%" alt="Ablation Impact">
  <br><em>Ablation impact — B2 (error-retry) is the most impactful sub-dimension</em>
</p>

## Supplementary Validation

In response to reviewer feedback, we conducted three additional experiments addressing: (1) proxy metric lacks downstream validation, (2) scale limited to 500–1000, and (3) composite score vs single-metric baseline.

### Experiment A: First-Action Evaluation

Since the 7B model achieves near-zero resolve rate on SWE-bench, end-to-end evaluation is infeasible. Instead, we evaluate **first-action generation quality**: the model generates its first action given an issue prompt, compared against ground truth using ROUGE-L, file match, and action type accuracy.

| Checkpoint | Strategy | CE Loss (Gold) | ROUGE-L | File Match |
|---|---|---|---|---|
| baseline | No fine-tune | 0.9100 | 0.137 | 0.573 |
| exp1 | Random-500 | 0.4737 | 0.200 | 0.680 |
| exp3 | TopQ-500 | 0.4704 | 0.212 | 0.670 |
| exp2 | Random-1000 | 0.4140 | 0.248 | 0.650 |

**Spearman ρ = −1.00 (p < 0.001)** between CE Loss and ROUGE-L, validating CE loss as a reliable proxy for downstream action quality.

<p align="center">
  <img src="figures/fig5_proxy_validation.png" width="80%" alt="Proxy Metric Validation">
  <br><em>CE Loss vs First-Action ROUGE-L — perfect negative correlation</em>
</p>

### Experiment B: Scale Extension to 2000

| Strategy | 500 | 1000 | 2000 | Δ (500→2000) |
|---|---|---|---|---|
| Random | 0.4737 | 0.4140 | 0.3871 | −18.3% |
| TopQ | 0.4704 | 0.4106 | 0.3732 | −20.7% |
| **TopQ − Random** | **0.003 (0.7%)** | **0.003 (0.8%)** | **0.014 (3.6%)** | **4× widening** |

The quality gap widens from 0.003 at 500 samples to 0.014 at 2000 samples, suggesting quality filtering becomes increasingly important at larger scales.

<p align="center">
  <img src="figures/fig6_scaling_curve.png" width="80%" alt="Scaling Curve">
  <br><em>CE Loss scaling curve — quality gap widens at 2000 samples</em>
</p>

### Experiment C: B2-Only Baseline

| Strategy | CE Loss (Gold) | PPL |
|---|---|---|
| Composite-Top500 | 0.4704 | 1.601 |
| B2Only-Top500 | 0.4714 | 1.602 |

The difference is negligible (Δ = 0.001, 0.2%). Practitioners can use B2 alone as a lightweight proxy, though the composite score may yield greater benefits at larger scales.

<p align="center">
  <img src="figures/fig8_b2only_ablation.png" width="70%" alt="B2-Only Ablation">
  <br><em>Composite vs B2-only — comparable at 500 samples</em>
</p>

## Reproducing the Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Hardware: NVIDIA A100 80GB GPU (PyTorch 2.4.0, CUDA 12.4.1)

### Step 1: Quality Scoring

```bash
python scripts/scoring/run_analysis.py
```

This writes the Phase 1 outputs to `data/quality_scores/full/`, including:
- `trajectory_stats.jsonl` for all trajectories
- `trajectory_scored_v3.csv` for the resolved, gate-passing scoring pool

### Step 2: Subset Construction

```bash
python scripts/subset/run_subset.py
```

By default this builds all 16 training subsets into `data/subsets/`, including:
- Core subsets: `Random-{500,1000}`, `TopQ-{500,1000}`, `ResolvedOnly-{500,1000}`, `BottomQ-500`
- Ablations: `Ablation-NoEfficiency-500`, `Ablation-NoStyle-500`, `Ablation-NoB2-500`, `Ablation-NoB3-500`, `Ablation-NoC2-500`, `Ablation-NoC3-500`
- Supplementary subsets: `Random-2000`, `TopQ-2000`, `B2Only-Top500`

Useful subset commands:

```bash
# List all available subset names
python scripts/subset/run_subset.py --list

# Build one subset only
python scripts/subset/run_subset.py --only TopQ-2000

# Skip comparison plots
python scripts/subset/run_subset.py --skip-viz
```

If your scoring outputs live outside the default location, point the subset builder at them with:

```bash
SCORING_ROOT=/path/to/quality_scores/full python scripts/subset/run_subset.py
```

### Step 3: Training

```bash
bash scripts/training/run_experiments.sh
```

### Step 4: Evaluation

```bash
# Core evaluation (exp1–13)
bash scripts/evaluation/run_pipeline.sh

# Supplementary: first-action evaluation
python scripts/evaluation/eval_first_action.py \
    --models baseline exp1 exp2 exp3

# Supplementary: extended perplexity (exp14–16)
python scripts/evaluation/eval_perplexity_extended.py \
    --models exp14 exp15 exp16

# Supplementary: first-action at scale
python scripts/evaluation/eval_first_action.py \
    --models exp14 exp15 exp16

# Regenerate figures from existing results
python scripts/evaluation/eval_first_action.py --plot-only
```
## Project Structure

```
swe-trajectory-quality-study/
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── scoring/                         # Phase 1: Quality Scoring
│   │   ├── run_analysis.py
│   │   ├── scoring.py                   # Core scoring logic (B2, B3, C2, C3)
│   │   ├── scoring_config.py
│   │   ├── analysis.py
│   │   └── scoring_visualize.py
│   │
│   ├── subset/                          # Phase 2: Subset Construction
│   │   ├── run_subset.py
│   │   ├── builder.py                   # Selection logic (TopQ, Random, etc.)
│   │   ├── subset_config.py             # 16 subset configurations
│   │   ├── subset_visualize.py
│   │   └── upload_to_hf.py
│   │
│   ├── training/                        # Phase 3: LoRA Fine-Tuning
│   │   ├── prepare_data.py              # ChatML serialization & tokenization
│   │   ├── train.py                     # QLoRA training (Qwen2.5-Coder-7B)
│   │   ├── configs.py
│   │   ├── run_experiments.sh
│   │   └── setup.sh
│   │
│   └── evaluation/                      # Phase 4: Evaluation
│       ├── build_test_set.py            # Gold/Random/Low-Q test set construction
│       ├── eval_perplexity.py           # Cross-entropy loss evaluation
│       ├── eval_perplexity_extended.py  # Extended eval (exp14–16)
│       ├── eval_next_action.py          # Next-action prediction
│       ├── eval_first_action.py         # First-action generation eval
│       ├── analyze_results.py           # Statistical analysis & hypothesis testing
│       ├── run_eval.sh
│       └── run_pipeline.sh
│
├── data/
│   ├── quality_scores/
│   │   ├── summary_statistics.csv
│   │   └── trajectory_analysis.csv
│   │
│   ├── subsets/                         # 16 training subsets (metadata.json each)
│   │   └── {TopQ,Random,ResolvedOnly,BottomQ,Ablation-*}-{500,1000,2000}/ + B2Only-Top500/
│   │
│   ├── training_logs/
│   │   └── exp{1..16}_summary.json
│   │
│   └── eval_results/
│       ├── perplexity_results.csv           # exp1–13 (14 models × 3 test sets)
│       ├── perplexity_results_extended.csv  # exp1–16 (17 models × 3 test sets)
│       ├── next_action_results.json
│       ├── first_action_results.json
│       ├── first_action_summary.csv
│       ├── first_action_summary_extended.csv
│       ├── first_action_correlation.json
│       ├── experiment_A_analysis.md
│       ├── SUMMARY_REPORT.md
│       ├── test_set_ids.json
│       ├── stats_report.md
│       └── experiment_report.md
│
├── figures/
│   ├── fig1_score_distributions.png
│   ├── fig2_loss_comparison.png
│   ├── fig3_hypothesis_validation.png
│   ├── fig4_ablation_impact.png
│   ├── fig5_proxy_validation.png        # CE Loss vs ROUGE-L (Exp A)
│   ├── fig6_scaling_curve.png           # Scaling 500→2000 (Exp B)
│   ├── fig7_fa_scaling_curve.png        # First-Action scaling (Exp B)
│   ├── fig8_b2only_ablation.png         # Composite vs B2-only (Exp C)
│   ├── fig9_first_action_bars.png       # First-action bar chart (Exp A)
│   └── supp_*.png                       # Supplementary figures
│
└── configs/
    └── experiment_design_v3.md
```

## Citation

If you use this repository or its results, please cite the published ICIC 2026
proceedings version:

```bibtex
@InProceedings{han2027trajectory,
  author    = {Han, Yunze},
  editor    = {Huang, De-Shuang and Li, Bo and Zhang, Qinhu and Bao, Wenzheng},
  title     = {A Systematic Evaluation of Trajectory Data Curation for
               LoRA Fine-Tuning of Code Agents},
  booktitle = {Advanced Intelligent Computing Technology and Applications},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer Nature Singapore},
  address   = {Singapore},
  pages     = {39--50},
  year      = {2027},
  isbn      = {978-981-92-3492-9},
  doi       = {10.1007/978-981-92-3492-9_4},
}
```

The proceedings carry a 2027 imprint year; the chapter was published online on
14 July 2026. If you need the open-access preprint instead:

```bibtex
@misc{han2026trajectoryarxiv,
  author        = {Han, Yunze},
  title         = {A Systematic Evaluation of Trajectory Data Curation for
                   LoRA Fine-Tuning of Code Agents},
  year          = {2026},
  eprint        = {2607.17205},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  doi           = {10.48550/arXiv.2607.17205},
}
```

## License

The code and experimental artefacts in this repository are released under the
MIT License — see [LICENSE](LICENSE) for details. The published paper itself
is © Springer Nature Singapore Pte Ltd.; please refer to the
[published version](https://doi.org/10.1007/978-981-92-3492-9_4) for the
authoritative copyright notice. The author's accepted manuscript is available
on [arXiv](https://arxiv.org/abs/2607.17205).
