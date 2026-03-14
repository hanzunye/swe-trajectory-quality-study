# Data Quantity Dominates Quality in LoRA Fine-Tuning for Code Agents

A systematic empirical study of data quality filtering strategies for LoRA fine-tuning of code agent LLMs on the SWE-trajectory dataset.

> **Paper**: *Data Quantity Dominates Quality in LoRA Fine-Tuning for Code Agents: A Systematic Study on Trajectory Data Filtering Strategies*
> **Venue**: ICIC 2026 (International Conference on Intelligent Computing)

## Key Finding

At the 500-1,000 trajectory scale, **data quantity effects (~12.7% relative loss reduction)** substantially exceed **data quality filtering effects (~0.7%, p > 0.10)**. Quality-aware selection becomes increasingly relevant only as dataset size grows and marginal scaling returns diminish.

<p align="center">
  <img src="figures/fig2_loss_comparison.png" width="90%" alt="Loss Comparison">
</p>

## HuggingFace Resources

| Resource | Link | Description |
|----------|------|-------------|
| Quality-Scored Subsets | [huggingface.co/datasets/YOUR_USERNAME/swe-trajectory-quality-subsets](https://huggingface.co/datasets/YOUR_USERNAME/swe-trajectory-quality-subsets) | 13 curated training subsets with quality scores |
| LoRA Adapters | [huggingface.co/YOUR_USERNAME/swe-trajectory-lora-adapters](https://huggingface.co/YOUR_USERNAME/swe-trajectory-lora-adapters) | All 13 trained LoRA adapters (Qwen2.5-Coder-7B) |
| Evaluation Results | [huggingface.co/datasets/YOUR_USERNAME/swe-trajectory-eval-results](https://huggingface.co/datasets/YOUR_USERNAME/swe-trajectory-eval-results) | Perplexity & next-action prediction results |

> Replace `YOUR_USERNAME` with your actual HuggingFace username before publishing.

## Project Structure

```
swe-trajectory-quality-study/
├── README.md
├── requirements.txt
│
├── scripts/
│   ├── scoring/                    # Phase 1: Quality Scoring Framework
│   │   ├── run_analysis.py         # Entry point for scoring pipeline
│   │   ├── scoring.py              # Core scoring logic (B2, B3, C2, C3)
│   │   ├── scoring_config.py       # Metric definitions and thresholds
│   │   ├── analysis.py             # Statistical analysis utilities
│   │   └── scoring_visualize.py    # Score distribution visualization
│   │
│   ├── subset/                     # Phase 2: Subset Construction
│   │   ├── run_subset.py           # Entry point for subset builder
│   │   ├── builder.py              # Subset selection logic (TopQ, Random, etc.)
│   │   ├── subset_config.py        # Subset definitions (13 configs)
│   │   ├── subset_visualize.py     # Subset comparison visualizations
│   │   └── upload_to_hf.py         # Upload subsets to HuggingFace Hub
│   │
│   ├── training/                   # Phase 3: LoRA Fine-Tuning
│   │   ├── prepare_data.py         # ChatML serialization & tokenization
│   │   ├── train.py                # QLoRA training script (Qwen2.5-Coder-7B)
│   │   ├── configs.py              # Training hyperparameters
│   │   ├── run_experiments.sh      # Batch experiment runner
│   │   └── setup.sh                # Environment setup
│   │
│   └── evaluation/                 # Phase 4: Evaluation
│       ├── build_test_set.py       # Gold/Random/Low-Q test set construction
│       ├── eval_perplexity.py      # Cross-entropy loss evaluation
│       ├── eval_next_action.py     # Next-action prediction evaluation
│       ├── analyze_results.py      # Statistical analysis & hypothesis testing
│       ├── run_eval.sh             # Single-model evaluation runner
│       └── run_pipeline.sh         # Full evaluation pipeline
│
├── data/
│   ├── quality_scores/
│   │   ├── summary_statistics.csv           # Aggregate scoring statistics
│   │   └── trajectory_analysis.csv          # Per-trajectory analysis data
│   │
│   ├── subsets/
│   │   ├── subset_comparison_table.csv      # Side-by-side subset statistics
│   │   ├── TopQ-500/metadata.json           # Subset metadata & stats
│   │   ├── TopQ-1000/metadata.json
│   │   ├── Random-500/metadata.json
│   │   ├── Random-1000/metadata.json
│   │   ├── ResolvedOnly-500/metadata.json
│   │   ├── ResolvedOnly-1000/metadata.json
│   │   ├── BottomQ-500/metadata.json
│   │   ├── Ablation-NoEfficiency-500/metadata.json
│   │   ├── Ablation-NoStyle-500/metadata.json
│   │   ├── Ablation-NoB2-500/metadata.json
│   │   ├── Ablation-NoB3-500/metadata.json
│   │   ├── Ablation-NoC2-500/metadata.json
│   │   └── Ablation-NoC3-500/metadata.json
│   │
│   ├── training_logs/
│   │   └── exp{1..13}_summary.json          # Per-experiment training stats
│   │
│   └── eval_results/
│       ├── perplexity_results.csv           # Full loss/PPL results (14 models x 3 test sets)
│       ├── summary_table.csv                # Condensed result table
│       ├── next_action_results.json         # Action prediction accuracy & ROUGE-L
│       ├── test_set_ids.json                # Test set trajectory IDs (reproducibility)
│       ├── stats_report.md                  # Statistical test report
│       └── experiment_report.md             # Full experiment analysis
│
├── figures/
│   ├── fig1_score_distributions.png         # Score distributions (9 sub-metrics)
│   ├── fig2_loss_comparison.png             # Cross-entropy loss across conditions
│   ├── fig3_hypothesis_validation.png       # H1/H2/H4 hypothesis tests
│   ├── fig4_ablation_impact.png             # Ablation impact bar chart
│   ├── supp_quality_scores.png              # Supplementary: composite scores
│   ├── supp_token_distribution.png          # Supplementary: token length dist.
│   ├── supp_turn_distribution.png           # Supplementary: turn count dist.
│   ├── supp_subset_quality_box.png          # Supplementary: subset quality box
│   └── supp_subset_tokens.png              # Supplementary: subset token dist.
│
└── configs/
    └── experiment_design_v3.md              # Full experiment design document
```

## Experiment Overview

### Scoring Framework

We decompose trajectory quality along two axes with four sub-metrics:

| Dimension | Sub-metric | Description | Discriminability (sigma) |
|-----------|-----------|-------------|------------------------|
| **Efficiency** | B2: Error-Retry Rate | Detects action-error-retry loops | 0.286 (highest) |
| **Efficiency** | B3: Step-Count Ratio | Trajectory length vs. median | 0.063 |
| **Style** | C2: Action Diversity | Shannon entropy of action types | 0.046 |
| **Style** | C3: Obs. Utilization | Reference to observation entities | 0.118 |

### 13 Controlled Experiments

```
Block 1 — Strategy Comparison (7 experiments)
  Baseline (no SFT), Random-{500,1000}, TopQ-{500,1000},
  ResolvedOnly-{500,1000}, BottomQ-500

Block 2 — High-Level Ablation (2 experiments)
  NoEfficiency-500, NoStyle-500

Block 3 — Sub-Dimension Ablation (4 experiments)
  NoB2-500, NoB3-500, NoC2-500, NoC3-500
```

### Main Results

| Hypothesis | Comparison | Result | p-value |
|-----------|-----------|--------|---------|
| H1: Gate effect | ResolvedOnly vs Random | Not supported | 0.783 |
| H2: Score effect | TopQ vs ResolvedOnly | Directionally correct | 0.105 |
| H3: Scaling | 500 -> 1000 | **Strongly supported** (~12.7%) | < 0.001 |
| H4: Sanity check | TopQ vs BottomQ | **Significant** | 0.007 |
| H5: Efficiency > Style | NoEff vs NoStyle | Directionally correct | 0.154 |

<p align="center">
  <img src="figures/fig3_hypothesis_validation.png" width="90%" alt="Hypothesis Validation">
  <br><em>Fig. 3: Hypothesis validation across three test sets</em>
</p>

<p align="center">
  <img src="figures/fig4_ablation_impact.png" width="80%" alt="Ablation Impact">
  <br><em>Fig. 4: Ablation impact — B2 (error-retry) is the most impactful sub-dimension</em>
</p>

## Reproducing the Experiments

### Prerequisites

```bash
pip install -r requirements.txt
```

Hardware: NVIDIA A100 80GB GPU (PyTorch 2.4.0, CUDA 12.4.1)

### Step 1: Quality Scoring

```bash
# Score all 67,074 trajectories
python scripts/scoring/run_analysis.py \
    --input /path/to/swe-trajectory-dataset \
    --output data/quality_scores/
```

### Step 2: Subset Construction

```bash
# Build all 13 training subsets
python scripts/subset/run_subset.py \
    --scores data/quality_scores/trajectory_scored_v3.csv \
    --output data/subsets/
```

### Step 3: Training (13 experiments)

```bash
# Run all experiments
bash scripts/training/run_experiments.sh
```

### Step 4: Evaluation

```bash
# Build test sets and run full evaluation
bash scripts/evaluation/run_pipeline.sh
```

## Citation

```bibtex
@inproceedings{anonymous2026quantity,
  title={Data Quantity Dominates Quality in {LoRA} Fine-Tuning for Code Agents: A Systematic Study on Trajectory Data Filtering Strategies},
  author={Anonymous},
  booktitle={Proceedings of the International Conference on Intelligent Computing (ICIC)},
  year={2026}
}
```

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.
