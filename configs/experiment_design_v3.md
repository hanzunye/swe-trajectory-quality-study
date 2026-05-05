# Trajectory-Quality-Aware Data Selection — Experiment Design v3

**Base model**: Qwen2.5-Coder-7B-Instruct
**Fine-tuning method**: LoRA
**Data source**: SWE-trajectory dataset
**Date**: February 2026
**Status**: Accepted to ICIC 2026 (Toronto, Canada). Paper ID 700.

---

## 1. Scoring Framework

### 1.1 Design Principle

The scoring framework answers a single core question: **"If a human expert were performing the same task, how would they evaluate this trajectory?"**

When a human expert evaluates a developer's debugging process, they care about two things: **how cleverly is it done (Efficiency)?** and **how clean is the process (Style)?** Whether the task was actually solved (Correctness) and whether the trajectory is complete (Completeness) are treated as up-front filtering conditions and do not enter the continuous score.

### 1.2 Pre-scoring Filters (Gate Conditions)

The following conditions are *not* scored; they only determine whether a trajectory enters the scoring pool.

| Gate | Condition | Rationale |
|------|-----------|-----------|
| **Completeness Gate** | Truncation Ratio >= 0.9 | The truncation ratio is essentially constant on this dataset (median = 1.0, std ~ 0). It carries no discriminating power and is used only to clean a small amount of dirty data. |
| **Correctness Gate** | Outcome Success = 1 (Resolved) | A binary variable should not be mixed with continuous variables in a weighted average. We use it as a stratification condition and rank only inside the resolved pool. |
| **Format Gate** | Trajectory parses into thought-action-observation structure | Format-broken trajectories cannot be scored reliably. |

**Why is Correctness not a scoring dimension?**
Outcome Success is binary (0/1). Combining it with continuous dimensions in a weighted average lets the 0/1 term dominate the score. After moving it to the gate, every trajectory in the scoring pool is resolved, and the score focuses on "how well was the task done".

### 1.3 Continuous Scoring Dimensions (4 active dims)

After filtering, **32,161 resolved trajectories** remain. They are ranked along the following four dimensions.

#### Efficiency — is the path to the goal concise?

| Sub-dimension | Definition | Scoring | Distribution |
|---------------|------------|---------|--------------|
| **B2: Error-Retry Cycles** | Cost of retrying after errors | Count "action -> error observation -> similar action" cycles, normalize, invert | std = 0.286, median = 0.300 — **most discriminating dimension** |
| **B3: Step Count Ratio** | Reasonableness of the step count | This trajectory's step count divided by the median across all resolved trajectories of the same task; clipped, normalized, inverted | std = 0.063, median = 0.800 |

#### Style — is the trajectory clean as training data?

| Sub-dimension | Definition | Scoring | Distribution |
|---------------|------------|---------|--------------|
| **C2: Action Diversity** | Reasonable variety of tool use | Entropy of action types, normalized to [0, 1] | std = 0.046, median = 0.655 |
| **C3: Observation Utilization** | Effective use of observation content | Fraction of filenames (basename) / error class names from observations that are referenced in subsequent actions | std = 0.118, median = 0.313 |

#### Excluded dimensions and rationale

| Dimension | Reason for exclusion |
|-----------|----------------------|
| **B1: Redundant Commands** | std = 0.033, median = 0.962 — almost no discriminating power. Agents in this dataset rarely repeat exact commands. |
| **C1: Observation Cleanliness** | std = 0.043, median = 0.967 — almost no discriminating power. The vast majority of observations are already clean. |

> **Paper framing**: We designed six candidate sub-dimensions. Variance analysis showed that B1 and C1 lacked discriminating power on this dataset (std < 0.05) and were therefore excluded. This is itself a finding — the agents in the SWE-trajectory dataset are highly homogeneous in command redundancy and observation cleanliness.

### 1.4 Score Aggregation

```
Efficiency = mean(B2, B3)            # std = 0.152, median = 0.529
Style      = mean(C2, C3)            # std = 0.063, median = 0.485
Composite  = 0.5 * Efficiency + 0.5 * Style   # std = 0.083, median = 0.507
```

### 1.5 C3 Implementation Note

The first version of C3 used full-path matching (e.g. `src/utils.py`), which caused matches to fail when the agent referred to `utils.py` without the path prefix; the median was only 0.201. After switching to basename matching, the median rose to 0.313. The remaining low utilization reflects a general "read but not use" pattern in agent behaviour and is itself a finding worth discussing.

---

## 2. Experimental Group Design

### 2.1 Overview

The experiment matrix is organized into three blocks. Block A (13 runs) is the original core matrix; Blocks B and C extend the study with a scaling experiment and a single-dimension ranking probe respectively.

| # | Experiment | Pool | Selection | Size | Block |
|---|------------|------|-----------|------|-------|
| 0 | baseline | — | no fine-tuning | — | — |
| 1 | Random-500 | full | random | 500 | A1 |
| 2 | Random-1000 | full | random | 1000 | A1 |
| 3 | TopQ-500 | resolved | top by composite | 500 | A1 |
| 4 | TopQ-1000 | resolved | top by composite | 1000 | A1 |
| 5 | ResolvedOnly-500 | resolved | random | 500 | A1 |
| 6 | ResolvedOnly-1000 | resolved | random | 1000 | A1 |
| 7 | BottomQ-500 | resolved | bottom by composite | 500 | A1 |
| 8 | Ablation-NoEfficiency-500 | resolved | rank by Style only | 500 | A2 |
| 9 | Ablation-NoStyle-500 | resolved | rank by Efficiency only | 500 | A2 |
| 10 | Ablation-NoB2-500 | resolved | Efficiency = B3 only | 500 | A3 |
| 11 | Ablation-NoB3-500 | resolved | Efficiency = B2 only | 500 | A3 |
| 12 | Ablation-NoC2-500 | resolved | Style = C3 only | 500 | A3 |
| 13 | Ablation-NoC3-500 | resolved | Style = C2 only | 500 | A3 |
| 14 | Random-2000 | full | random | 2000 | B |
| 15 | TopQ-2000 | resolved | top by composite | 2000 | B |
| 16 | B2Only-Top500 | resolved | top by B2 alone | 500 | C |

### 2.2 Research Questions per Block

#### Block A1 — Data scale and selection strategy (7 runs)

| Comparison | Research question |
|------------|-------------------|
| exp1 vs exp5 | **Does the gate help?** Random from full pool vs random from resolved pool. |
| exp5 vs exp3 | **Does scoring help?** Random in resolved pool vs top of resolved pool. |
| exp1 -> exp2, exp3 -> exp4, exp5 -> exp6 | **Scaling**: how much does each strategy gain from 500 -> 1000? |
| exp3 vs exp7 | **Sanity check**: best vs worst — does the score actually correlate with downstream quality? |

#### Block A2 — Group-level ablations (2 runs)

| Comparison | Research question |
|------------|-------------------|
| exp8 vs exp9 vs exp3 | **Which group matters more, Efficiency or Style?** Single-group vs combined. |

#### Block A3 — Sub-dimension ablations (4 runs)

| Comparison | Research question |
|------------|-------------------|
| exp10 vs exp11 vs exp3 | **Inside Efficiency**: error-retry cycles vs step-count ratio. |
| exp12 vs exp13 vs exp3 | **Inside Style**: action diversity vs observation utilization. |

#### Block B — Scale extension (2 runs)

| Comparison | Research question |
|------------|-------------------|
| exp2 -> exp14, exp4 -> exp15 | **Does the quality vs quantity gap close as data scales?** Compare 1000 -> 2000 for both Random and TopQ. |
| exp14 vs exp15 | **Does TopQ still beat Random at 2000?** Confirms that the scoring advantage is not an artefact of small-sample variance. |

#### Block C — Single-dimension ranking (1 run)

| Comparison | Research question |
|------------|-------------------|
| exp16 vs exp3 | **Is the composite score necessary, or is ranking by the single most discriminating dimension (B2) enough?** Stress-tests whether the multi-dim composite adds value over its strongest constituent. |

### 2.3 Reusable Prior Runs

| Experiment | Reusable | Reason |
|------------|----------|--------|
| baseline | yes | no fine-tuning |
| Random-500 (exp1) | yes | random sampling is independent of the scoring system |
| Random-1000 (exp2) | yes | same as above |
| All others | no | new scoring formula -> different selected samples |

**New training runs needed: 14 (11 in Block A, 2 in Block B, 1 in Block C).**

---

## 3. Evaluation Plan

### 3.1 Perplexity (Cross-Entropy Loss)

Average cross-entropy loss over assistant tokens, computed on three independent test sets.

| Test set | Size | Source |
|----------|------|--------|
| Gold | 200 | Trajectories with the highest composite score under the new scheme. |
| Random | 200 | Random sample. |
| Low-Q | 200 | Trajectories with the lowest composite score under the new scheme. |

> **Note**: Test sets must also be rebuilt under the new scoring system to ensure that Gold / Low-Q reflect the current notion of quality.

### 3.2 Expected Validation Pattern

All trained models should exhibit a **Gold < Random < Low-Q** loss gradient, validating that the scoring system tracks downstream quality.

---

## 4. Paper Storyline

```
Layer 1: Does fine-tuning itself help?       baseline vs any fine-tuned model
Layer 2: Does the gate help?                 Random-500 vs ResolvedOnly-500
Layer 3: Does scoring help?                  ResolvedOnly-500 vs TopQ-500
Layer 4: Scale vs quality?                   500 -> 1000 -> 2000 scaling curve (Block B)
Layer 5: Which group matters more?           EfficiencyOnly vs StyleOnly vs TopQ
Layer 6: Which sub-dimension matters most?   sub-dim ablations (B2/B3/C2/C3)
Layer 7: Is the composite necessary?         B2Only-Top500 vs TopQ-500 (Block C)
Validation: Is the scoring system valid?     TopQ vs BottomQ + test-set quality gradient
```

---

## 5. Design Decision Log

| Decision | Choice | Alternative | Rationale |
|----------|--------|-------------|-----------|
| Truncation ratio | gate (filter) | scoring dimension | std ~ 0, no discriminating power |
| Outcome success | gate (stratification) | continuous scoring dimension | binary variable should not be averaged with continuous ones |
| B1 / C1 | drop from composite | rank-normalize them | rank normalization amplifies noise; dropping them is more honest |
| C3 file matching | basename match | full-path match | agents often drop the path prefix; full-path matching makes C3 systematically too low |
| Aggregation | hierarchical mean, equal weights | weighted average / learned weights | equal weights is the principled default; weight differences are exposed indirectly through ablations |
| Data sizes | 500 / 1000 (Block A) + 2000 (Block B) | 5000+ | GPU budget |
