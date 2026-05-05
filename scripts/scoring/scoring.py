"""
Quality scoring framework — three versions maintained side by side.

v1 (flat, 5-dim) — legacy, kept for backwards compat:
  Q(t) = 0.30 * TruncRatio
       + 0.25 * Outcome
       + 0.20 * StepEfficiency
       + 0.15 * (1 - NormObsLength)
       + 0.10 * ActionDiversity

v2 (hierarchical, 4 groups) — legacy, kept for backwards compat:
  Q_v2(t) = 0.35 * Correctness  [outcome_success × 0.70 + test_alignment × 0.30]
           + 0.30 * Efficiency   [redundant_actions × 0.50 + step_count_ratio × 0.50]
           + 0.25 * Style        [obs_noise × 0.40 + action_div × 0.40 + reasoning_clarity × 0.20]
           + 0.10 * Completeness [truncation_ratio × 1.00]

v3 (Gate + Efficiency + Style) — CURRENT:
  Gate (filter only, not scored):
    truncation_ratio < 0.9      → discard
    resolved != 1               → not in scoring pool

  Score = 0.5 * mean(B1, B2, B3) + 0.5 * mean(C1, C2, C3)

  Efficiency:
    B1: Redundant Commands   — 1 - redundant_cmds_ratio
    B2: Error-Retry Cycles   — 1 - clip(cycles/B2_MAX_CYCLES, 0, 1)
    B3: Step Count Ratio     — 1 - normalize(clip(steps/median, 0.5, 3.0))

  Style:
    C1: Observation Cleanliness — 1 - obs_noise_char_ratio
    C2: Action Diversity        — Shannon entropy of tool types, normalized
    C3: Observation Utilization — obs_utilization_ratio
"""
import math
import logging
from collections import Counter

import pandas as pd

try:
    from .scoring_config import (
        QUALITY_WEIGHTS,
        HIERARCHY_WEIGHTS,
        SUB_DIM_WEIGHTS,
        MODEL_CONTEXT_WINDOW,
        GATE_MIN_TRUNCATION_RATIO,
        B2_MAX_CYCLES,
        B3_RATIO_CLIP,
    )
    from .analysis import TrajectoryStats
except ImportError:
    from scoring_config import (
        QUALITY_WEIGHTS,
        HIERARCHY_WEIGHTS,
        SUB_DIM_WEIGHTS,
        MODEL_CONTEXT_WINDOW,
        GATE_MIN_TRUNCATION_RATIO,
        B2_MAX_CYCLES,
        B3_RATIO_CLIP,
    )
    from analysis import TrajectoryStats

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  Shared sub-dimension scorers (used by v1/v2)
# ═══════════════════════════════════════════════════════════

def truncation_ratio(stats: TrajectoryStats) -> float:
    """Completeness: proportion of trajectory within context window."""
    if stats.total_tokens == 0:
        return 0.0
    return min(stats.total_tokens, MODEL_CONTEXT_WINDOW) / stats.total_tokens


def outcome_success(stats: TrajectoryStats) -> float:
    """
    Correctness sub-dim: Did the trajectory solve the issue?
      1.0 → resolved=1
      0.5 → submitted but not resolved
      0.0 → error / timeout
    """
    if stats.is_error_or_timeout:
        return 0.0
    if stats.ends_with_submit and stats.resolved == 1:
        return 1.0
    if stats.ends_with_submit:
        return 0.5
    return 0.0


def observation_noise(stats: TrajectoryStats) -> float:
    """v1/v2 Style: inverted normalized average observation length."""
    REFERENCE_MAX = 10_000
    if not stats.observation_lengths:
        return 1.0
    norm = min(1.0, stats.avg_observation_length / REFERENCE_MAX)
    return 1.0 - norm


def action_diversity(stats: TrajectoryStats) -> float:
    """
    Style sub-dim: Shannon entropy of tool type distribution, normalized to [0,1].
    Shared by v1, v2, and v3 (C2).
    """
    if stats.total_tool_calls == 0:
        return 0.0
    counts = list(stats.tool_call_counts.values())
    total = sum(counts)
    if total == 0 or len(counts) <= 1:
        return 0.0
    entropy = -sum((c / total) * math.log2(c / total) for c in counts if c > 0)
    max_entropy = math.log2(len(counts))
    return entropy / max_entropy if max_entropy > 0 else 0.0


# ── v1-only ──────────────────────────────────────────────

def step_efficiency(stats: TrajectoryStats) -> float:
    """v1 Efficiency proxy: unique tool types / total tool calls × 10, capped at 1."""
    if stats.total_tool_calls == 0:
        return 0.0
    unique_types = len(stats.tool_call_counts)
    return min(1.0, unique_types / max(1, stats.total_tool_calls) * 10)


# ── v2-only ──────────────────────────────────────────────

def test_alignment(stats: TrajectoryStats) -> float:
    return stats.test_alignment_score


def redundant_actions(stats: TrajectoryStats) -> float:
    return stats.unique_action_ratio


def step_count_ratio(stats: TrajectoryStats, median_resolved_steps: float) -> float:
    if stats.assistant_turns == 0 or median_resolved_steps <= 0:
        return 0.0
    return min(1.0, median_resolved_steps / stats.assistant_turns)


def reasoning_clarity(stats: TrajectoryStats) -> float:
    return stats.unique_action_ratio


# ═══════════════════════════════════════════════════════════
#  v3 Gate
# ═══════════════════════════════════════════════════════════

def passes_gate(stats: TrajectoryStats) -> bool:
    """
    Return True if this trajectory passes the v3 gate and enters the scoring pool.

    Gate conditions (both must hold):
      1. truncation_ratio >= GATE_MIN_TRUNCATION_RATIO (default 0.9)
      2. resolved == 1

    Trajectories that fail the gate are discarded and should not be scored.
    """
    tr = truncation_ratio(stats)
    return tr >= GATE_MIN_TRUNCATION_RATIO and stats.resolved == 1


# ═══════════════════════════════════════════════════════════
#  v3 Sub-dimension scorers
# ═══════════════════════════════════════════════════════════

def b1_redundant_commands(stats: TrajectoryStats) -> float:
    """
    B1: Efficiency — penalize repeated commands within adjacent window.
    score = 1 - redundant_cmds_ratio
    """
    return 1.0 - stats.redundant_cmds_ratio


def b2_error_retry(stats: TrajectoryStats) -> float:
    """
    B2: Efficiency — penalize error-retry cycles.
    Normalize: cycles / B2_MAX_CYCLES, clip to [0, 1], then invert.
    """
    normalized = min(1.0, stats.error_retry_count / B2_MAX_CYCLES)
    return 1.0 - normalized


def b3_step_count_ratio(
    stats: TrajectoryStats,
    per_task_median: dict[str, float],
) -> float:
    """
    B3: Efficiency — compare this trajectory's step count to the per-task median.

    ratio = assistant_turns / median_steps_for_same_task (resolved only)
    clip ratio to B3_RATIO_CLIP = [0.5, 3.0]
    normalize to [0, 1]:  (clipped - low) / (high - low)
    score = 1 - normalized   (fewer steps relative to median → higher score)

    Falls back to 0.5 (neutral) if no per-task median is available
    (e.g., this is the only resolved trajectory for this task).

    Args:
        per_task_median: dict mapping instance_id → median resolved steps.
                         Build with compute_per_task_medians().
    """
    median = per_task_median.get(stats.instance_id)
    if median is None or median <= 0 or stats.assistant_turns == 0:
        return 0.5   # neutral fallback

    low, high = B3_RATIO_CLIP
    ratio = stats.assistant_turns / median
    clipped = max(low, min(high, ratio))
    normalized = (clipped - low) / (high - low)
    return 1.0 - normalized


def c1_observation_cleanliness(stats: TrajectoryStats) -> float:
    """
    C1: Style — fraction of observation characters that are clean (not errors/tracebacks).
    score = 1 - obs_noise_char_ratio
    """
    return 1.0 - stats.obs_noise_char_ratio


def c2_action_diversity(stats: TrajectoryStats) -> float:
    """
    C2: Style — Shannon entropy of tool type distribution (same as v1/v2 action_diversity).
    """
    return action_diversity(stats)


def c3_observation_utilization(stats: TrajectoryStats) -> float:
    """
    C3: Style — fraction of observation keywords (filenames, error classes) that
    appear in subsequent actions.
    """
    return stats.obs_utilization_ratio


# ═══════════════════════════════════════════════════════════
#  v3 Dataset-level helper
# ═══════════════════════════════════════════════════════════

def compute_per_task_medians(df: pd.DataFrame) -> dict[str, float]:
    """
    Compute the per-task (instance_id) median assistant_turns for resolved trajectories.

    Only resolved trajectories (resolved == 1 AND passes_gate) contribute to the median.
    Trajectories whose instance_id has no resolved peers get a None entry → B3 returns 0.5.

    Args:
        df: DataFrame with columns: instance_id, assistant_turns, resolved,
            total_tokens (to recompute gate status without full stats objects).

    Returns:
        dict mapping instance_id → median assistant_turns (float).
    """
    # Use the gate flag if pre-computed, otherwise approximate via resolved column.
    if "passes_gate" in df.columns:
        resolved_df = df[df["passes_gate"]]
    else:
        resolved_df = df[df["resolved"] == 1]

    if resolved_df.empty:
        logger.warning("compute_per_task_medians: no resolved trajectories found.")
        return {}

    medians = (
        resolved_df.groupby("instance_id")["assistant_turns"]
        .median()
        .to_dict()
    )
    logger.info(
        "compute_per_task_medians: %d unique tasks with resolved trajectories.",
        len(medians),
    )
    return medians


# ═══════════════════════════════════════════════════════════
#  v3 Composite scorer
# ═══════════════════════════════════════════════════════════

def new_quality_score(
    stats: TrajectoryStats,
    per_task_median: dict[str, float],
) -> dict:
    """
    Compute v3 quality score for a single trajectory.

    Should only be called on trajectories that pass passes_gate().
    Calling on non-passing trajectories is allowed but results are meaningless.

    Active scoring formula (4-dim, after variance analysis):
        Efficiency = mean(B2, B3)
        Style      = mean(C2, C3)
        Composite  = 0.5 * Efficiency + 0.5 * Style

    B1 (Redundant Commands) and C1 (Observation Cleanliness) are computed and
    returned for transparency but excluded from the composite because empirical
    variance analysis showed they lack discriminating power on this dataset
    (std < 0.05): SWE-trajectory agents are already homogeneous on these dimensions.

    Args:
        stats: trajectory statistics from analyze_trajectory().
        per_task_median: dict from compute_per_task_medians().

    Returns dict with all 6 sub-dim scores, group scores, and composite_score.
    """
    b1 = b1_redundant_commands(stats)
    b2 = b2_error_retry(stats)
    b3 = b3_step_count_ratio(stats, per_task_median)
    c1 = c1_observation_cleanliness(stats)
    c2 = c2_action_diversity(stats)
    c3 = c3_observation_utilization(stats)

    # Only B2+B3 and C2+C3 enter the composite (B1, C1 low-variance, excluded)
    efficiency = (b2 + b3) / 2.0
    style = (c2 + c3) / 2.0
    composite = 0.5 * efficiency + 0.5 * style

    return {
        # All 6 sub-dims stored for reporting / ablation
        "b1_redundant_commands":      b1,   # excluded from composite (low variance)
        "b2_error_retry":             b2,
        "b3_step_count_ratio":        b3,
        "c1_observation_cleanliness": c1,   # excluded from composite (low variance)
        "c2_action_diversity":        c2,
        "c3_observation_utilization": c3,
        # Group scores (based on active dims only)
        "efficiency_score":           efficiency,   # mean(B2, B3)
        "style_score":                style,        # mean(C2, C3)
        "composite_score":            composite,    # 0.5*E + 0.5*S
    }


# ═══════════════════════════════════════════════════════════
#  v1 composite (backwards compatible)
# ═══════════════════════════════════════════════════════════

def composite_quality_score(stats: TrajectoryStats) -> dict:
    """
    Compute v1 flat composite score Q(t) and all sub-scores.
    Returns dict with individual dimension scores and composite_Q.
    """
    w = QUALITY_WEIGHTS
    scores = {
        "truncation_ratio":  truncation_ratio(stats),
        "outcome_success":   outcome_success(stats),
        "step_efficiency":   step_efficiency(stats),
        "observation_noise": observation_noise(stats),
        "action_diversity":  action_diversity(stats),
    }
    scores["composite_Q"] = sum(w[dim] * scores[dim] for dim in w)
    return scores


# ═══════════════════════════════════════════════════════════
#  v2 hierarchical composite (backwards compatible)
# ═══════════════════════════════════════════════════════════

def hierarchical_quality_score(
    stats: TrajectoryStats,
    median_resolved_steps: float,
) -> dict:
    """
    Compute v2 hierarchical composite score Q_v2(t).
    """
    sub_scores = {
        "outcome_success":   outcome_success(stats),
        "test_alignment":    test_alignment(stats),
        "redundant_actions": redundant_actions(stats),
        "step_count_ratio":  step_count_ratio(stats, median_resolved_steps),
        "observation_noise": observation_noise(stats),
        "action_diversity":  action_diversity(stats),
        "reasoning_clarity": reasoning_clarity(stats),
        "truncation_ratio":  truncation_ratio(stats),
    }

    group_scores: dict[str, float] = {g: 0.0 for g in HIERARCHY_WEIGHTS}
    for dim, (group, w_within) in SUB_DIM_WEIGHTS.items():
        group_scores[group] += w_within * sub_scores[dim]

    composite_v2 = sum(
        HIERARCHY_WEIGHTS[g] * group_scores[g]
        for g in HIERARCHY_WEIGHTS
    )

    return {
        **sub_scores,
        "correctness_score":  group_scores["correctness"],
        "efficiency_score":   group_scores["efficiency"],
        "style_score":        group_scores["style"],
        "completeness_score": group_scores["completeness"],
        "composite_Q_v2":     composite_v2,
    }
