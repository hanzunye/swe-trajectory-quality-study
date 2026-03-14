"""
Subset construction strategies (Phase 2, §3.3).

Constructs experimental subsets from Quality_Scoring v3 output:

  Two data pools:
    df_all    — ALL trajectories (67k), from trajectory_stats.jsonl
    df_scored — RESOLVED trajectories only (32k), from trajectory_scored_v3.csv
                (passes gate: resolved=1 AND truncation_ratio >= 0.9)

  Subsets built:
    Random-500 / Random-1000          random from df_all (all trajectories)
    TopQ-500 / TopQ-1000              top N by composite_score (df_scored)
    ResolvedOnly-500 / -1000          random from df_scored (resolved pool)
    BottomQ-500                       bottom 500 by composite_score (sanity check)
    Ablation-NoEfficiency-500         score = Style only  (no Efficiency term)
    Ablation-NoStyle-500              score = Efficiency only  (no Style term)
    Ablation-NoB2-500                 Efficiency = B3 alone (drop B2)
    Ablation-NoB3-500                 Efficiency = B2 alone (drop B3)
    Ablation-NoC2-500                 Style = C3 alone      (drop C2)
    Ablation-NoC3-500                 Style = C2 alone      (drop C3)
"""
import logging
from dataclasses import dataclass, field

import pandas as pd

from .config import RANDOM_SEED, SCORE_COL

logger = logging.getLogger(__name__)


@dataclass
class SubsetResult:
    """Container for a constructed subset with metadata."""
    name: str
    description: str
    purpose: str
    selection_criteria: str
    expected_size: int
    actual_size: int
    df: pd.DataFrame
    stats: dict = field(default_factory=dict)

    def summary(self) -> str:
        """Human-readable one-line summary."""
        mean_q = self.stats.get("mean_Q", 0)
        return (
            f"{self.name:<35s} {self.actual_size:>5d}/{self.expected_size} trajectories"
            f"  |  score mean={mean_q:.4f}"
        )


# ── Statistics ──────────────────────────────────────────────────────────────

def _compute_subset_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics for a subset DataFrame."""
    stats = {}

    # Primary quality score: prefer v3 composite_score, fall back to v1 composite_Q
    for col in ("composite_score", "composite_Q"):
        if col in df.columns:
            s = df[col].dropna()
            if len(s) > 0:
                stats["mean_Q"]   = float(s.mean())
                stats["median_Q"] = float(s.median())
                stats["std_Q"]    = float(s.std())
                stats["min_Q"]    = float(s.min())
                stats["max_Q"]    = float(s.max())
            break

    if "total_tokens" in df.columns:
        stats["mean_tokens"]   = float(df["total_tokens"].mean())
        stats["median_tokens"] = float(df["total_tokens"].median())
    if "assistant_turns" in df.columns:
        stats["mean_turns"]    = float(df["assistant_turns"].mean())
    if "ends_with_submit" in df.columns:
        stats["submit_rate"]   = float(df["ends_with_submit"].mean())
    if "resolved" in df.columns:
        stats["resolved_rate"] = float((df["resolved"] == 1).mean())
    if "repo" in df.columns:
        stats["unique_repos"]  = int(df["repo"].nunique())

    # v3 group scores
    for col in ("efficiency_score", "style_score"):
        if col in df.columns:
            stats[f"mean_{col}"] = float(df[col].mean())

    # v3 sub-dimension means
    for dim in ("b2_error_retry", "b3_step_count_ratio",
                "c2_action_diversity", "c3_observation_utilization"):
        if dim in df.columns:
            stats[f"mean_{dim}"] = float(df[dim].mean())

    return stats


# ── Core Builders ────────────────────────────────────────────────────────────

def build_random(df_all: pd.DataFrame, size: int, name: str) -> SubsetResult:
    """
    Random-N: Uniformly random sample from ALL trajectories (no filtering).
    Baseline for comparison — includes both resolved and unresolved trajectories.
    """
    actual_size = min(size, len(df_all))
    subset = df_all.sample(n=actual_size, random_state=RANDOM_SEED)
    stats = _compute_subset_stats(subset)

    return SubsetResult(
        name=name,
        description=f"{actual_size} random trajectories from all {len(df_all):,} (no filtering)",
        purpose="Baseline" if size <= 500 else "Scale baseline",
        selection_criteria="Random sample (all trajectories)",
        expected_size=size,
        actual_size=actual_size,
        df=subset,
        stats=stats,
    )


def build_topq(df_scored: pd.DataFrame, size: int, name: str) -> SubsetResult:
    """
    TopQ-N: Top N trajectories by composite_score from the resolved pool.
    Validates the quality filtering approach.
    """
    sorted_df = df_scored.sort_values(SCORE_COL, ascending=False)
    actual_size = min(size, len(sorted_df))
    subset = sorted_df.head(actual_size)
    stats = _compute_subset_stats(subset)

    return SubsetResult(
        name=name,
        description=f"Top {actual_size} by composite_score from resolved pool ({len(df_scored):,})",
        purpose="Quality filter validation" if size <= 500 else "Scale quality filter",
        selection_criteria=f"Top composite_score (resolved pool)",
        expected_size=size,
        actual_size=actual_size,
        df=subset,
        stats=stats,
    )


def build_resolvedonly(df_scored: pd.DataFrame, size: int, name: str) -> SubsetResult:
    """
    ResolvedOnly-N: Random N from the resolved pool (gate-passing trajectories).
    Tests whether resolved-only filtering alone is sufficient, without quality ranking.
    """
    actual_size = min(size, len(df_scored))
    subset = df_scored.sample(n=actual_size, random_state=RANDOM_SEED)
    stats = _compute_subset_stats(subset)

    return SubsetResult(
        name=name,
        description=f"{actual_size} random from resolved pool ({len(df_scored):,} gate-passing)",
        purpose="Resolved-only filter baseline",
        selection_criteria="Random sample (resolved pool only)",
        expected_size=size,
        actual_size=actual_size,
        df=subset,
        stats=stats,
    )


def build_bottomq(df_scored: pd.DataFrame, size: int = 500) -> SubsetResult:
    """
    BottomQ-500: Bottom 500 by composite_score from the resolved pool.
    Sanity-check subset — expected to perform worst in fine-tuning evaluation.
    """
    sorted_df = df_scored.sort_values(SCORE_COL, ascending=True)
    actual_size = min(size, len(sorted_df))
    subset = sorted_df.head(actual_size)
    stats = _compute_subset_stats(subset)

    return SubsetResult(
        name="BottomQ-500",
        description=f"Bottom {actual_size} by composite_score (sanity check)",
        purpose="Sanity check (worst quality)",
        selection_criteria="Bottom composite_score (resolved pool)",
        expected_size=size,
        actual_size=actual_size,
        df=subset,
        stats=stats,
    )


# ── Ablation Builders ────────────────────────────────────────────────────────

def _build_ablation(
    df_scored: pd.DataFrame,
    name: str,
    description: str,
    score_fn,
    size: int = 500,
) -> SubsetResult:
    """
    Generic ablation builder.
    score_fn: callable(df) → pd.Series — computes the ablated ranking score.
    """
    df_copy = df_scored.copy()
    df_copy["_ablation_score"] = score_fn(df_copy)

    sorted_df = df_copy.sort_values("_ablation_score", ascending=False)
    actual_size = min(size, len(sorted_df))
    subset = sorted_df.head(actual_size).copy()

    stats = _compute_subset_stats(subset)
    stats["mean_ablation_score"]   = float(subset["_ablation_score"].mean())
    stats["median_ablation_score"] = float(subset["_ablation_score"].median())

    subset = subset.drop(columns=["_ablation_score"])

    return SubsetResult(
        name=name,
        description=description,
        purpose="Ablation study",
        selection_criteria=description,
        expected_size=size,
        actual_size=actual_size,
        df=subset,
        stats=stats,
    )


def build_ablation_no_efficiency(df_scored: pd.DataFrame, size: int = 500) -> SubsetResult:
    """
    Ablation-NoEfficiency-500: Style only.
    Score = style_score = mean(C2, C3). No Efficiency term.
    """
    return _build_ablation(
        df_scored, "Ablation-NoEfficiency-500",
        "Top 500 by Style score only (no Efficiency component)",
        lambda d: d["style_score"],
        size,
    )


def build_ablation_no_style(df_scored: pd.DataFrame, size: int = 500) -> SubsetResult:
    """
    Ablation-NoStyle-500: Efficiency only.
    Score = efficiency_score = mean(B2, B3). No Style term.
    """
    return _build_ablation(
        df_scored, "Ablation-NoStyle-500",
        "Top 500 by Efficiency score only (no Style component)",
        lambda d: d["efficiency_score"],
        size,
    )


def build_ablation_no_b2(df_scored: pd.DataFrame, size: int = 500) -> SubsetResult:
    """
    Ablation-NoB2-500: Efficiency = B3 alone (drop B2=error_retry).
    Score = 0.5 * b3_step_count_ratio + 0.5 * style_score
    """
    return _build_ablation(
        df_scored, "Ablation-NoB2-500",
        "Top 500 with Efficiency = B3 alone (drop B2 error_retry)",
        lambda d: 0.5 * d["b3_step_count_ratio"] + 0.5 * d["style_score"],
        size,
    )


def build_ablation_no_b3(df_scored: pd.DataFrame, size: int = 500) -> SubsetResult:
    """
    Ablation-NoB3-500: Efficiency = B2 alone (drop B3=step_count_ratio).
    Score = 0.5 * b2_error_retry + 0.5 * style_score
    """
    return _build_ablation(
        df_scored, "Ablation-NoB3-500",
        "Top 500 with Efficiency = B2 alone (drop B3 step_count_ratio)",
        lambda d: 0.5 * d["b2_error_retry"] + 0.5 * d["style_score"],
        size,
    )


def build_ablation_no_c2(df_scored: pd.DataFrame, size: int = 500) -> SubsetResult:
    """
    Ablation-NoC2-500: Style = C3 alone (drop C2=action_diversity).
    Score = 0.5 * efficiency_score + 0.5 * c3_observation_utilization
    """
    return _build_ablation(
        df_scored, "Ablation-NoC2-500",
        "Top 500 with Style = C3 alone (drop C2 action_diversity)",
        lambda d: 0.5 * d["efficiency_score"] + 0.5 * d["c3_observation_utilization"],
        size,
    )


def build_ablation_no_c3(df_scored: pd.DataFrame, size: int = 500) -> SubsetResult:
    """
    Ablation-NoC3-500: Style = C2 alone (drop C3=observation_utilization).
    Score = 0.5 * efficiency_score + 0.5 * c2_action_diversity
    """
    return _build_ablation(
        df_scored, "Ablation-NoC3-500",
        "Top 500 with Style = C2 alone (drop C3 observation_utilization)",
        lambda d: 0.5 * d["efficiency_score"] + 0.5 * d["c2_action_diversity"],
        size,
    )


# ── Master Builder ───────────────────────────────────────────────────────────

def build_all_subsets(
    df_all: pd.DataFrame,
    df_scored: pd.DataFrame,
) -> list[SubsetResult]:
    """
    Construct all experimental subsets.

    Args:
        df_all:    ALL trajectories (resolved + unresolved), from trajectory_stats.jsonl
        df_scored: Resolved gate-passing trajectories with v3 scores, from trajectory_scored_v3.csv

    Returns:
        List of all SubsetResult objects.
    """
    logger.info(
        "Building all subsets: %d total trajectories, %d resolved/scored.",
        len(df_all), len(df_scored),
    )
    results = []

    # ── Random (from full pool) ─────────────────────────────────────────────
    logger.info("Building Random-500...")
    results.append(build_random(df_all, size=500, name="Random-500"))

    logger.info("Building Random-1000...")
    results.append(build_random(df_all, size=1000, name="Random-1000"))

    # ── Quality-ranked (from resolved pool) ─────────────────────────────────
    logger.info("Building TopQ-500...")
    results.append(build_topq(df_scored, size=500, name="TopQ-500"))

    logger.info("Building TopQ-1000...")
    results.append(build_topq(df_scored, size=1000, name="TopQ-1000"))

    # ── Resolved-only random (from resolved pool) ────────────────────────────
    logger.info("Building ResolvedOnly-500...")
    results.append(build_resolvedonly(df_scored, size=500, name="ResolvedOnly-500"))

    logger.info("Building ResolvedOnly-1000...")
    results.append(build_resolvedonly(df_scored, size=1000, name="ResolvedOnly-1000"))

    # ── Bottom quality (sanity check) ────────────────────────────────────────
    logger.info("Building BottomQ-500...")
    results.append(build_bottomq(df_scored, size=500))

    # ── Ablation studies ─────────────────────────────────────────────────────
    logger.info("Building Ablation-NoEfficiency-500...")
    results.append(build_ablation_no_efficiency(df_scored))

    logger.info("Building Ablation-NoStyle-500...")
    results.append(build_ablation_no_style(df_scored))

    logger.info("Building Ablation-NoB2-500...")
    results.append(build_ablation_no_b2(df_scored))

    logger.info("Building Ablation-NoB3-500...")
    results.append(build_ablation_no_b3(df_scored))

    logger.info("Building Ablation-NoC2-500...")
    results.append(build_ablation_no_c2(df_scored))

    logger.info("Building Ablation-NoC3-500...")
    results.append(build_ablation_no_c3(df_scored))

    logger.info(
        "All %d subsets built (4 standard + 2 resolved-only + 1 bottom + 6 ablation).",
        len(results),
    )
    return results
