"""
Visualization module for trajectory analysis results.
Generates charts and summary tables saved to output/.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for M2 Mac
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from .config import OUTPUT_ROOT

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.1)

# v3 sub-dimension column names and display labels
V3_SUB_DIMS = [
    ("b1_redundant_commands",      "B1: Redundant Commands"),
    ("b2_error_retry",             "B2: Error-Retry Cycles"),
    ("b3_step_count_ratio",        "B3: Step Count Ratio"),
    ("c1_observation_cleanliness", "C1: Obs Cleanliness"),
    ("c2_action_diversity",        "C2: Action Diversity"),
    ("c3_observation_utilization", "C3: Obs Utilization"),
]


def plot_token_distribution(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Histogram of total tokens per trajectory."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["total_tokens"], bins=80, color="#4C72B0", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Trajectory Count")
    ax.set_title("Token Count Distribution")
    ax.axvline(df["total_tokens"].median(), color="red", ls="--", label=f'Median: {df["total_tokens"].median():,.0f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "token_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved token_distribution.png")


def plot_turn_distribution(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Histogram of assistant turn counts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(df["assistant_turns"], bins=50, color="#55A868", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Assistant Turns")
    ax.set_ylabel("Trajectory Count")
    ax.set_title("Turn Count Distribution")
    ax.axvline(df["assistant_turns"].median(), color="red", ls="--", label=f'Median: {df["assistant_turns"].median():.0f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "turn_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved turn_distribution.png")


def plot_tool_call_distribution(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Bar chart of tool call types aggregated across all trajectories."""
    from collections import Counter
    agg = Counter()
    for counts_dict in df["tool_call_counts"]:
        if isinstance(counts_dict, dict):
            agg.update(counts_dict)

    if not agg:
        logger.warning("No tool call data to plot.")
        return

    tools = pd.Series(agg).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(10, max(4, len(tools) * 0.5)))
    tools.plot.barh(ax=ax, color="#C44E52", edgecolor="white")
    ax.set_xlabel("Total Calls")
    ax.set_title("Tool Call Distribution (All Trajectories)")
    for i, (v, name) in enumerate(zip(tools.values, tools.index)):
        ax.text(v + agg.most_common(1)[0][1] * 0.01, i, f"{v:,}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "tool_call_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved tool_call_distribution.png")


def plot_outcome_pie(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Pie chart: submit vs error/timeout, and resolved breakdown."""
    submit_resolved = ((df["ends_with_submit"]) & (df["resolved"] == 1)).sum()
    submit_failed = ((df["ends_with_submit"]) & (df["resolved"] == 0)).sum()
    error_timeout = df["is_error_or_timeout"].sum()

    labels = ["Submit + Resolved", "Submit + Failed", "Error / Timeout"]
    sizes = [submit_resolved, submit_failed, error_timeout]
    colors = ["#55A868", "#F0E442", "#C44E52"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", colors=colors,
           startangle=140, textprops={"fontsize": 11})
    ax.set_title("Trajectory Outcome Distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "outcome_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved outcome_distribution.png")


def plot_repo_distribution(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT, top_n: int = 30):
    """Bar chart of top N most frequent repositories."""
    repo_counts = df["repo"].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(12, max(6, top_n * 0.35)))
    repo_counts.sort_values().plot.barh(ax=ax, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Trajectory Count")
    ax.set_title(f"Top {top_n} Repositories by Trajectory Count")
    fig.tight_layout()
    fig.savefig(out_dir / "repo_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved repo_distribution.png")


def plot_observation_length(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Histogram of average observation lengths."""
    fig, ax = plt.subplots(figsize=(10, 5))
    valid = df["avg_observation_length"][df["avg_observation_length"] > 0]
    ax.hist(valid, bins=80, color="#8172B2", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Avg Observation Length (chars)")
    ax.set_ylabel("Trajectory Count")
    ax.set_title("Average Tool Output Length Distribution")
    ax.axvline(valid.median(), color="red", ls="--", label=f"Median: {valid.median():,.0f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "observation_length_distribution.png", dpi=150)
    plt.close(fig)
    logger.info("Saved observation_length_distribution.png")


def plot_quality_scores(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Distribution of v1 composite quality scores Q(t)."""
    if "composite_Q" not in df.columns:
        logger.warning("No composite_Q column found, skipping quality score plot.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    dims = ["truncation_ratio", "outcome_success", "step_efficiency",
            "observation_noise", "action_diversity", "composite_Q"]
    titles = ["Truncation Ratio", "Outcome Success", "Step Efficiency",
              "Observation Noise (inv)", "Action Diversity", "Composite Q(t)"]

    for ax, dim, title in zip(axes.flat, dims, titles):
        if dim in df.columns:
            ax.hist(df[dim].dropna(), bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.set_title(title)
            ax.axvline(df[dim].median(), color="red", ls="--", alpha=0.7)
    fig.suptitle("v1 Quality Score Distributions", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_scores.png", dpi=150)
    plt.close(fig)
    logger.info("Saved quality_scores.png")


def plot_v3_score_distributions(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """
    Plot distributions of all v3 sub-dimensions and the composite score.

    Shows 8 panels: B1, B2, B3, C1, C2, C3, Efficiency, Style + Composite.
    Only includes rows where passes_gate == True (if column exists).

    Prints a variance warning for any sub-dim with std < 0.05.
    """
    # Filter to scored (gate-passing) trajectories only
    if "passes_gate" in df.columns:
        scored = df[df["passes_gate"]].copy()
    elif "composite_score" in df.columns:
        scored = df[df["composite_score"].notna()].copy()
    else:
        logger.warning("plot_v3_score_distributions: no gate or score column found.")
        scored = df.copy()

    if scored.empty:
        logger.warning("plot_v3_score_distributions: no scored trajectories to plot.")
        return

    # Check variance on each sub-dim
    for col, label in V3_SUB_DIMS:
        if col in scored.columns:
            std = scored[col].std()
            if std < 0.05:
                logger.warning(
                    "LOW VARIANCE — %s (std=%.4f). Check scoring logic.", label, std
                )

    # ── 3×3 grid: 6 sub-dims + efficiency + style + composite ──
    plot_cols = [
        *[col for col, _ in V3_SUB_DIMS],
        "efficiency_score", "style_score", "composite_score",
    ]
    plot_labels = [
        *[lbl for _, lbl in V3_SUB_DIMS],
        "Efficiency (mean B2-B3)", "Style (mean C2-C3)", "Composite Score",
    ]
    colors = [
        "#4C72B0", "#4C72B0", "#4C72B0",   # B dims — blue
        "#55A868", "#55A868", "#55A868",   # C dims — green
        "#C44E52", "#8172B2", "#DD8452",   # group scores — red/purple/orange
    ]

    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    for ax, col, label, color in zip(axes.flat, plot_cols, plot_labels, colors):
        if col not in scored.columns:
            ax.set_visible(False)
            continue
        data = scored[col].dropna()
        ax.hist(data, bins=40, color=color, edgecolor="white", alpha=0.85)
        med = data.median()
        std = data.std()
        ax.axvline(med, color="red", ls="--", lw=1.5, label=f"med={med:.3f}")
        ax.set_title(f"{label}\n(std={std:.3f})", fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    n = len(scored)
    fig.suptitle(
        f"v3 Score Distributions  (n={n:,} resolved, gate-passing trajectories)",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "v3_score_distributions.png", dpi=150)
    plt.close(fig)
    logger.info("Saved v3_score_distributions.png")

    # ── Print variance summary to log ──
    logger.info("─── v3 Sub-dimension variance summary ───")
    for col, label in V3_SUB_DIMS + [
        ("efficiency_score", "Efficiency"),
        ("style_score", "Style"),
        ("composite_score", "Composite"),
    ]:
        if col in scored.columns:
            s = scored[col]
            logger.info(
                "  %-35s  mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                label, s.mean(), s.std(), s.min(), s.max(),
            )


def plot_v3_gate_summary(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """
    Bar chart showing how many trajectories pass / fail each gate condition,
    plus the joint gate pass rate.
    """
    n_total = len(df)

    if "total_tokens" in df.columns:
        from .config import MODEL_CONTEXT_WINDOW, GATE_MIN_TRUNCATION_RATIO
        trunc_ok = (
            df["total_tokens"].clip(upper=MODEL_CONTEXT_WINDOW) / df["total_tokens"].replace(0, 1)
            >= GATE_MIN_TRUNCATION_RATIO
        ).sum()
    else:
        trunc_ok = n_total  # unknown

    resolved_ok = (df["resolved"] == 1).sum() if "resolved" in df.columns else n_total

    passes = df["passes_gate"].sum() if "passes_gate" in df.columns else None

    categories = ["All trajectories", "Trunc ≥ 0.9", "Resolved == 1"]
    counts = [n_total, trunc_ok, resolved_ok]
    if passes is not None:
        categories.append("Gate PASS\n(both conditions)")
        counts.append(passes)

    colors = ["#4C72B0", "#55A868", "#8172B2", "#C44E52"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(categories, counts, color=colors[: len(categories)], edgecolor="white")
    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + n_total * 0.01,
            f"{cnt:,}\n({cnt/n_total*100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Trajectory Count")
    ax.set_title("v3 Gate Filtering Summary")
    ax.set_ylim(0, n_total * 1.15)
    fig.tight_layout()
    fig.savefig(out_dir / "v3_gate_summary.png", dpi=150)
    plt.close(fig)
    logger.info("Saved v3_gate_summary.png")


def generate_summary_table(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Generate and save a summary statistics table."""
    numeric_cols = [
        "total_tokens", "assistant_turns", "total_tool_calls",
        "avg_observation_length", "patch_length",
    ]
    for col in ["composite_Q", "composite_score", "efficiency_score", "style_score"]:
        if col in df.columns:
            numeric_cols.append(col)

    available = [c for c in numeric_cols if c in df.columns]
    summary = df[available].describe().round(4)
    summary.to_csv(out_dir / "summary_statistics.csv")
    logger.info("Saved summary_statistics.csv")
    return summary


def generate_all_plots(df: pd.DataFrame, out_dir: Path = OUTPUT_ROOT):
    """Run all visualization functions."""
    out_dir.mkdir(exist_ok=True)
    logger.info("Generating all plots to %s", out_dir)

    plot_token_distribution(df, out_dir)
    plot_turn_distribution(df, out_dir)
    plot_tool_call_distribution(df, out_dir)
    plot_outcome_pie(df, out_dir)
    plot_repo_distribution(df, out_dir)
    plot_observation_length(df, out_dir)
    plot_quality_scores(df, out_dir)

    # v3 plots
    if "composite_score" in df.columns or "passes_gate" in df.columns:
        plot_v3_gate_summary(df, out_dir)
        plot_v3_score_distributions(df, out_dir)

    summary = generate_summary_table(df, out_dir)
    logger.info("All plots generated successfully.")
    return summary
