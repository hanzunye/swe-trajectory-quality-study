"""
Visualization module for Phase 2: Subset Comparison.

Generates charts comparing quality distributions, composition,
and key metrics across all constructed subsets.
"""
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from subset_config import OUTPUT_ROOT

logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", font_scale=1.1)

# Color palette — standard subsets
SUBSET_COLORS = {
    "Random-500":          "#95a5a6",
    "Random-1000":         "#7f8c8d",
    "TopQ-500":            "#2ecc71",
    "TopQ-1000":           "#27ae60",
    "ResolvedOnly-500":    "#3498db",
    "ResolvedOnly-1000":   "#2980b9",
    "BottomQ-500":         "#e74c3c",
}
ABLATION_COLOR = "#9b59b6"
DEFAULT_COLOR  = "#4C72B0"


def _get_color(name: str) -> str:
    if name in SUBSET_COLORS:
        return SUBSET_COLORS[name]
    if name.startswith("Ablation-"):
        return ABLATION_COLOR
    return DEFAULT_COLOR


def _is_ablation(name: str) -> bool:
    return name.startswith("Ablation-")


# ── Plot Functions ────────────────────────────────────────────────────────────

def plot_composite_q_comparison(subsets: list, out_dir: Path = OUTPUT_ROOT):
    """Box plot of composite score distribution for standard (non-ablation) subsets."""
    standard = [s for s in subsets if not _is_ablation(s.name)]
    if not standard:
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    data = []
    score_col = None
    for s in standard:
        for col in ("composite_score", "composite_Q"):
            if col in s.df.columns:
                score_col = col
                break
        if score_col:
            for val in s.df[score_col].dropna():
                data.append({"Subset": s.name, "Score": val})

    if not data:
        plt.close(fig)
        return

    plot_df = pd.DataFrame(data)
    color_map = {s.name: _get_color(s.name) for s in standard}
    sns.boxplot(data=plot_df, x="Subset", y="Score",
                hue="Subset", palette=color_map, legend=False, ax=ax)
    ax.set_title("Composite Score Distribution by Subset")
    ax.set_ylabel("composite_score (v3)")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    fig.tight_layout()
    fig.savefig(out_dir / "subset_q_comparison_box.png", dpi=150)
    plt.close(fig)
    logger.info("Saved subset_q_comparison_box.png")


def plot_score_group_comparison(subsets: list, out_dir: Path = OUTPUT_ROOT):
    """
    Grouped bar chart: mean efficiency_score and style_score for each standard subset.
    Shows which subsets excel on each scoring group.
    """
    standard = [s for s in subsets
                if not _is_ablation(s.name)
                and "mean_efficiency_score" in s.stats]
    if not standard:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    groups = [("efficiency_score", "Efficiency (B2+B3)"),
              ("style_score",      "Style (C2+C3)")]
    x = np.arange(len(groups))
    n = len(standard)
    width = 0.8 / n

    for i, s in enumerate(standard):
        means = [s.stats.get(f"mean_{g}", 0) for g, _ in groups]
        offset = (i - n / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=s.name,
                      color=_get_color(s.name), edgecolor="white", alpha=0.9)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in groups])
    ax.set_ylabel("Mean Score")
    ax.set_title("Efficiency vs Style Score by Subset")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 1.10)
    fig.tight_layout()
    fig.savefig(out_dir / "subset_score_group_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved subset_score_group_comparison.png")


def plot_subset_sizes(subsets: list, out_dir: Path = OUTPUT_ROOT):
    """Bar chart: actual vs expected size for non-ablation subsets."""
    filtered = [s for s in subsets if not _is_ablation(s.name)]
    if not filtered:
        return

    names    = [s.name for s in filtered]
    expected = [s.expected_size for s in filtered]
    actual   = [s.actual_size   for s in filtered]

    fig, ax = plt.subplots(figsize=(13, 6))
    x     = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, expected, width, label="Expected", color="#bdc3c7", edgecolor="white")
    ax.bar(x + width / 2, actual,   width, label="Actual",
           color=[_get_color(n) for n in names], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Number of Trajectories")
    ax.set_title("Subset Sizes: Expected vs Actual")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "subset_sizes.png", dpi=150)
    plt.close(fig)
    logger.info("Saved subset_sizes.png")


def plot_token_distribution_by_subset(subsets: list, out_dir: Path = OUTPUT_ROOT):
    """Overlapping token count histograms for standard subsets."""
    standard = [s for s in subsets if not _is_ablation(s.name)]
    if not standard:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for s in standard:
        if "total_tokens" in s.df.columns and len(s.df) > 0:
            ax.hist(s.df["total_tokens"], bins=50, alpha=0.4,
                    label=f"{s.name} (n={len(s.df)})", color=_get_color(s.name))
    ax.set_xlabel("Total Tokens")
    ax.set_ylabel("Count")
    ax.set_title("Token Distribution Across Subsets")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "subset_token_distributions.png", dpi=150)
    plt.close(fig)
    logger.info("Saved subset_token_distributions.png")


def plot_outcome_by_subset(subsets: list, out_dir: Path = OUTPUT_ROOT):
    """
    Stacked bar chart of outcome composition
    (submit+resolved / submit+unresolved / error) per standard subset.
    """
    standard = [s for s in subsets if not _is_ablation(s.name)]
    if not standard:
        return

    names, resolved_pcts, failed_pcts, error_pcts = [], [], [], []
    for s in standard:
        n = len(s.df)
        if n == 0:
            continue
        names.append(s.name)
        sr = ((s.df["ends_with_submit"]) & (s.df["resolved"] == 1)).sum() / n
        sf = ((s.df["ends_with_submit"]) & (s.df["resolved"] == 0)).sum() / n
        er = s.df["is_error_or_timeout"].sum() / n if "is_error_or_timeout" in s.df.columns else 0
        resolved_pcts.append(sr * 100)
        failed_pcts.append(sf * 100)
        error_pcts.append(er * 100)

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(names))
    ax.bar(x, resolved_pcts, label="Submit + Resolved", color="#55A868", edgecolor="white")
    ax.bar(x, failed_pcts, bottom=resolved_pcts, label="Submit + Unresolved",
           color="#F0E442", edgecolor="white")
    bottoms = [r + f for r, f in zip(resolved_pcts, failed_pcts)]
    ax.bar(x, error_pcts, bottom=bottoms, label="Error / Timeout",
           color="#C44E52", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Percentage (%)")
    ax.set_title("Outcome Composition by Subset")
    ax.legend()
    ax.set_ylim(0, 108)
    fig.tight_layout()
    fig.savefig(out_dir / "subset_outcome_composition.png", dpi=150)
    plt.close(fig)
    logger.info("Saved subset_outcome_composition.png")


def plot_ablation_comparison(subsets: list, out_dir: Path = OUTPUT_ROOT):
    """
    Bar chart comparing ablation subsets:
    mean composite_score vs mean ablation_score for each variant,
    with TopQ-500 as reference.
    """
    ablation = [s for s in subsets if _is_ablation(s.name)]
    if not ablation:
        return

    topq_ref  = [s for s in subsets if s.name == "TopQ-500"]
    topq_mean = topq_ref[0].stats.get("mean_Q", 0) if topq_ref else 0

    names, mean_orig, mean_abl = [], [], []
    for s in ablation:
        short = s.name.replace("Ablation-", "")
        names.append(short)
        mean_orig.append(s.stats.get("mean_Q", 0))
        mean_abl.append(s.stats.get("mean_ablation_score", 0))

    fig, ax = plt.subplots(figsize=(13, 6))
    x     = np.arange(len(names))
    width = 0.35
    ax.bar(x - width / 2, mean_orig, width, label="Original composite_score",
           color="#4C72B0", edgecolor="white")
    ax.bar(x + width / 2, mean_abl,  width, label="Ablated score",
           color=ABLATION_COLOR, edgecolor="white")

    if topq_mean > 0:
        ax.axhline(topq_mean, color="red", ls="--", alpha=0.6,
                   label=f"TopQ-500 mean = {topq_mean:.4f}")

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Mean Score")
    ax.set_title("Ablation Study: Effect of Removing Score Components")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "ablation_comparison.png", dpi=150)
    plt.close(fig)
    logger.info("Saved ablation_comparison.png")


def plot_overlap_heatmap(subsets: list, out_dir: Path = OUTPUT_ROOT):
    """
    Jaccard similarity heatmap between standard subsets.
    Shows trajectory overlap (which subsets share the most trajectories).
    """
    standard = [s for s in subsets if not _is_ablation(s.name) and len(s.df) > 0]
    if len(standard) < 2:
        return

    names = [s.name for s in standard]
    n     = len(names)
    jaccard = np.zeros((n, n))
    for i in range(n):
        set_i = set(standard[i].df["trajectory_id"]) if "trajectory_id" in standard[i].df.columns else set(standard[i].df.index)
        for j in range(n):
            set_j = set(standard[j].df["trajectory_id"]) if "trajectory_id" in standard[j].df.columns else set(standard[j].df.index)
            union = len(set_i | set_j)
            jaccard[i][j] = len(set_i & set_j) / union if union > 0 else 0

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(jaccard, annot=True, fmt=".2f", cmap="YlOrRd",
                xticklabels=names, yticklabels=names, ax=ax, vmin=0, vmax=1)
    ax.set_title("Subset Overlap (Jaccard Similarity)")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "subset_overlap_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Saved subset_overlap_heatmap.png")


def generate_comparison_table(subsets: list, out_dir: Path = OUTPUT_ROOT) -> pd.DataFrame:
    """Generate and save a comprehensive comparison CSV for all subsets."""
    rows = []
    for s in subsets:
        row = {
            "subset_name":        s.name,
            "description":        s.description,
            "purpose":            s.purpose,
            "selection_criteria": s.selection_criteria,
            "expected_size":      s.expected_size,
            "actual_size":        s.actual_size,
        }
        row.update(s.stats)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "subset_comparison_table.csv", index=False)
    logger.info("Saved subset_comparison_table.csv")
    return table


def generate_all_plots(subsets: list, out_dir: Path = OUTPUT_ROOT) -> pd.DataFrame:
    """Run all visualization functions for Phase 2."""
    out_dir.mkdir(exist_ok=True)
    logger.info("Generating all Phase 2 plots to %s", out_dir)

    plot_composite_q_comparison(subsets, out_dir)
    plot_score_group_comparison(subsets, out_dir)
    plot_subset_sizes(subsets, out_dir)
    plot_token_distribution_by_subset(subsets, out_dir)
    plot_outcome_by_subset(subsets, out_dir)
    plot_ablation_comparison(subsets, out_dir)
    plot_overlap_heatmap(subsets, out_dir)
    table = generate_comparison_table(subsets, out_dir)

    logger.info("All Phase 2 plots generated.")
    return table
