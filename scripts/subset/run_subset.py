#!/usr/bin/env python3
"""
Phase 2 Pipeline: Subset Construction

Reads Quality_Scoring v3 output and constructs experimental subsets for
downstream fine-tuning evaluation.

Data sources:
  trajectory_stats.jsonl      — ALL 67k trajectories (resolved + unresolved)
  trajectory_scored_v3.csv    — 32k resolved gate-passing trajectories with v3 scores

Subset groups:
  Random-500 / Random-1000        random from ALL trajectories
  TopQ-500 / TopQ-1000            top N by composite_score (resolved pool)
  ResolvedOnly-500 / -1000        random from resolved pool
  BottomQ-500                     bottom 500 by composite_score (sanity check)
  Ablation-NoEfficiency-500       Style only   (drop Efficiency term)
  Ablation-NoStyle-500            Efficiency only  (drop Style term)
  Ablation-NoB2-500               Efficiency = B3 alone
  Ablation-NoB3-500               Efficiency = B2 alone
  Ablation-NoC2-500               Style = C3 alone
  Ablation-NoC3-500               Style = C2 alone

Usage:
    python run_subset.py                             # full pipeline
    python run_subset.py --skip-viz                  # skip plots
    python run_subset.py --only TopQ-500             # single subset
    python run_subset.py --list                      # list all subset names
"""
import argparse
import json
import logging
import time
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from subset_config import PHASE1_JSONL, SCORED_CSV, OUTPUT_ROOT
from builder import (
    SubsetResult,
    build_all_subsets,
    build_random,
    build_topq,
    build_resolvedonly,
    build_bottomq,
    build_ablation_no_efficiency,
    build_ablation_no_style,
    build_ablation_no_b2,
    build_ablation_no_b3,
    build_ablation_no_c2,
    build_ablation_no_c3,
)
from subset_visualize import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("phase2")


AVAILABLE_SUBSETS = [
    "Random-500",
    "Random-1000",
    "TopQ-500",
    "TopQ-1000",
    "ResolvedOnly-500",
    "ResolvedOnly-1000",
    "BottomQ-500",
    "Ablation-NoEfficiency-500",
    "Ablation-NoStyle-500",
    "Ablation-NoB2-500",
    "Ablation-NoB3-500",
    "Ablation-NoC2-500",
    "Ablation-NoC3-500",
]


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_all_trajectories(jsonl_path: Path) -> pd.DataFrame:
    """
    Load ALL trajectories from trajectory_stats.jsonl (Phase 1 output).
    Includes both resolved and unresolved trajectories.
    """
    logger.info("Loading all trajectories from %s ...", jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(
            f"trajectory_stats.jsonl not found: {jsonl_path}\n"
            "Please run Quality_Scoring first."
        )

    records = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    df = pd.DataFrame(records)
    logger.info("Loaded %d total trajectories.", len(df))

    required = ["trajectory_id", "total_tokens", "assistant_turns",
                "ends_with_submit", "resolved", "repo"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trajectory_stats.jsonl missing columns: {missing}")

    return df


def load_scored_trajectories(csv_path: Path) -> pd.DataFrame:
    """
    Load resolved gate-passing trajectories with v3 quality scores
    from trajectory_scored_v3.csv.
    """
    logger.info("Loading scored trajectories from %s ...", csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"trajectory_scored_v3.csv not found: {csv_path}\n"
            "Please run Quality_Scoring (v3) first."
        )

    df = pd.read_csv(csv_path)
    logger.info(
        "Loaded %d resolved/scored trajectories with %d columns.",
        len(df), len(df.columns),
    )

    required = ["trajectory_id", "composite_score", "efficiency_score",
                "style_score", "b2_error_retry", "b3_step_count_ratio",
                "c2_action_diversity", "c3_observation_utilization"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"trajectory_scored_v3.csv missing columns: {missing}")

    return df


# ── Saving ───────────────────────────────────────────────────────────────────

def save_subset(result: SubsetResult, out_dir: Path):
    """
    Save a single subset to disk:
      {name}/trajectories.jsonl  — trajectory data
      {name}/trajectory_ids.txt  — plain ID list (for downstream filtering)
      {name}/metadata.json       — subset metadata and stats
    """
    subset_dir = out_dir / result.name
    subset_dir.mkdir(exist_ok=True)

    df_out = result.df.drop(columns=["tool_call_counts"], errors="ignore")

    # JSONL
    jsonl_path = subset_dir / "trajectories.jsonl"
    with open(jsonl_path, "w") as f:
        for _, row in df_out.iterrows():
            f.write(json.dumps(row.to_dict(), ensure_ascii=False, default=str) + "\n")

    # CSV (convenience)
    df_out.to_csv(subset_dir / "trajectories.csv", index=False)

    # Plain ID list
    with open(subset_dir / "trajectory_ids.txt", "w") as f:
        for tid in result.df["trajectory_id"]:
            f.write(str(tid) + "\n")

    # Metadata
    metadata = {
        "name":               result.name,
        "description":        result.description,
        "purpose":            result.purpose,
        "selection_criteria": result.selection_criteria,
        "expected_size":      result.expected_size,
        "actual_size":        result.actual_size,
        "stats":              result.stats,
    }
    with open(subset_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)

    logger.info(
        "Saved '%s' → %s (%d trajectories)",
        result.name, subset_dir, result.actual_size,
    )


# ── Single-subset Dispatch ────────────────────────────────────────────────────

def build_single_subset(
    df_all: pd.DataFrame,
    df_scored: pd.DataFrame,
    name: str,
) -> SubsetResult:
    """Build one subset by name."""
    dispatch = {
        "Random-500":               lambda: build_random(df_all, 500, "Random-500"),
        "Random-1000":              lambda: build_random(df_all, 1000, "Random-1000"),
        "TopQ-500":                 lambda: build_topq(df_scored, 500, "TopQ-500"),
        "TopQ-1000":                lambda: build_topq(df_scored, 1000, "TopQ-1000"),
        "ResolvedOnly-500":         lambda: build_resolvedonly(df_scored, 500, "ResolvedOnly-500"),
        "ResolvedOnly-1000":        lambda: build_resolvedonly(df_scored, 1000, "ResolvedOnly-1000"),
        "BottomQ-500":              lambda: build_bottomq(df_scored),
        "Ablation-NoEfficiency-500": lambda: build_ablation_no_efficiency(df_scored),
        "Ablation-NoStyle-500":     lambda: build_ablation_no_style(df_scored),
        "Ablation-NoB2-500":        lambda: build_ablation_no_b2(df_scored),
        "Ablation-NoB3-500":        lambda: build_ablation_no_b3(df_scored),
        "Ablation-NoC2-500":        lambda: build_ablation_no_c2(df_scored),
        "Ablation-NoC3-500":        lambda: build_ablation_no_c3(df_scored),
    }
    if name not in dispatch:
        raise ValueError(
            f"Unknown subset: '{name}'. Available: {AVAILABLE_SUBSETS}"
        )
    return dispatch[name]()


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_subset_pipeline(skip_viz: bool = False, only: str | None = None):
    """
    Main Phase 2 pipeline.

    Args:
        skip_viz: Skip visualization generation.
        only:     If set, build only the named subset.
    """
    start_time = time.time()

    # ── Step 1: Load data ──────────────────────────────────────────────────
    df_all    = load_all_trajectories(PHASE1_JSONL)
    df_scored = load_scored_trajectories(SCORED_CSV)

    # ── Step 2: Build subsets ──────────────────────────────────────────────
    if only:
        logger.info("Building single subset: %s", only)
        subsets = [build_single_subset(df_all, df_scored, only)]
    else:
        subsets = build_all_subsets(df_all, df_scored)

    # ── Step 3: Save ───────────────────────────────────────────────────────
    for s in subsets:
        save_subset(s, OUTPUT_ROOT)

    # ── Step 4: Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 75)
    print("  PHASE 2: SUBSET CONSTRUCTION SUMMARY")
    print("=" * 75)
    print(f"  All trajectories:  {len(df_all):,}  (trajectory_stats.jsonl)")
    print(f"  Resolved/scored:   {len(df_scored):,}  (trajectory_scored_v3.csv)")
    print(f"  Subsets built:     {len(subsets)}")
    print("-" * 75)
    for s in subsets:
        print(f"  {s.summary()}")
    print("=" * 75)

    # ── Step 5: Visualize ──────────────────────────────────────────────────
    if not skip_viz and len(subsets) > 1:
        logger.info("Generating comparison visualizations...")
        table = generate_all_plots(subsets, out_dir=OUTPUT_ROOT)
        print("\nComparison table (key columns):")
        display_cols = ["subset_name", "actual_size", "mean_Q",
                        "submit_rate", "resolved_rate"]
        display_cols = [c for c in display_cols if c in table.columns]
        print(table[display_cols].to_string(index=False))
    elif skip_viz:
        logger.info("Skipping visualization (--skip-viz).")
    else:
        logger.info("Only one subset built, skipping comparison plots.")

    elapsed = time.time() - start_time
    logger.info("Phase 2 pipeline complete in %.1f seconds.", elapsed)
    print(f"\nResults saved to: {OUTPUT_ROOT}")

    return subsets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2: Subset Construction Pipeline"
    )
    parser.add_argument(
        "--skip-viz", action="store_true",
        help="Skip visualization generation",
    )
    parser.add_argument(
        "--only", type=str, default=None,
        metavar="SUBSET_NAME",
        help="Build only a specific subset (e.g. 'TopQ-500')",
    )
    parser.add_argument(
        "--list", action="store_true", dest="list_subsets",
        help="List all available subset names and exit",
    )
    args = parser.parse_args()

    if args.list_subsets:
        print("Available subsets:")
        for name in AVAILABLE_SUBSETS:
            print(f"  - {name}")
        sys.exit(0)

    run_subset_pipeline(skip_viz=args.skip_viz, only=args.only)
    sys.exit(0)
