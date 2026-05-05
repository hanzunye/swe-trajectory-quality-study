#!/usr/bin/env python3
"""
Main pipeline: Download → Analyze → Score (v3) → Visualize

Usage:
    python run_analysis.py                    # full pipeline (streaming, all 67k rows)
    python run_analysis.py --limit 1000       # quick test with first 1000 rows
    python run_analysis.py --skip-viz         # skip visualization step
    python run_analysis.py --resume           # resume from saved intermediate results
    python run_analysis.py --resume --skip-viz --export-splits  # re-score + export splits

v3 Scoring (4 active dims, after variance analysis):
    Gate:        truncation_ratio >= 0.9 AND resolved == 1
    Efficiency:  mean(B2, B3)          [B1 excluded: low variance on this dataset]
    Style:       mean(C2, C3)          [C1 excluded: low variance on this dataset]
    Composite:   0.5 * Efficiency + 0.5 * Style

Experiment groups (11 total, --export-splits):
    #   Random-500 / Random-1000     random from ALL trajectories
    #   TopQ-500 / TopQ-1000        top N by composite_score (resolved pool)
    #   ResolvedOnly-500/ResolvedOnly-1000  random from resolved pool
    #   BottomQ-500                bottom 500 by composite_score (sanity check)
    #   Ablation-NoEfficiency-500 (Style only)
    #   Ablation-NoStyle-500      (Efficiency only)
    #   Ablation-NoB2-500         (Efficiency = B3 alone)
    #   Ablation-NoB3-500         (Efficiency = B2 alone)
    #   Ablation-NoC2-500         (Style = C3 alone)
    #   Ablation-NoC3-500         (Style = C2 alone)
"""
import argparse
import json
import logging
import os
import time
import sys
from pathlib import Path

import pandas as pd
import psutil
from datasets import load_dataset
from tqdm import tqdm

_here = Path(__file__).resolve().parent
sys.path.insert(0, str(_here))

from scoring_config import (
    DATASET_NAME,
    DATASET_SPLIT,
    GATE_MIN_TRUNCATION_RATIO,
    MODEL_CONTEXT_WINDOW,
    STREAMING,
    get_output_dir,
)
from analysis import TrajectoryStats, analyze_trajectory
from scoring import (
    composite_quality_score,
    compute_per_task_medians,
    b1_redundant_commands,
    b2_error_retry,
    b3_step_count_ratio,
    c1_observation_cleanliness,
    c2_action_diversity,
    c3_observation_utilization,
)
from scoring_visualize import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


def load_streaming(skip_n: int = 0):
    """Load the raw trajectory dataset as an iterable stream."""
    stream = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=STREAMING)
    if skip_n:
        stream = stream.skip(skip_n)
    return stream


def close_stream(stream) -> None:
    """Close a streaming dataset if the object exposes a close method."""
    close = getattr(stream, "close", None)
    if callable(close):
        close()


def log_memory():
    mem = psutil.Process().memory_info()
    logger.info("Memory usage: %.1f MB (RSS)", mem.rss / 1024 / 1024)


def _analyze_stream(stream, records, jsonl_f, limit, initial_offset=0):
    """Inner loop: analyze trajectory stream, write JSONL, append to records."""
    for i, row in enumerate(tqdm(
        stream, desc="Analyzing", unit="traj", initial=initial_offset,
    )):
        if limit is not None and i >= limit:
            break
        stats = analyze_trajectory(row)
        scores_v1 = composite_quality_score(stats)
        record = stats.to_dict()
        record.update(scores_v1)
        records.append(record)
        jsonl_f.write(json.dumps(record, ensure_ascii=False) + "\n")
        total_done = initial_offset + i + 1
        if total_done % 5000 == 0:
            log_memory()
            jsonl_f.flush()
            logger.info("Checkpoint saved at row %d", total_done)


def run_analysis(
    limit: int | None = None,
    skip_viz: bool = False,
    resume: bool = False,
    export_splits: bool = False,
):
    start_time = time.time()

    output_dir = get_output_dir(limit)
    INTERMEDIATE_FILE = output_dir / "trajectory_stats.jsonl"
    FINAL_CSV = output_dir / "trajectory_analysis.csv"
    SCORED_CSV = output_dir / "trajectory_scored_v3.csv"
    logger.info("Output directory: %s", output_dir)

    # ══════════════════════════════════════════════════════════
    #  Step 1: Analyze trajectories (stream → JSONL checkpoint)
    # ══════════════════════════════════════════════════════════
    existing_records: list[dict] = []
    if resume and INTERMEDIATE_FILE.exists():
        logger.info("Loading checkpoint from %s ...", INTERMEDIATE_FILE)
        with open(INTERMEDIATE_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    existing_records.append(json.loads(line))
        logger.info("Checkpoint: %d rows already processed.", len(existing_records))

    skip_n = len(existing_records)
    already_done = resume and existing_records and (limit is None or skip_n >= limit)

    if already_done:
        records = existing_records
        logger.info("All %d rows already processed. Skipping streaming analysis.", len(records))
    elif resume and existing_records:
        records = existing_records
        remaining = (limit - skip_n) if limit is not None else None
        logger.info("Resuming from row %d (remaining: %s)...", skip_n, remaining or "ALL")
        log_memory()
        stream = load_streaming(skip_n=skip_n)
        with open(INTERMEDIATE_FILE, "a") as jsonl_f:
            try:
                _analyze_stream(stream, records, jsonl_f,
                                limit=remaining, initial_offset=skip_n)
            finally:
                close_stream(stream)
        logger.info("Analysis complete: %d trajectories total.", len(records))
        log_memory()
    else:
        logger.info("Starting trajectory analysis (limit=%s)...", limit or "ALL")
        log_memory()
        records = []
        INTERMEDIATE_FILE.parent.mkdir(exist_ok=True)
        stream = load_streaming()
        with open(INTERMEDIATE_FILE, "w") as jsonl_f:
            try:
                _analyze_stream(stream, records, jsonl_f, limit=limit)
            finally:
                close_stream(stream)
        logger.info("Analysis complete: %d trajectories processed.", len(records))
        log_memory()

    # ══════════════════════════════════════════════════════════
    #  Step 2: Build DataFrame + apply v3 gate
    # ══════════════════════════════════════════════════════════
    logger.info("Building DataFrame...")
    df = pd.DataFrame(records)

    if "truncation_ratio" in df.columns:
        tr_col = df["truncation_ratio"]
    else:
        total_tok = df["total_tokens"].replace(0, 1)
        tr_col = total_tok.clip(upper=MODEL_CONTEXT_WINDOW) / total_tok
    df["passes_gate"] = (tr_col >= GATE_MIN_TRUNCATION_RATIO) & (df["resolved"] == 1)

    n_pass = df["passes_gate"].sum()
    logger.info(
        "Gate: %d / %d trajectories pass (%.1f%%)",
        n_pass, len(df), 100 * n_pass / max(1, len(df)),
    )

    # ══════════════════════════════════════════════════════════
    #  Step 3: v3 scoring (gate-passing pool only)
    # ══════════════════════════════════════════════════════════
    logger.info("Computing per-task step medians (B3)...")
    per_task_median = compute_per_task_medians(df)

    logger.info("Computing v3 scores for %d gate-passing trajectories...", n_pass)

    def _score_row_v3(row: pd.Series) -> pd.Series:
        """Reconstruct v3 scores from stored raw metrics. No re-streaming needed."""
        nan = float("nan")
        if not row["passes_gate"]:
            return pd.Series({
                "b1_redundant_commands":      nan,
                "b2_error_retry":             nan,
                "b3_step_count_ratio":        nan,
                "c1_observation_cleanliness": nan,
                "c2_action_diversity":        nan,
                "c3_observation_utilization": nan,
                "efficiency_score":           nan,
                "style_score":                nan,
                "composite_score":            nan,
            })

        stats = TrajectoryStats(
            trajectory_id=row.get("trajectory_id", ""),
            instance_id=row.get("instance_id", ""),
            repo=row.get("repo", ""),
            exit_status=row.get("exit_status", ""),
            resolved=int(row.get("resolved", 0)),
            total_tokens=int(row.get("total_tokens", 0)),
            assistant_turns=int(row.get("assistant_turns", 0)),
            tool_call_counts=row.get("tool_call_counts") or {},
            total_tool_calls=int(row.get("total_tool_calls", 0)),
            ends_with_submit=bool(row.get("ends_with_submit", False)),
            is_error_or_timeout=bool(row.get("is_error_or_timeout", True)),
            unique_action_ratio=float(row.get("unique_action_ratio", 1.0)),
            test_alignment_score=float(row.get("test_alignment_score", 0.0)),
            redundant_cmds_ratio=float(row.get("redundant_cmds_ratio", 0.0)),
            error_retry_count=int(row.get("error_retry_count", 0)),
            total_action_pairs=int(row.get("total_action_pairs", 0)),
            obs_noise_char_ratio=float(row.get("obs_noise_char_ratio", 0.0)),
            obs_utilization_ratio=float(row.get("obs_utilization_ratio", 1.0)),
        )

        b1 = b1_redundant_commands(stats)   # stored for reporting, not in composite
        b2 = b2_error_retry(stats)
        b3 = b3_step_count_ratio(stats, per_task_median)
        c1 = c1_observation_cleanliness(stats)  # stored for reporting, not in composite
        c2 = c2_action_diversity(stats)
        c3 = c3_observation_utilization(stats)

        # Active formula: Efficiency = mean(B2, B3), Style = mean(C2, C3)
        eff = (b2 + b3) / 2.0
        sty = (c2 + c3) / 2.0

        return pd.Series({
            "b1_redundant_commands":      b1,
            "b2_error_retry":             b2,
            "b3_step_count_ratio":        b3,
            "c1_observation_cleanliness": c1,
            "c2_action_diversity":        c2,
            "c3_observation_utilization": c3,
            "efficiency_score":           eff,
            "style_score":                sty,
            "composite_score":            0.5 * eff + 0.5 * sty,
        })

    v3_cols = df.apply(_score_row_v3, axis=1)
    df = pd.concat([df, v3_cols], axis=1)

    # ══════════════════════════════════════════════════════════
    #  Step 4: Save CSVs
    # ══════════════════════════════════════════════════════════
    csv_df = df.drop(columns=["tool_call_counts"], errors="ignore")
    csv_df.to_csv(FINAL_CSV, index=False)
    logger.info("Saved full analysis to %s", FINAL_CSV)

    scored_df = csv_df[csv_df["passes_gate"]].sort_values(
        "composite_score", ascending=False
    )
    scored_df.to_csv(SCORED_CSV, index=False)
    logger.info("Saved %d scored trajectories to %s", len(scored_df), SCORED_CSV)

    # ══════════════════════════════════════════════════════════
    #  Step 5: Export experiment splits
    # ══════════════════════════════════════════════════════════
    if export_splits:
        _export_experiment_splits(csv_df, scored_df, output_dir)

    # ══════════════════════════════════════════════════════════
    #  Step 6: Print summary
    # ══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("  TRAJECTORY ANALYSIS SUMMARY  (v3 scoring)")
    print("=" * 70)
    print(f"  Total trajectories:          {len(df):,}")
    print(f"  Resolved (gate-passing):     {n_pass:,}  "
          f"({100*n_pass/max(1,len(df)):.1f}%)")
    print(f"  Avg tokens/trajectory:       {df['total_tokens'].mean():,.0f}")
    print(f"  Avg turns/trajectory:        {df['assistant_turns'].mean():.1f}")
    print(f"  Resolved rate:               {(df['resolved'] == 1).mean() * 100:.1f}%")
    print(f"  Unique repos:                {df['repo'].nunique()}")
    if n_pass > 0:
        sd = scored_df
        print(f"\n  ── v3 Sub-dimension scores (resolved pool, n={n_pass:,}) ──")
        print(f"  {'Dimension':<32}  {'mean':>6}  {'std':>6}  {'median':>6}  note")
        print(f"  {'-'*65}")
        rows = [
            ("b1_redundant_commands",      "B1 Redundant Cmds",      "excluded (low var)"),
            ("b2_error_retry",             "B2 Error-Retry",         "active"),
            ("b3_step_count_ratio",        "B3 Step Ratio",          "active"),
            ("c1_observation_cleanliness", "C1 Obs Cleanliness",     "excluded (low var)"),
            ("c2_action_diversity",        "C2 Action Diversity",    "active"),
            ("c3_observation_utilization", "C3 Obs Utilization",     "active"),
            ("efficiency_score",           "Efficiency  mean(B2,B3)",""),
            ("style_score",                "Style       mean(C2,C3)",""),
            ("composite_score",            "Composite Score",        ""),
        ]
        for col, lbl, note in rows:
            if col in sd.columns:
                print(f"  {lbl:<32}  {sd[col].mean():>6.3f}  "
                      f"{sd[col].std():>6.3f}  {sd[col].median():>6.3f}  {note}")
    print("=" * 70)

    # ══════════════════════════════════════════════════════════
    #  Step 7: Visualize
    # ══════════════════════════════════════════════════════════
    if not skip_viz:
        logger.info("Generating visualizations...")
        summary = generate_all_plots(df, out_dir=output_dir)
        print("\nSummary statistics:")
        print(summary.to_string())
    else:
        logger.info("Skipping visualization (--skip-viz).")

    elapsed = time.time() - start_time
    logger.info("Pipeline complete in %.1f minutes.", elapsed / 60)
    print(f"\nResults saved to: {output_dir}")
    return df


def _export_experiment_splits(
    csv_df: pd.DataFrame,      # full dataset (no tool_call_counts)
    scored_df: pd.DataFrame,   # gate-passing, sorted by composite_score desc
    output_dir: Path,
    sizes: tuple[int, ...] = (500, 1000),
) -> None:
    """
    Export experiment split CSVs for SFT data selection experiments.

    Active scoring: Efficiency = mean(B2, B3),  Style = mean(C2, C3)

    Splits (11 total):
      Random-N             random from ALL trajectories
      TopQ-N               top N by composite_score (resolved pool)
      ResolvedOnly-N       random from resolved pool
      BottomQ-500          bottom 500 by composite_score (sanity check)
      Ablation-NoEfficiency-500   rank by Style only       (C2+C3)/2
      Ablation-NoStyle-500        rank by Efficiency only  (B2+B3)/2
      Ablation-NoB2-500           Efficiency = B3 only
      Ablation-NoB3-500           Efficiency = B2 only
      Ablation-NoC2-500           Style = C3 only
      Ablation-NoC3-500           Style = C2 only
    """
    splits_dir = output_dir / "splits"
    splits_dir.mkdir(exist_ok=True)
    logger.info("Exporting experiment splits to %s ...", splits_dir)

    # Work on a copy to safely add temporary columns
    sd = scored_df.copy()

    for N in sizes:
        # ── Random-N ──
        csv_df.sample(n=min(N, len(csv_df)), random_state=42).to_csv(
            splits_dir / f"Random-{N}.csv", index=False)

        # ── TopQ-N ──
        sd.head(N).to_csv(splits_dir / f"TopQ-{N}.csv", index=False)

        # ── ResolvedOnly-N ──
        sd.sample(n=min(N, len(sd)), random_state=42).to_csv(
            splits_dir / f"ResolvedOnly-{N}.csv", index=False)

        if N == 500:
            # ── BottomQ-500 ──
            sd.tail(500).to_csv(splits_dir / "BottomQ-500.csv", index=False)

            # ── Ablation: NoEfficiency — rank by Style only ──
            sd.sort_values("style_score", ascending=False).head(500).to_csv(
                splits_dir / "Ablation-NoEfficiency-500.csv", index=False)

            # ── Ablation: NoStyle — rank by Efficiency only ──
            sd.sort_values("efficiency_score", ascending=False).head(500).to_csv(
                splits_dir / "Ablation-NoStyle-500.csv", index=False)

            # ── Ablation: NoB2 — Efficiency = B3 only ──
            if "b3_step_count_ratio" in sd.columns and "style_score" in sd.columns:
                sd["_score"] = 0.5 * sd["b3_step_count_ratio"] + 0.5 * sd["style_score"]
                sd.sort_values("_score", ascending=False).head(500).drop(
                    columns=["_score"]).to_csv(
                    splits_dir / "Ablation-NoB2-500.csv", index=False)

            # ── Ablation: NoB3 — Efficiency = B2 only ──
            if "b2_error_retry" in sd.columns and "style_score" in sd.columns:
                sd["_score"] = 0.5 * sd["b2_error_retry"] + 0.5 * sd["style_score"]
                sd.sort_values("_score", ascending=False).head(500).drop(
                    columns=["_score"]).to_csv(
                    splits_dir / "Ablation-NoB3-500.csv", index=False)

            # ── Ablation: NoC2 — Style = C3 only ──
            if "c3_observation_utilization" in sd.columns and "efficiency_score" in sd.columns:
                sd["_score"] = 0.5 * sd["efficiency_score"] + 0.5 * sd["c3_observation_utilization"]
                sd.sort_values("_score", ascending=False).head(500).drop(
                    columns=["_score"]).to_csv(
                    splits_dir / "Ablation-NoC2-500.csv", index=False)

            # ── Ablation: NoC3 — Style = C2 only ──
            if "c2_action_diversity" in sd.columns and "efficiency_score" in sd.columns:
                sd["_score"] = 0.5 * sd["efficiency_score"] + 0.5 * sd["c2_action_diversity"]
                sd.sort_values("_score", ascending=False).head(500).drop(
                    columns=["_score"]).to_csv(
                    splits_dir / "Ablation-NoC3-500.csv", index=False)

    # Clean up any leftover temp column
    if "_score" in scored_df.columns:
        scored_df.drop(columns=["_score"], inplace=True, errors="ignore")

    logger.info("Experiment splits exported.")
    splits = sorted(splits_dir.glob("*.csv"))
    print(f"\n  ── Experiment splits ({len(splits)} files) ──")
    for s in splits:
        n_rows = sum(1 for _ in open(s)) - 1
        print(f"    {s.name:<42}  {n_rows:>5} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OpenHands Trajectory Analysis Pipeline (v3 scoring)"
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of trajectories (default: all)")
    parser.add_argument("--skip-viz", action="store_true",
                        help="Skip visualization generation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from intermediate JSONL (skips streaming analysis)")
    parser.add_argument("--export-splits", action="store_true",
                        help="Export all experiment split CSVs after scoring")
    args = parser.parse_args()

    run_analysis(
        limit=args.limit,
        skip_viz=args.skip_viz,
        resume=args.resume,
        export_splits=args.export_splits,
    )
    sys.exit(0)
