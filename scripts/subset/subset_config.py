"""
Phase 2 Configuration: Subset Construction.

Data sources:
  PHASE1_JSONL  — trajectory_stats.jsonl from Quality_Scoring (ALL 67k trajectories)
  SCORED_CSV    — trajectory_scored_v3.csv (32k resolved trajectories with v3 scores)

v3 scoring formula (active dims only):
  Efficiency = mean(B2, B3)          B2=error_retry  B3=step_count_ratio
  Style      = mean(C2, C3)          C2=action_diversity  C3=obs_utilization
  Composite  = 0.5 * Efficiency + 0.5 * Style
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
REPO_ROOT       = Path(__file__).resolve().parents[2]
SCORING_ROOT    = Path(os.environ.get("SCORING_ROOT", REPO_ROOT / "data" / "quality_scores" / "full"))

# ALL trajectories (67k, resolved + unresolved), used for Random-* subsets
PHASE1_JSONL    = SCORING_ROOT / "trajectory_stats.jsonl"

# Resolved trajectories only (32k), with v3 quality scores — used for all other subsets
SCORED_CSV      = SCORING_ROOT / "trajectory_scored_v3.csv"

OUTPUT_ROOT     = Path(os.environ.get("SUBSET_OUTPUT_ROOT", REPO_ROOT / "data" / "subsets"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ── v3 Score Columns ────────────────────────────────────────────────────────
SCORE_COL = "composite_score"          # primary ranking column

# Active efficiency sub-dimensions (B1 excluded — low variance)
V3_EFFICIENCY_DIMS = ["b2_error_retry", "b3_step_count_ratio"]

# Active style sub-dimensions (C1 excluded — low variance)
V3_STYLE_DIMS = ["c2_action_diversity", "c3_observation_utilization"]

# All v3 sub-dimension columns (including excluded ones, stored for transparency)
V3_ALL_SUBDIMS = [
    "b1_redundant_commands",
    "b2_error_retry",
    "b3_step_count_ratio",
    "c1_observation_cleanliness",
    "c2_action_diversity",
    "c3_observation_utilization",
]

V3_SCORE_COLS = V3_ALL_SUBDIMS + ["efficiency_score", "style_score", "composite_score"]

# ── Subset Sizes ────────────────────────────────────────────────────────────
SUBSET_SIZES = {
    "small": 500,
    "large": 1000,
}

# ── Random Seed ─────────────────────────────────────────────────────────────
RANDOM_SEED = 42
