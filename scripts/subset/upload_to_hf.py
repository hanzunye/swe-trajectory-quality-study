#!/usr/bin/env python3
"""
Upload Phase 2 subsets to Hugging Face Hub.

Each subset is uploaded as a separate config (e.g. "TopQ-500", "Random-500").
Trajectory data + quality scores are stored; use trajectory_id to join back
to full trajectory content from the source dataset.

Usage:
    # First time: authenticate
    huggingface-cli login

    # Upload all subsets
    python upload_to_hf.py --repo-id YOUR_USERNAME/swe-bench-trajectory-quality-subsets

    # Upload specific subsets only
    python upload_to_hf.py --repo-id YOUR_USERNAME/swe-bench-trajectory-quality-subsets --only TopQ-500 Random-500

    # Dry run (show what would be uploaded)
    python upload_to_hf.py --repo-id YOUR_USERNAME/swe-bench-trajectory-quality-subsets --dry-run
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from datasets import Dataset, Features, Value
from huggingface_hub import HfApi, DatasetCard

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("upload")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOTS = [
    Path(__file__).resolve().parent / "output",
    REPO_ROOT / "scripts" / "output",
    REPO_ROOT / "data" / "subsets",
]

# ── HF Dataset Features Schema ───────────────────────────────────────────────
# Columns present in ALL subsets (trajectory_stats.jsonl base columns)
BASE_FEATURES = {
    "trajectory_id":      Value("string"),
    "instance_id":        Value("string"),
    "repo":               Value("string"),
    "total_tokens":       Value("int64"),
    "assistant_turns":    Value("int64"),
    "total_tool_calls":   Value("int64"),
    "exit_status":        Value("string"),
    "ends_with_submit":   Value("bool"),
    "is_error_or_timeout": Value("bool"),
    "resolved":           Value("int64"),
    "has_patch":          Value("bool"),
    "patch_length":       Value("int64"),
    "avg_observation_length": Value("float64"),
    "num_observations":   Value("int64"),
    "truncation_ratio":   Value("float64"),
    # v1 legacy score (present for all trajectories)
    "composite_Q":        Value("float64"),
}

# v3 score columns — present only for resolved/scored subsets (NaN for Random-*)
V3_FEATURES = {
    "passes_gate":                Value("bool"),
    "b1_redundant_commands":      Value("float64"),
    "b2_error_retry":             Value("float64"),
    "b3_step_count_ratio":        Value("float64"),
    "c1_observation_cleanliness": Value("float64"),
    "c2_action_diversity":        Value("float64"),
    "c3_observation_utilization": Value("float64"),
    "efficiency_score":           Value("float64"),
    "style_score":                Value("float64"),
    "composite_score":            Value("float64"),
}

FEATURES = Features({**BASE_FEATURES, **V3_FEATURES})


# ── Helpers ───────────────────────────────────────────────────────────────────

def discover_subsets(output_root: Path) -> list[str]:
    """Find all subset directories that contain trajectories.jsonl."""
    if not output_root.exists():
        return []
    return [
        d.name for d in sorted(output_root.iterdir())
        if d.is_dir() and (d / "trajectories.jsonl").exists()
    ]


def resolve_output_root(output_root: Path | None = None) -> Path:
    """Resolve the subset directory, preferring an explicit CLI/env path."""
    if output_root is not None:
        return output_root

    env_root = os.environ.get("SUBSET_OUTPUT_ROOT")
    if env_root:
        return Path(env_root)

    for candidate in DEFAULT_OUTPUT_ROOTS:
        if discover_subsets(candidate):
            return candidate

    return DEFAULT_OUTPUT_ROOTS[-1]


def load_subset_data(subset_name: str, output_root: Path) -> tuple[list[dict], dict]:
    """Load a subset's trajectory data and metadata."""
    subset_dir = output_root / subset_name

    records = []
    with open(subset_dir / "trajectories.jsonl", "r") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    metadata = {}
    meta_path = subset_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path, "r") as f:
            metadata = json.load(f)

    return records, metadata


def _normalise_record(record: dict) -> dict:
    """Ensure all FEATURES columns are present, filling missing ones with None."""
    out = {}
    for col in FEATURES:
        val = record.get(col)
        # Coerce basic types
        feature = FEATURES[col]
        if val is None:
            out[col] = None
        elif feature.dtype == "bool" and not isinstance(val, bool):
            out[col] = bool(val)
        elif feature.dtype == "int64" and not isinstance(val, int):
            try:
                out[col] = int(val) if val is not None else None
            except (ValueError, TypeError):
                out[col] = None
        elif feature.dtype == "float64" and not isinstance(val, float):
            try:
                out[col] = float(val) if val is not None else None
            except (ValueError, TypeError):
                out[col] = None
        else:
            out[col] = val
    return out


def build_dataset_card(repo_id: str, all_metadata: dict[str, dict]) -> str:
    """Generate a dataset card (README.md) for the HF dataset."""
    lines = [
        "---",
        "license: apache-2.0",
        "task_categories:",
        "  - text-generation",
        "tags:",
        "  - swe-bench",
        "  - openhands",
        "  - trajectory-quality-scoring",
        "  - code-generation",
        "  - fine-tuning",
        "language:",
        "  - en",
        "pretty_name: SWE-bench Trajectory Quality Subsets",
        "size_categories:",
        "  - 1K<n<10K",
        "---",
        "",
        "# SWE-bench Trajectory Quality Subsets",
        "",
        "Curated subsets of [nebius/SWE-rebench-openhands-trajectories]"
        "(https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories) "
        "constructed using the v3 quality scoring framework for fine-tuning evaluation.",
        "",
        "## Subsets Overview",
        "",
        "| Subset | Size | Selection | Mean Score | Resolved Rate | Purpose |",
        "|--------|------|-----------|------------|---------------|---------|",
    ]

    for name, meta in sorted(all_metadata.items()):
        stats = meta.get("stats", {})
        lines.append(
            f"| **{name}** "
            f"| {meta.get('actual_size', '?')} "
            f"| {meta.get('selection_criteria', '?')} "
            f"| {stats.get('mean_Q', 0):.4f} "
            f"| {stats.get('resolved_rate', 0) * 100:.0f}% "
            f"| {meta.get('purpose', '?')} |"
        )

    lines.extend([
        "",
        "## Quality Score v3",
        "",
        "The v3 scoring formula uses a gate + two-group composite:",
        "",
        "```",
        "Gate (filter, not scored):",
        "  truncation_ratio >= 0.9  AND  resolved == 1",
        "",
        "Score = 0.5 * Efficiency + 0.5 * Style",
        "",
        "Efficiency = mean(B2, B3)",
        "  B2: Error-Retry Cycles   — 1 - clip(cycles / 10, 0, 1)",
        "  B3: Step Count Ratio     — 1 - normalize(clip(steps/median, 0.5, 3.0))",
        "",
        "Style = mean(C2, C3)",
        "  C2: Action Diversity     — Shannon entropy of tool types, normalized",
        "  C3: Obs. Utilization     — fraction of obs. keywords reused in actions",
        "",
        "B1 (redundant_commands) and C1 (observation_cleanliness) are stored",
        "but excluded from the composite (low variance on this dataset).",
        "```",
        "",
        "## Subset Groups",
        "",
        "| Group | Subsets | Pool |",
        "|-------|---------|------|",
        "| Random baseline | Random-500, Random-1000, Random-2000 | ALL trajectories |",
        "| Top quality | TopQ-500, TopQ-1000, TopQ-2000 | Resolved pool |",
        "| Resolved baseline | ResolvedOnly-500, ResolvedOnly-1000 | Resolved pool |",
        "| Bottom quality | BottomQ-500 | Resolved pool (sanity check) |",
        "| Ablation (group) | Ablation-NoEfficiency-500, Ablation-NoStyle-500 | Resolved pool |",
        "| Ablation (dim) | Ablation-NoB2-500, Ablation-NoB3-500, Ablation-NoC2-500, Ablation-NoC3-500 | Resolved pool |",
        "| Single-dimension | B2Only-Top500 | Resolved pool |",
        "",
        "## Usage",
        "",
        "```python",
        "from datasets import load_dataset",
        "",
        "# Load a specific subset",
        f'ds = load_dataset("{repo_id}", "TopQ-500")',
        "",
        "# Retrieve trajectory IDs for downstream filtering",
        "ids = set(ds['train']['trajectory_id'])",
        "",
        "# Filter the full source dataset",
        'full = load_dataset("nebius/SWE-rebench-openhands-trajectories",',
        '                    split="train", streaming=True)',
        "filtered = (row for row in full if row['trajectory_id'] in ids)",
        "```",
        "",
        "## Column Reference",
        "",
        "| Column | Type | Description |",
        "|--------|------|-------------|",
        "| trajectory_id | string | Unique trajectory identifier |",
        "| instance_id | string | SWE-bench problem instance ID |",
        "| repo | string | GitHub repository name |",
        "| total_tokens | int | Total token count |",
        "| assistant_turns | int | Number of assistant messages |",
        "| total_tool_calls | int | Total tool invocations |",
        "| exit_status | string | Final status (submit/error/timeout) |",
        "| ends_with_submit | bool | Ended with submit action |",
        "| is_error_or_timeout | bool | Ended in error or timeout |",
        "| resolved | int | 1 = issue solved, 0 = not |",
        "| has_patch | bool | Patch was generated |",
        "| patch_length | int | Patch character length |",
        "| truncation_ratio | float | Fraction of trajectory within context window |",
        "| composite_Q | float | v1 composite quality score [0,1] |",
        "| passes_gate | bool | Passes v3 gate (resolved + truncation_ratio ≥ 0.9) |",
        "| b2_error_retry | float | B2: Error-retry penalty score [0,1] |",
        "| b3_step_count_ratio | float | B3: Step count efficiency score [0,1] |",
        "| c2_action_diversity | float | C2: Action diversity score [0,1] |",
        "| c3_observation_utilization | float | C3: Observation utilization score [0,1] |",
        "| efficiency_score | float | mean(B2, B3) [0,1] |",
        "| style_score | float | mean(C2, C3) [0,1] |",
        "| composite_score | float | 0.5*Efficiency + 0.5*Style [0,1] |",
        "",
        "## Source Dataset",
        "",
        "[nebius/SWE-rebench-openhands-trajectories]"
        "(https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)",
        "",
    ])

    return "\n".join(lines)


# ── Upload ────────────────────────────────────────────────────────────────────

def upload(
    repo_id: str,
    subset_names: list[str] | None = None,
    dry_run: bool = False,
    private: bool = False,
    output_root: Path | None = None,
):
    """
    Upload subsets to Hugging Face Hub.

    Args:
        repo_id:       HF repo ID (e.g. "username/swe-bench-trajectory-quality-subsets")
        subset_names:  Specific subsets to upload. None = all discovered.
        dry_run:       Show plan without uploading.
        private:       Create private dataset.
    """
    output_root = resolve_output_root(output_root)
    available = discover_subsets(output_root)
    if not available:
        logger.error("No subsets found in %s", output_root)
        sys.exit(1)

    logger.info("Using subset directory: %s", output_root)
    logger.info("Found %d subsets: %s", len(available), available)

    if subset_names:
        for name in subset_names:
            if name not in available:
                logger.error("Subset '%s' not found. Available: %s", name, available)
                sys.exit(1)
        targets = subset_names
    else:
        targets = available

    # Load data
    dataset_dict = {}
    all_metadata = {}

    for name in targets:
        logger.info("Loading subset: %s", name)
        records, metadata = load_subset_data(name, output_root)
        all_metadata[name] = metadata

        # Normalise to FEATURES schema
        normalised = [_normalise_record(r) for r in records]
        ds = Dataset.from_list(normalised, features=FEATURES)
        dataset_dict[name] = ds
        logger.info("  → %d rows", len(ds))

    # Summary
    print("\n" + "=" * 65)
    print("  UPLOAD SUMMARY")
    print("=" * 65)
    print(f"  Repository:  {repo_id}")
    print(f"  Subsets:     {len(dataset_dict)}")
    print(f"  Private:     {private}")
    for name, ds in dataset_dict.items():
        meta = all_metadata.get(name, {})
        print(f"    {name:<40s} → {len(ds):>5d} rows  ({meta.get('purpose', '')})")
    print("=" * 65)

    if dry_run:
        print("\n[DRY RUN] No data uploaded. Remove --dry-run to upload.")
        return

    # Create repo
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset",
                    private=private, exist_ok=True)
    logger.info("Repository ready: %s", repo_id)

    # Push each subset as a config
    for name, ds in dataset_dict.items():
        logger.info("Uploading '%s' (%d rows)...", name, len(ds))
        ds.push_to_hub(repo_id=repo_id, config_name=name, split="train")
        logger.info("  ✓ %s uploaded", name)

    # Push dataset card
    logger.info("Uploading dataset card...")
    card = DatasetCard(build_dataset_card(repo_id, all_metadata))
    card.push_to_hub(repo_id)
    logger.info("  ✓ Dataset card uploaded")

    print(f"\nDone! Dataset available at:")
    print(f"  https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Upload Phase 2 subsets to Hugging Face Hub",
    )
    parser.add_argument(
        "--repo-id", type=str, required=True,
        help="HF repo ID, e.g. 'your-username/swe-bench-trajectory-quality-subsets'",
    )
    parser.add_argument(
        "--only", type=str, nargs="+", default=None,
        metavar="SUBSET",
        help="Upload specific subsets only (e.g. --only TopQ-500 Random-500)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--private", action="store_true",
        help="Create private dataset (default: public)",
    )
    parser.add_argument(
        "--output-root", type=Path, default=None,
        help="Subset root directory (default: auto-detect data/subsets or scripts/output)",
    )
    args = parser.parse_args()

    upload(
        repo_id=args.repo_id,
        subset_names=args.only,
        dry_run=args.dry_run,
        private=args.private,
        output_root=args.output_root,
    )
