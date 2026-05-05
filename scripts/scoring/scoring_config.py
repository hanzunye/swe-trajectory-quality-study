"""
Configuration for M2 Mac (16GB RAM) optimized processing.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = Path(os.environ.get("SCORING_OUTPUT_ROOT", REPO_ROOT / "data" / "quality_scores"))
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


def get_output_dir(limit: int | None = None) -> Path:
    """
    Return a run-specific output directory based on --limit.
      --limit 1000  → output/limit_1000/
      (no limit)    → output/full/
    """
    if limit:
        sub = f"limit_{limit}"
    else:
        sub = "full"
    d = OUTPUT_ROOT / sub
    d.mkdir(exist_ok=True)
    return d

# ── Dataset ────────────────────────────────────────────
DATASET_NAME = "nebius/SWE-rebench-openhands-trajectories"
DATASET_SPLIT = "train"

# ── M2 Mac Memory Optimization ────────────────────────
# 16GB total, ~10GB usable for Python after OS overhead
BATCH_SIZE = 500          # rows per processing batch
MAX_WORKERS = 4           # M2 has 4 perf + 4 efficiency cores
MEMORY_LIMIT_GB = 10      # soft limit for dataset loading
STREAMING = True          # stream dataset to avoid loading all 2GB into RAM

# ── Tokenizer ─────────────────────────────────────────
TIKTOKEN_ENCODING = "cl100k_base"  # GPT-4/Claude compatible BPE encoding

# ── Tool Name Mapping ─────────────────────────────────
# OpenHands uses these tool function names in trajectories
KNOWN_TOOLS = {
    "execute_bash",
    "str_replace_editor",
    "browser",
    "submit",
    "search",
    "file_editor",
}

# ── Context Window ────────────────────────────────────
# Used for truncation ratio gate
MODEL_CONTEXT_WINDOW = 131072  # 128k tokens (Qwen3-Coder context)

# ── Gate Thresholds ───────────────────────────────────
GATE_MIN_TRUNCATION_RATIO = 0.9   # trajectories below this are discarded
# resolved == 1 is also required (checked directly on stats.resolved)

# ═══════════════════════════════════════════════════════════
#  v3 Scoring System (Gate + Efficiency + Style)
#
#  Gate (filter only, not scored):
#    truncation_ratio < 0.9 → discard
#    resolved == 1 required to enter the scoring pool
#
#  Score = 0.5 * mean(B1, B2, B3) + 0.5 * mean(C1, C2, C3)
#
#  Efficiency sub-dims:
#    B1: Redundant Commands   — inverted redundant cmd ratio within window
#    B2: Error-Retry Cycles   — inverted normalized error-retry cycle count
#    B3: Step Count Ratio     — inverted normalized per-task step ratio
#
#  Style sub-dims:
#    C1: Observation Cleanliness — inverted traceback/error/warning char ratio
#    C2: Action Diversity        — Shannon entropy of tool types (normalized)
#    C3: Observation Utilization — fraction of obs keywords used in later actions
# ═══════════════════════════════════════════════════════════

# B1: sliding window size for redundant command detection
B1_WINDOW_SIZE = 5

# B2: keywords that mark an observation as an error response
B2_ERROR_KEYWORDS = {
    "traceback", "error", "exception", "failed", "failure",
    "syntaxerror", "typeerror", "valueerror", "assertionerror",
    "nameerror", "attributeerror", "importerror", "keyerror",
    "runtimeerror", "oserror", "errno", "stderr",
    "command not found", "no such file", "permission denied",
}

# B2: max cycles to clip at before normalizing (cycles / B2_MAX_CYCLES → [0,1])
B2_MAX_CYCLES = 10

# B3: step ratio clip bounds [low, high] — outside these is extreme behavior
B3_RATIO_CLIP = (0.5, 3.0)

# C1: substrings that mark a line as "noisy" (case-insensitive)
C1_NOISE_PATTERNS = [
    "traceback",
    "error:",
    "warning:",
    "exception:",
    " failed",
    "errno",
    "stderr",
    "at line ",
    "raise ",
    "syntaxerror",
    "typeerror",
    "valueerror",
    "assertionerror",
    "nameerror",
    "attributeerror",
    "importerror",
]

# C3: file extensions to extract from observations as filename candidates
C3_FILE_EXTENSIONS = {
    "py", "txt", "json", "yaml", "yml", "cfg", "ini", "md",
    "sh", "js", "ts", "cpp", "h", "c", "java", "rb", "go",
    "rs", "toml", "xml", "html", "css", "sql",
}

# Test-running keywords for test_alignment detection (kept for backwards compat)
TEST_KEYWORDS = {"pytest", "unittest", "tox", "nose", "coverage", "run_tests", "test_"}

# ── Quality Score Weights v1 (flat, 5-dim, kept for backwards compat) ──
QUALITY_WEIGHTS = {
    "truncation_ratio": 0.30,
    "outcome_success": 0.25,
    "step_efficiency": 0.20,
    "observation_noise": 0.15,
    "action_diversity": 0.10,
}

# ── Quality Score v2: Hierarchical 4-dim framework (kept for backwards compat) ──
HIERARCHY_WEIGHTS = {
    "correctness":  0.35,
    "efficiency":   0.30,
    "style":        0.25,
    "completeness": 0.10,
}

SUB_DIM_WEIGHTS: dict[str, tuple[str, float]] = {
    "outcome_success":   ("correctness",  0.70),
    "test_alignment":    ("correctness",  0.30),
    "redundant_actions": ("efficiency",   0.50),
    "step_count_ratio":  ("efficiency",   0.50),
    "observation_noise": ("style",        0.40),
    "action_diversity":  ("style",        0.40),
    "reasoning_clarity": ("style",        0.20),
    "truncation_ratio":  ("completeness", 1.00),
}
