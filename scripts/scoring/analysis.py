"""
Per-trajectory statistical analysis.

Computes for each trajectory:
  1. Token count (total tokens via tiktoken)
  2. Turn count (assistant actions)
  3. Tool call distribution (type → frequency)
  4. Outcome (submit vs error/timeout)
  5. Patch validity (exit_status + resolved)
  6. Repository info
  7. Tool output length (average observation length)
  8. v3 metrics: B1 redundant commands, B2 error-retry cycles,
                 C1 observation noise ratio, C3 observation utilization
"""
import json
import re
import logging
from collections import Counter
from dataclasses import dataclass, field

try:
    from .scoring_config import (
        TIKTOKEN_ENCODING,
        TEST_KEYWORDS,
        B1_WINDOW_SIZE,
        B2_ERROR_KEYWORDS,
        C1_NOISE_PATTERNS,
        C3_FILE_EXTENSIONS,
    )
except ImportError:
    from scoring_config import (
        TIKTOKEN_ENCODING,
        TEST_KEYWORDS,
        B1_WINDOW_SIZE,
        B2_ERROR_KEYWORDS,
        C1_NOISE_PATTERNS,
        C3_FILE_EXTENSIONS,
    )

logger = logging.getLogger(__name__)

# Lazy-init global encoder (thread-safe, reuse across calls)
_encoder = None

# Pre-compiled regex for C3 filename extraction
_C3_FILE_RE = re.compile(
    r'[\w./\-]+\.(?:' + '|'.join(C3_FILE_EXTENSIONS) + r')\b',
    re.IGNORECASE,
)
# Pre-compiled regex for C3 error class name extraction
_C3_ERROR_CLASS_RE = re.compile(r'\b\w+(?:Error|Exception|Warning)\b')

# C1: lowercase noise patterns list (pre-lowered for fast `in` check)
_C1_NOISE_LOWER = [p.lower() for p in C1_NOISE_PATTERNS]


def _get_encoder():
    global _encoder
    if _encoder is None:
        import tiktoken
        _encoder = tiktoken.get_encoding(TIKTOKEN_ENCODING)
    return _encoder


@dataclass
class TrajectoryStats:
    """Statistics computed for a single trajectory."""
    trajectory_id: str
    instance_id: str
    repo: str

    # 1. Token count
    total_tokens: int = 0

    # 2. Turn count
    assistant_turns: int = 0

    # 3. Tool call distribution
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    total_tool_calls: int = 0

    # 4. Outcome
    exit_status: str = ""
    ends_with_submit: bool = False
    is_error_or_timeout: bool = False

    # 5. Patch validity
    resolved: int = 0           # 1 = solved, 0 = not
    has_patch: bool = False
    patch_length: int = 0

    # 7. Tool output / observation length
    observation_lengths: list[int] = field(default_factory=list)
    avg_observation_length: float = 0.0

    # v2 metrics (kept for backwards compat)
    unique_action_ratio: float = 1.0    # unique (tool, args[:200]) / total_tool_calls
    test_alignment_score: float = 0.0   # 1.0 if trajectory runs tests, else 0.0

    # ── v3 raw metrics ──────────────────────────────────
    # B1: fraction of actions that are redundant within B1_WINDOW_SIZE
    redundant_cmds_ratio: float = 0.0
    # B2: number of detected error-retry cycles; denominator for normalization
    error_retry_count: int = 0
    total_action_pairs: int = 0         # max possible cycles (num_actions - 1)
    # C1: fraction of observation characters that are "noise" (errors/tracebacks)
    obs_noise_char_ratio: float = 0.0
    # C3: fraction of extracted observation keywords used in later actions
    obs_utilization_ratio: float = 1.0  # 1.0 when no keywords extracted (no penalty)

    def to_dict(self) -> dict:
        return {
            "trajectory_id": self.trajectory_id,
            "instance_id": self.instance_id,
            "repo": self.repo,
            "total_tokens": self.total_tokens,
            "assistant_turns": self.assistant_turns,
            "tool_call_counts": self.tool_call_counts,
            "total_tool_calls": self.total_tool_calls,
            "exit_status": self.exit_status,
            "ends_with_submit": self.ends_with_submit,
            "is_error_or_timeout": self.is_error_or_timeout,
            "resolved": self.resolved,
            "has_patch": self.has_patch,
            "patch_length": self.patch_length,
            "avg_observation_length": self.avg_observation_length,
            "num_observations": len(self.observation_lengths),
            "unique_action_ratio": self.unique_action_ratio,
            "test_alignment_score": self.test_alignment_score,
            # v3
            "redundant_cmds_ratio": self.redundant_cmds_ratio,
            "error_retry_count": self.error_retry_count,
            "total_action_pairs": self.total_action_pairs,
            "obs_noise_char_ratio": self.obs_noise_char_ratio,
            "obs_utilization_ratio": self.obs_utilization_ratio,
        }


def _count_tokens(text: str) -> int:
    """Count tokens in a text string using tiktoken."""
    if not text:
        return 0
    return len(_get_encoder().encode(text, disallowed_special=()))


def _extract_text_for_tokenization(msg: dict) -> str:
    """Extract all text content from a message for token counting."""
    parts = []
    content = msg.get("content")
    if content:
        parts.append(str(content))

    # Include tool call arguments in token count
    for tc in msg.get("tool_calls") or []:
        func = tc.get("function", {})
        name = func.get("name", "")
        if name:
            parts.append(name)
        args = func.get("arguments")
        if isinstance(args, dict):
            parts.append(json.dumps(args))
        elif isinstance(args, str):
            parts.append(args)

    return " ".join(parts)


# ═══════════════════════════════════════════════════════════
#  v3 metric helpers
# ═══════════════════════════════════════════════════════════

def _compute_b1_redundant_cmds(action_sigs: list[tuple[str, str]]) -> float:
    """
    B1 raw: fraction of actions that repeat a command seen in the
    previous B1_WINDOW_SIZE steps.

    Returns redundant_count / total_actions, or 0.0 if fewer than 2 actions.
    """
    n = len(action_sigs)
    if n < 2:
        return 0.0
    count = 0
    for i in range(1, n):
        window_start = max(0, i - B1_WINDOW_SIZE)
        if action_sigs[i] in set(action_sigs[window_start:i]):
            count += 1
    return count / n


def _obs_has_error(obs_text: str) -> bool:
    """Return True if the observation contains any B2 error keyword."""
    lower = obs_text.lower()
    return any(kw in lower for kw in B2_ERROR_KEYWORDS)


def _actions_similar(sig_a: tuple[str, str], sig_b: tuple[str, str]) -> bool:
    """
    Two actions are 'similar' if they share the same tool name.
    (Same tool name after an error = retry cycle.)
    """
    return sig_a[0] == sig_b[0]


def _compute_b2_error_retry(
    action_sigs: list[tuple[str, str]],
    obs_texts: list[str],
) -> tuple[int, int]:
    """
    B2 raw: detect "action → error observation → similar action" cycles.

    action_sigs and obs_texts are parallel lists of equal length:
      action_sigs[i] = tool signature of the i-th action
      obs_texts[i]   = text of the observation that followed action i
                       (empty string if the action had no following observation)

    Returns (cycle_count, total_action_pairs) where total_action_pairs = n - 1.
    """
    n = len(action_sigs)
    if n < 2:
        return 0, 0

    cycles = 0
    for i in range(n - 1):
        obs = obs_texts[i]
        if obs and _obs_has_error(obs) and _actions_similar(action_sigs[i], action_sigs[i + 1]):
            cycles += 1

    return cycles, n - 1


def _compute_c1_obs_noise(observation_texts: list[str]) -> float:
    """
    C1 raw: fraction of observation characters that belong to noisy lines
    (lines containing traceback / error / warning patterns).

    Uses character count as a token proxy for efficiency.
    Returns 0.0 if no observations.
    """
    total_chars = 0
    noise_chars = 0
    for obs in observation_texts:
        if not obs:
            continue
        total_chars += len(obs)
        for line in obs.splitlines():
            line_lower = line.lower()
            if any(pat in line_lower for pat in _C1_NOISE_LOWER):
                noise_chars += len(line)
    if total_chars == 0:
        return 0.0
    return noise_chars / total_chars


def _extract_c3_keywords(obs_text: str) -> set[str]:
    """
    Extract filenames and error class names from an observation text.
    Returns a set of lowercase keyword strings used for subsequent-action matching.

    For file paths (e.g., src/utils.py) we store only the **basename** (utils.py)
    so that matching succeeds whether the agent references the file with or without
    the leading directory path.  Error class names (AttributeError etc.) are kept
    as-is since they contain no path component.
    """
    keywords: set[str] = set()
    # Filenames — keep only basename to tolerate path-prefix differences
    for m in _C3_FILE_RE.findall(obs_text):
        full = m.strip().lower()
        basename = full.rsplit("/", 1)[-1]   # strip path prefix if present
        if len(basename) > 4:               # skip trivial matches like "a.py"
            keywords.add(basename)
    # Error class names (e.g., AttributeError, ValueError)
    for m in _C3_ERROR_CLASS_RE.findall(obs_text):
        keywords.add(m.lower())
    return keywords


def _compute_c3_obs_utilization(
    action_sigs: list[tuple[str, str]],
    action_texts: list[str],
    obs_texts: list[str],
) -> float:
    """
    C3 raw: for each observation, extract filenames and error keywords,
    then check whether each keyword appears in any subsequent action text.

    Returns utilized_keywords / total_extracted_keywords.
    Returns 1.0 if no keywords were extracted (no penalty for sparse observations).

    Args:
        action_sigs: not used directly, kept for interface consistency
        action_texts: stringified tool call text for each action
        obs_texts: observation text that follows each action (parallel to action_texts)
    """
    n = len(obs_texts)
    total_extracted = 0
    total_utilized = 0

    for i, obs in enumerate(obs_texts):
        if not obs:
            continue
        keywords = _extract_c3_keywords(obs)
        if not keywords:
            continue

        # Concatenate all subsequent action texts for fast substring search
        subsequent = " ".join(action_texts[i + 1:]).lower() if i + 1 < n else ""

        for kw in keywords:
            total_extracted += 1
            if kw in subsequent:
                total_utilized += 1

    if total_extracted == 0:
        return 1.0
    return total_utilized / total_extracted


# ═══════════════════════════════════════════════════════════
#  Main trajectory analyzer
# ═══════════════════════════════════════════════════════════

def analyze_trajectory(row: dict) -> TrajectoryStats:
    """
    Compute all statistics for a single trajectory row.

    Args:
        row: A single row from the dataset (after cleaning/deserialization).

    Returns:
        TrajectoryStats with all computed metrics.
    """
    stats = TrajectoryStats(
        trajectory_id=row.get("trajectory_id", ""),
        instance_id=row.get("instance_id", ""),
        repo=row.get("repo", ""),
        exit_status=row.get("exit_status", ""),
        resolved=row.get("resolved", 0),
    )

    trajectory = row.get("trajectory", [])
    tool_counter = Counter()
    total_tokens = 0

    # For v2 redundancy (kept for backwards compat)
    action_signatures: list[tuple[str, str]] = []
    test_found = False

    # For v3: interleaved action/observation sequence
    # action_sigs[i]  = (func_name, args_prefix) of i-th action
    # action_texts[i] = full stringified tool-call text for i-th action
    # obs_after[i]    = observation text that immediately followed action i
    #                   (empty string if no observation followed before next action)
    v3_action_sigs: list[tuple[str, str]] = []
    v3_action_texts: list[str] = []
    v3_obs_after: list[str] = []   # parallel to v3_action_sigs, set when obs arrives

    # Accumulate observation texts for C1
    all_obs_texts: list[str] = []

    # State machine: track whether we're between actions waiting for an observation
    _last_action_idx = -1   # index into v3_action_sigs for the most recent action

    for msg in trajectory:
        role = msg.get("role", "")

        # ── Token counting ──
        text = _extract_text_for_tokenization(msg)
        total_tokens += _count_tokens(text)

        if role == "assistant":
            stats.assistant_turns += 1
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                # Start a new action slot
                combined_tool_text_parts = []

                for tc in tool_calls:
                    func = tc.get("function", {})
                    func_name = func.get("name", "unknown")
                    tool_counter[func_name] += 1

                    args = func.get("arguments", "")
                    if isinstance(args, dict):
                        args_str = json.dumps(args, sort_keys=True)
                    else:
                        args_str = str(args)
                    args_prefix = args_str[:200]

                    # v2 action signature (tool, args[:200])
                    action_signatures.append((func_name, args_prefix))

                    # v3 action signature (first tool call dominates)
                    if not combined_tool_text_parts:  # use first tool call for sig
                        v3_action_sigs.append((func_name, args_prefix))

                    combined_tool_text_parts.append(f"{func_name} {args_str}")

                    # Test alignment (v2/v3 shared)
                    if not test_found and func_name == "execute_bash":
                        if any(kw in args_prefix.lower() for kw in TEST_KEYWORDS):
                            test_found = True

                # Record combined action text and a placeholder obs slot
                v3_action_texts.append(" ".join(combined_tool_text_parts))
                v3_obs_after.append("")   # will be filled when observation arrives
                _last_action_idx = len(v3_action_sigs) - 1

        elif role in ("user", "tool"):
            content = msg.get("content", "")
            if content:
                obs_str = str(content)
                stats.observation_lengths.append(len(obs_str))
                all_obs_texts.append(obs_str)

                # Attach observation to the most recent action slot
                if _last_action_idx >= 0 and not v3_obs_after[_last_action_idx]:
                    v3_obs_after[_last_action_idx] = obs_str

    stats.total_tokens = total_tokens
    stats.tool_call_counts = dict(tool_counter)
    stats.total_tool_calls = sum(tool_counter.values())

    # ── v2: Unique action ratio ──
    if action_signatures:
        unique_sigs = len(set(action_signatures))
        stats.unique_action_ratio = unique_sigs / len(action_signatures)
    else:
        stats.unique_action_ratio = 1.0

    # ── v2: Test alignment ──
    stats.test_alignment_score = 1.0 if test_found else 0.0

    # ── Outcome ──
    exit_status = str(stats.exit_status).strip().lower()
    stats.ends_with_submit = exit_status == "submit"
    stats.is_error_or_timeout = not stats.ends_with_submit

    # ── Patch validity ──
    model_patch = row.get("model_patch", "")
    stats.has_patch = bool(model_patch and model_patch.strip())
    stats.patch_length = len(model_patch) if model_patch else 0

    # ── Avg observation length ──
    if stats.observation_lengths:
        stats.avg_observation_length = sum(stats.observation_lengths) / len(
            stats.observation_lengths
        )

    # ── v3: B1 Redundant commands ──
    stats.redundant_cmds_ratio = _compute_b1_redundant_cmds(v3_action_sigs)

    # ── v3: B2 Error-retry cycles ──
    cycles, pairs = _compute_b2_error_retry(v3_action_sigs, v3_obs_after)
    stats.error_retry_count = cycles
    stats.total_action_pairs = pairs

    # ── v3: C1 Observation noise ──
    stats.obs_noise_char_ratio = _compute_c1_obs_noise(all_obs_texts)

    # ── v3: C3 Observation utilization ──
    stats.obs_utilization_ratio = _compute_c3_obs_utilization(
        v3_action_sigs, v3_action_texts, v3_obs_after
    )

    return stats
