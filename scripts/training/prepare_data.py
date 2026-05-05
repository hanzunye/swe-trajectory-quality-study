"""
Data Preparation: Convert trajectory subsets to SFT training format.

Pipeline:
  1. Load trajectory IDs for a subset — either from local Subset/output/
     or from HF (davongluck/swe-bench-trajectory-quality-subsets, per-config)
  2. Stream raw trajectories from HF (nebius/SWE-rebench-openhands-trajectories)
     → filter by subset IDs
  3. Convert messages → ChatML text → tokenize + assistant-only labels
  4. Save as Arrow dataset (input_ids, attention_mask, labels)

The key insight (from H200 notebook): we manually build labels where
only assistant tokens have loss computed (non-assistant tokens = -100).

Local usage (reads IDs from Subset/output/ — no extra HF download):
    python prepare_data.py --subset TopQ-500 --subset-dir /path/to/Subset/output
    python prepare_data.py --all --subset-dir /path/to/Subset/output

HF-only usage (loads IDs from the uploaded HF dataset configs):
    python prepare_data.py --subset TopQ-500
    python prepare_data.py --all --max-seq-len 32768
"""
import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
from datasets import load_dataset, Dataset
from huggingface_hub import login
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths & Constants ────────────────────────────────────
# On cloud (H200): set WORKSPACE=/workspace via env.
# Locally: defaults to the parent of the scripts/ directory (i.e. Training/).
_SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE = Path(os.environ.get("WORKSPACE", _SCRIPT_DIR.parents[1]))
DATA_DIR = WORKSPACE / "data"

# HF dataset: each subset is a separate config (e.g. "TopQ-500", "Random-500")
SUBSET_HF_REPO = "davongluck/swe-bench-trajectory-quality-subsets"

# Original trajectory dataset (contains full conversation data)
RAW_HF_DATASET = "nebius/SWE-rebench-openhands-trajectories"

# Model for tokenizer
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# All subsets matching the experiment matrix (must match Subset/output/ dir names)
ALL_SUBSETS = [
    # Block A — core matrix (exp1–exp13)
    "Random-500", "Random-1000",
    "TopQ-500", "TopQ-1000",
    "ResolvedOnly-500", "ResolvedOnly-1000",
    "BottomQ-500",
    "Ablation-NoEfficiency-500",
    "Ablation-NoStyle-500",
    "Ablation-NoB2-500",
    "Ablation-NoB3-500",
    "Ablation-NoC2-500",
    "Ablation-NoC3-500",
    # Block B — scale extension (exp14–exp15)
    "Random-2000",
    "TopQ-2000",
    # Block C — single-dimension ranking (exp16)
    "B2Only-Top500",
]


def load_trajectory_ids_local(subset_name: str, subset_dir: Path) -> set[str]:
    """
    Load trajectory IDs from a local Subset/output/{subset_name}/trajectory_ids.txt.

    This is the fast path for local runs — avoids downloading the HF metadata
    dataset and guarantees the IDs match exactly what was uploaded.
    """
    ids_file = subset_dir / subset_name / "trajectory_ids.txt"
    if not ids_file.exists():
        raise FileNotFoundError(
            f"trajectory_ids.txt not found at {ids_file}. "
            f"Run the Subset pipeline first, or omit --subset-dir to load from HF."
        )
    ids = set(ids_file.read_text().splitlines())
    ids = {i.strip() for i in ids if i.strip()}
    logger.info("Loaded %d trajectory IDs from local file: %s", len(ids), ids_file)
    return ids


def load_trajectory_ids_from_hf(subset_name: str) -> set[str]:
    """
    Load trajectory IDs from the HF dataset.

    The HF dataset uses per-subset configs (one config per subset name),
    e.g. load_dataset(REPO, "TopQ-500", split="train").
    Just read the trajectory_id column — subset selection already happened
    during upload; no need to recompute it here.
    """
    if subset_name not in ALL_SUBSETS:
        raise ValueError(f"Unknown subset: {subset_name!r}. Available: {ALL_SUBSETS}")

    logger.info("Loading subset '%s' from HF: %s ...", subset_name, SUBSET_HF_REPO)
    ds = load_dataset(SUBSET_HF_REPO, subset_name, split="train")
    ids = set(ds["trajectory_id"])
    logger.info("Loaded %d trajectory IDs for subset '%s'", len(ids), subset_name)
    return ids


def load_raw_trajectories() -> Dataset:
    """
    Download the full raw trajectory dataset to local HF cache (once).

    Uses HF datasets built-in caching: if the download was interrupted,
    re-running will resume from where it left off. Subsequent calls
    hit the local cache instantly.
    """
    logger.info("Loading raw trajectories from %s (full download, cached) ...", RAW_HF_DATASET)
    ds = load_dataset(RAW_HF_DATASET, split="train")
    logger.info("Raw dataset loaded: %d rows", len(ds))
    return ds


def trajectory_to_messages(row: dict) -> list[dict] | None:
    """
    Convert a raw HF trajectory row to ChatML messages list.

    Maps OpenHands roles to standard ChatML roles:
      system → system, user → user, assistant → assistant, tool → user
    """
    try:
        trajectory = row.get("trajectory", [])
        if isinstance(trajectory, str):
            trajectory = json.loads(trajectory)

        if not trajectory:
            return None

        messages = []
        for msg in trajectory:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""

            if role == "system":
                messages.append({"role": "system", "content": content})
            elif role == "user":
                messages.append({"role": "user", "content": content})
            elif role == "assistant":
                # Include tool calls as part of assistant content
                tool_calls = msg.get("tool_calls") or []
                if tool_calls:
                    tc_text = []
                    for tc in tool_calls:
                        func = tc.get("function", {})
                        # arguments may be a JSON string — deserialize if needed
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            try:
                                args = json.loads(args)
                            except json.JSONDecodeError:
                                pass
                        args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                        tc_text.append(
                            f"<tool_call>\n"
                            f'{{"name": "{func.get("name", "")}", '
                            f'"arguments": {args_str}}}\n'
                            f"</tool_call>"
                        )
                    full_content = (content + "\n" + "\n".join(tc_text)) if content else "\n".join(tc_text)
                    messages.append({"role": "assistant", "content": full_content})
                elif content:
                    messages.append({"role": "assistant", "content": content})
            elif role == "tool":
                # Tool responses → user role (observation)
                messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})

        if len(messages) < 2:
            return None

        return messages

    except Exception as e:
        logger.warning("Failed to convert trajectory: %s", e)
        return None


def tokenize_and_label(text: str, tokenizer, max_seq_len: int) -> dict | None:
    """
    Tokenize ChatML text and build labels with assistant-only loss.

    Strategy:
    1. Tokenize once with return_offsets_mapping=True
    2. Find ALL `<|im_start|>assistant` markers in the raw text (char positions)
    3. Use offset_mapping to map char ranges → token indices
    4. Set labels = input_ids for assistant token spans, -100 for everything else

    This ensures only assistant responses contribute to the loss,
    while system prompts, user messages, and tool outputs are masked.
    """
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_seq_len,
        padding=False,
        return_tensors=None,
        return_offsets_mapping=True,
    )

    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    labels = [-100] * len(input_ids)

    marker = "<|im_start|>assistant"

    # Find ALL assistant turn char positions in the text
    assistant_positions = []
    start_idx = 0
    while True:
        idx = text.find(marker, start_idx)
        if idx == -1:
            break
        assistant_positions.append(idx)
        start_idx = idx + 1

    if not assistant_positions:
        return None

    # Build list of (char_start, char_end) ranges for assistant turns
    assistant_char_ranges = []
    for assistant_text_pos in assistant_positions:
        search_start = assistant_text_pos + len(marker)
        end_pos_1 = text.find("<|im_end|>", search_start)
        end_pos_2 = text.find("<|im_start|>", search_start)

        if end_pos_1 == -1 and end_pos_2 == -1:
            end_text_pos = len(text)
        elif end_pos_1 == -1:
            end_text_pos = end_pos_2
        elif end_pos_2 == -1:
            end_text_pos = end_pos_1
        else:
            end_text_pos = min(end_pos_1, end_pos_2)

        assistant_char_ranges.append((assistant_text_pos, end_text_pos))

    # Map char ranges → token labels via offset_mapping (single pass)
    for i, (cs, ce) in enumerate(offsets):
        if ce == 0:  # skip special tokens with zero-width offsets
            continue
        for range_start, range_end in assistant_char_ranges:
            if cs >= range_start and cs < range_end:
                labels[i] = input_ids[i]
                break

    # Sanity check: at least some tokens should be unmasked
    num_unmasked = sum(1 for l in labels if l != -100)
    if num_unmasked == 0:
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": enc["attention_mask"],
        "labels": labels,
    }


def prepare_subset(
    subset_name: str,
    max_seq_len: int = 32768,
    raw_ds: Dataset | None = None,
    subset_dir: Path | None = None,
):
    """
    Prepare training data for a single subset.

    1. Load trajectory IDs — from local subset_dir if provided, else from HF
    2. Filter nebius raw trajectories by those IDs
    3. Convert to ChatML, tokenize, build assistant-only labels
    4. Save as Arrow dataset
    """
    output_path = DATA_DIR / subset_name

    if output_path.exists() and (output_path / "dataset_info.json").exists():
        logger.info("Subset '%s' already prepared at %s, skipping.", subset_name, output_path)
        return output_path

    # Load tokenizer
    logger.info("Loading tokenizer: %s", MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Step 1: Get trajectory IDs (local fast path or HF)
    if subset_dir is not None:
        target_ids = load_trajectory_ids_local(subset_name, subset_dir)
    else:
        target_ids = load_trajectory_ids_from_hf(subset_name)

    # Step 2: Load raw trajectories (use pre-loaded dataset or download)
    if raw_ds is None:
        raw_ds = load_raw_trajectories()
    logger.info("Looking for %d trajectory IDs in %d rows ...", len(target_ids), len(raw_ds))

    processed_examples = []
    found = 0
    skipped = 0
    truncated_count = 0
    scanned = 0

    for row in raw_ds:
        scanned += 1
        traj_id = row.get("trajectory_id", "")
        if traj_id not in target_ids:
            if scanned % 10000 == 0:
                logger.info("  Scanned %d rows, found %d/%d ...", scanned, found, len(target_ids))
            continue

        found += 1

        # Convert to ChatML messages
        messages = trajectory_to_messages(row)
        if messages is None:
            skipped += 1
            if found >= len(target_ids):
                break
            continue

        # Apply chat template → text
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        # Track how many are truncated (for logging), but do NOT skip them.
        # Right truncation: tokenize_and_label will truncate to max_seq_len,
        # keeping the beginning of the trajectory (system prompt + early turns).
        # This matches the experiment design: "Right truncation (keep beginning)"
        quick_len = len(tokenizer.encode(text, add_special_tokens=True))
        if quick_len > max_seq_len:
            truncated_count += 1
            if truncated_count <= 5:
                logger.info("  Truncated: %s — %d tokens → %d",
                            traj_id[:30], quick_len, max_seq_len)

        # Tokenize + build assistant-only labels (truncation happens inside)
        result = tokenize_and_label(text, tokenizer, max_seq_len)
        if result is None:
            skipped += 1
            if found >= len(target_ids):
                break
            continue

        processed_examples.append(result)

        if len(processed_examples) % 50 == 0:
            logger.info(
                "  Progress: scanned=%d, found=%d/%d, processed=%d, truncated=%d, skipped=%d",
                scanned, found, len(target_ids), len(processed_examples), truncated_count, skipped,
            )

        # Early exit when all IDs found
        if found >= len(target_ids):
            break

    # Convert to HF Dataset and save as Arrow
    if not processed_examples:
        logger.error("No examples processed for subset '%s'!", subset_name)
        return None

    dataset = Dataset.from_dict({
        "input_ids": [ex["input_ids"] for ex in processed_examples],
        "attention_mask": [ex["attention_mask"] for ex in processed_examples],
        "labels": [ex["labels"] for ex in processed_examples],
    })

    output_path.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))

    # Print stats
    all_lens = [len(ex["input_ids"]) for ex in processed_examples]
    all_unmasked = [sum(1 for l in ex["labels"] if l != -100) for ex in processed_examples]
    total_tokens = sum(all_lens)
    total_unmasked = sum(all_unmasked)

    logger.info("=" * 60)
    logger.info("Subset '%s' preparation complete:", subset_name)
    logger.info("  Scanned: %d rows from raw dataset", scanned)
    logger.info("  Found: %d / %d target IDs", found, len(target_ids))
    logger.info("  Processed: %d examples", len(processed_examples))
    logger.info("  Truncated (> %d tokens): %d", max_seq_len, truncated_count)
    logger.info("  Skipped (invalid): %d", skipped)
    if all_lens:
        logger.info("  Sequence lengths: min=%d, max=%d, mean=%d, median=%d",
                    min(all_lens), max(all_lens), int(np.mean(all_lens)), int(np.median(all_lens)))
        logger.info("  Assistant tokens: %d / %d total (%.1f%%)",
                    total_unmasked, total_tokens, 100 * total_unmasked / total_tokens)
    logger.info("  Saved to: %s", output_path)
    logger.info("=" * 60)

    return output_path


def main():
    global MODEL_ID, DATA_DIR

    parser = argparse.ArgumentParser(description="Prepare training data from trajectory subsets")
    parser.add_argument("--subset", type=str, help="Subset name (e.g., TopQ-500, Random-500)")
    parser.add_argument("--all", action="store_true", help="Prepare all subsets")
    parser.add_argument("--max-seq-len", type=int, default=32768, help="Max sequence length (default: 32768)")
    parser.add_argument("--model-id", type=str, default=MODEL_ID, help="Tokenizer model ID")
    parser.add_argument(
        "--subset-dir", type=str, default=None,
        help=(
            "Path to local Subset/output/ directory (e.g. /path/to/Subset/output). "
            "If provided, trajectory IDs are read from local trajectory_ids.txt files "
            "instead of downloading from HF — much faster for local runs."
        ),
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help=(
            "Output directory for prepared Arrow datasets. "
            "Defaults to $WORKSPACE/data (or ./data if WORKSPACE is not set). "
            "Example: --data-dir /path/to/Training/data"
        ),
    )
    args = parser.parse_args()

    if args.model_id != MODEL_ID:
        MODEL_ID = args.model_id

    if args.data_dir:
        DATA_DIR = Path(args.data_dir)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Output data directory: %s", DATA_DIR)

    subset_dir = Path(args.subset_dir) if args.subset_dir else None
    if subset_dir is not None:
        if not subset_dir.exists():
            parser.error(f"--subset-dir does not exist: {subset_dir}")
        logger.info("Using local subset directory: %s", subset_dir)
    else:
        logger.info("No --subset-dir given; will load trajectory IDs from HF.")

    # HF login (needed for raw trajectory download; optional for public repos)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    # Download raw trajectories once (cached by HF datasets library)
    raw_ds = load_raw_trajectories()

    if args.all:
        for subset in ALL_SUBSETS:
            try:
                prepare_subset(subset, args.max_seq_len, raw_ds=raw_ds, subset_dir=subset_dir)
            except Exception as e:
                logger.error("Failed to prepare '%s': %s", subset, e)
    elif args.subset:
        prepare_subset(args.subset, args.max_seq_len, raw_ds=raw_ds, subset_dir=subset_dir)
    else:
        parser.error("Specify --subset NAME or --all")


if __name__ == "__main__":
    main()
