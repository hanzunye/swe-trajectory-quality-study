"""
eval_perplexity_new.py — Perplexity/Loss 评估（含 exp14/15/16 新实验）
A100 服务器版本

在原 eval_perplexity.py 基础上扩展：
  exp14 — Random-2000      (Experiment B: scale extension)
  exp15 — TopQ-2000        (Experiment B: scale extension)
  exp16 — B2Only-Top500    (Experiment C: B2-only vs composite ablation)

新增特性：
  - --outputs-dir 参数支持自定义 adapter 路径
  - 评估完成后自动生成论文级插图（scaling curve + ablation 对比）
  - 兼容已有 perplexity_results.csv/json（增量合并）

Usage:
    # 只评估新实验
    python eval_perplexity_new.py --models exp14 exp15 exp16

    # 全量评估（含原 14 个实验）
    python eval_perplexity_new.py

    # 自定义 adapter 路径
    python eval_perplexity_new.py --models exp14 exp15 exp16 --outputs-dir /workspace/outputs
"""

import argparse
import gc
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────
WORKSPACE    = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
DATA_ROOT    = WORKSPACE / "data"
EVAL_DATA    = DATA_ROOT / "eval_data"
OUTPUT_DIR   = DATA_ROOT / "eval_results"
OUTPUTS_ROOT = WORKSPACE / "outputs"   # LoRA adapter 默认目录

# ── A100 推理配置 ──────────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-Coder-7B-Instruct"
TORCH_DTYPE  = torch.bfloat16           # A100 原生支持 bfloat16
DEVICE       = "cuda:0"
MAX_SEQ_LEN  = 32768
BATCH_SIZE   = 8                        # A100 40/80GB 显存充裕

# 完整实验矩阵（exp1-13 原有 + exp14-16 新增）
EXPERIMENT_MAP = {
    "baseline":  None,                  # 基础模型，无 LoRA
    # ── Block 1: Core ──
    "exp1":      "exp1/final",          # Random-500
    "exp2":      "exp2/final",          # Random-1000
    "exp3":      "exp3/final",          # TopQ-500
    "exp4":      "exp4/final",          # TopQ-1000
    "exp5":      "exp5/final",          # ResolvedOnly-500
    "exp6":      "exp6/final",          # ResolvedOnly-1000
    "exp7":      "exp7/final",          # BottomQ-500
    # ── Block 2: Ablation ──
    "exp8":      "exp8/final",          # Ablation-NoEfficiency-500
    "exp9":      "exp9/final",          # Ablation-NoStyle-500
    # ── Block 3: Sub-dimension ──
    "exp10":     "exp10/final",         # Ablation-NoB2-500
    "exp11":     "exp11/final",         # Ablation-NoB3-500
    "exp12":     "exp12/final",         # Ablation-NoC2-500
    "exp13":     "exp13/final",         # Ablation-NoC3-500
    # ── NEW: Experiment B (Scale) + C (B2-only) ──
    "exp14":     "exp14/final",         # Random-2000
    "exp15":     "exp15/final",         # TopQ-2000
    "exp16":     "exp16/final",         # B2Only-Top500
}

# 显示名称映射（用于图表）
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "exp1": "Random-500", "exp2": "Random-1000", "exp14": "Random-2000",
    "exp3": "TopQ-500",   "exp4": "TopQ-1000",   "exp15": "TopQ-2000",
    "exp5": "Resolved-500", "exp6": "Resolved-1000",
    "exp7": "BottomQ-500",
    "exp8": "NoEfficiency", "exp9": "NoStyle",
    "exp10": "NoB2", "exp11": "NoB3", "exp12": "NoC2", "exp13": "NoC3",
    "exp16": "B2Only-500",
}

TEST_SPLITS = ["gold", "random", "low_q"]


# ── 数据格式工具 ──────────────────────────────────────────────────────────
def trajectory_to_chatml(row: dict, tokenizer) -> str | None:
    """将 raw 轨迹行转换为 ChatML 格式文本。"""
    import json as _json

    trajectory = row.get("trajectory", [])
    if isinstance(trajectory, str):
        try:
            trajectory = _json.loads(trajectory)
        except Exception:
            return None

    if not trajectory:
        return None

    messages = []
    for msg in trajectory:
        role    = msg.get("role", "")
        content = msg.get("content", "") or ""

        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    func    = tc.get("function", {})
                    args    = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = _json.loads(args)
                        except Exception:
                            pass
                    args_str = _json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    tc_parts.append(
                        f'<tool_call>\n{{"name": "{func.get("name","")}", "arguments": {args_str}}}\n</tool_call>'
                    )
                full_content = (content + "\n" + "\n".join(tc_parts)) if content else "\n".join(tc_parts)
                messages.append({"role": "assistant", "content": full_content})
            elif content:
                messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})

    if len(messages) < 2:
        return None

    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)


def tokenize_with_assistant_labels(text: str, tokenizer) -> dict | None:
    """Tokenize 文本并构建 assistant-only labels。"""
    enc = tokenizer(
        text,
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"][0]
    labels    = torch.full_like(input_ids, -100)

    marker = "<|im_start|>assistant"
    pos    = 0
    while True:
        idx = text.find(marker, pos)
        if idx == -1:
            break

        prefix_ids = tokenizer(
            text[:idx], add_special_tokens=False,
            truncation=True, max_length=MAX_SEQ_LEN,
        )["input_ids"]
        start_tok = len(prefix_ids)

        search_from = idx + len(marker)
        e1 = text.find("<|im_end|>",  search_from)
        e2 = text.find("<|im_start|>", search_from)
        if e1 == -1 and e2 == -1:
            end_char = len(text)
        elif e1 == -1:
            end_char = e2
        elif e2 == -1:
            end_char = e1
        else:
            end_char = min(e1, e2)

        end_ids = tokenizer(
            text[:end_char], add_special_tokens=False,
            truncation=True, max_length=MAX_SEQ_LEN,
        )["input_ids"]
        end_tok = len(end_ids)

        for i in range(start_tok, min(end_tok, len(input_ids))):
            labels[i] = input_ids[i]

        pos = idx + 1

    if (labels != -100).sum() == 0:
        return None

    return {
        "input_ids":      input_ids,
        "attention_mask": enc["attention_mask"][0],
        "labels":         labels,
    }


# ── 数据加载 ──────────────────────────────────────────────────────────────
def _process_one(item: tuple) -> dict | None:
    line_no, line, tokenizer = item
    row = json.loads(line.strip())
    tid = row.get("trajectory_id", f"row_{line_no}")
    text = trajectory_to_chatml(row, tokenizer)
    if text is None:
        return None
    enc = tokenize_with_assistant_labels(text, tokenizer)
    if enc is None:
        return None
    return {"trajectory_id": tid, **enc}


def load_test_split(split_name: str, tokenizer) -> list[dict]:
    """加载并 tokenize 测试集。"""
    jsonl_path = EVAL_DATA / f"{split_name}.jsonl"
    cache_path = EVAL_DATA / f"{split_name}.cache.pt"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"测试集文件不存在: {jsonl_path}\n请先运行 build_test_set.py")

    if cache_path.exists() and cache_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
        logger.info("[%s] 从缓存加载: %s", split_name, cache_path)
        try:
            samples = torch.load(cache_path, weights_only=True)
            if not isinstance(samples, list):
                raise TypeError(f"unexpected cache type: {type(samples)!r}")
            logger.info("[%s] 缓存命中，%d 条", split_name, len(samples))
            return samples
        except Exception as exc:
            logger.warning("[%s] 缓存读取失败，重新生成: %s", split_name, exc)

    logger.info("[%s] 未找到缓存，开始预处理...", split_name)
    with open(jsonl_path) as f:
        lines = f.readlines()

    num_workers = min(20, os.cpu_count() or 1)
    items = [(i, line, tokenizer) for i, line in enumerate(lines)]

    results = [None] * len(items)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {executor.submit(_process_one, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            results[future_to_idx[future]] = future.result()

    samples = [r for r in results if r is not None]
    torch.save(samples, cache_path)
    logger.info("[%s] %d 条已缓存: %s", split_name, len(samples), cache_path)
    return samples


# ── 模型加载 ──────────────────────────────────────────────────────────────
def load_base_model(tokenizer_only: bool = False):
    logger.info("加载基础模型: %s (BF16, Flash Attention 2)", BASE_MODEL)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer_only:
        return None, tokenizer

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=TORCH_DTYPE,
        device_map={"": DEVICE},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.config.use_cache = True
    model.eval()
    mem_gb = torch.cuda.memory_allocated(0) / 1024**3
    logger.info("模型加载完成，显存: %.2f GB", mem_gb)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    return model, tokenizer


def load_lora_model(base_model, adapter_path: str):
    full_path = Path(adapter_path)
    if not full_path.exists():
        raise FileNotFoundError(f"LoRA adapter 不存在: {full_path}")
    logger.info("挂载 LoRA adapter: %s", full_path)
    model = PeftModel.from_pretrained(
        base_model, str(full_path),
        torch_dtype=TORCH_DTYPE, is_trainable=False,
    )
    model.eval()
    return model


# ── Dataset / collate ─────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_len = max(s["input_ids"].size(0) for s in batch)
    input_ids_list, attn_mask_list, labels_list, tids = [], [], [], []
    for s in batch:
        pad = max_len - s["input_ids"].size(0)
        input_ids_list.append(torch.cat([s["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        attn_mask_list.append(torch.cat([s["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
        labels_list.append(torch.cat([s["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        tids.append(s["trajectory_id"])
    return {
        "trajectory_ids": tids,
        "input_ids":      torch.stack(input_ids_list),
        "attention_mask": torch.stack(attn_mask_list),
        "labels":         torch.stack(labels_list),
    }


# ── Loss 计算 ─────────────────────────────────────────────────────────────
@torch.no_grad()
def compute_batch_loss(model, input_ids, attention_mask, labels):
    input_ids      = input_ids.to(DEVICE, non_blocking=True)
    attention_mask = attention_mask.to(DEVICE, non_blocking=True)
    labels         = labels.to(DEVICE, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", dtype=TORCH_DTYPE):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    shift_logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:].contiguous()
    loss_fct = nn.CrossEntropyLoss(reduction="none")
    results = []
    for i in range(shift_logits.size(0)):
        token_loss = loss_fct(shift_logits[i].float(), shift_labels[i])
        mask       = (shift_labels[i] != -100).float()
        num_tokens = int(mask.sum().item())
        if num_tokens == 0:
            results.append((float("nan"), 0))
        else:
            results.append(((token_loss * mask).sum().item() / num_tokens, num_tokens))
    del shift_logits, outputs
    return results


@torch.no_grad()
def evaluate_model_on_split(model, samples, split_name, exp_name):
    losses, n_tokens = [], []
    failed, processed = 0, 0

    loader = DataLoader(
        TrajectoryDataset(samples),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, collate_fn=collate_fn,
    )

    t0 = time.time()
    for batch in loader:
        try:
            batch_results = compute_batch_loss(
                model, batch["input_ids"], batch["attention_mask"], batch["labels"],
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                logger.warning(
                    "  [%s/%s] OOM at batch (size=%d), falling back to single-sample...",
                    exp_name, split_name, batch["input_ids"].size(0),
                )
                torch.cuda.empty_cache()
                gc.collect()
                batch_results = []
                for i in range(batch["input_ids"].size(0)):
                    try:
                        single = compute_batch_loss(
                            model,
                            batch["input_ids"][i:i+1],
                            batch["attention_mask"][i:i+1],
                            batch["labels"][i:i+1],
                        )
                        batch_results.extend(single)
                    except (torch.cuda.OutOfMemoryError, RuntimeError):
                        logger.warning(
                            "  [%s/%s] single sample still OOM (seq_len=%d), skipping",
                            exp_name, split_name, batch["input_ids"].size(1),
                        )
                        batch_results.append((float("nan"), 0))
                        torch.cuda.empty_cache()
                        gc.collect()
            else:
                raise

        for loss, n in batch_results:
            if not np.isnan(loss):
                losses.append(loss)
                n_tokens.append(n)
            else:
                failed += 1
        processed += len(batch_results)
        if processed % 50 < BATCH_SIZE:
            logger.info(
                "  [%s/%s] %d/%d  avg_loss=%.4f  (%.0fs)",
                exp_name, split_name, processed, len(samples),
                float(np.mean(losses)) if losses else float("nan"),
                time.time() - t0,
            )

    if not losses:
        return {"mean_loss": float("nan"), "std_loss": float("nan"),
                "perplexity": float("nan"), "n_valid": 0, "n_failed": failed}

    mean_loss = float(np.mean(losses))
    std_loss  = float(np.std(losses))
    ppl       = float(np.exp(mean_loss))
    logger.info(
        "[%s/%s] mean_loss=%.4f  std=%.4f  PPL=%.2f  valid=%d  failed=%d",
        exp_name, split_name, mean_loss, std_loss, ppl, len(losses), failed,
    )
    return {
        "mean_loss": mean_loss, "std_loss": std_loss, "perplexity": ppl,
        "losses": losses, "n_valid": len(losses), "n_failed": failed,
        "mean_assistant_tokens": float(np.mean(n_tokens)),
    }


# ── 主评估循环 ────────────────────────────────────────────────────────────
def run_evaluation(target_models, target_splits, outputs_root, force=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _, tokenizer = load_base_model(tokenizer_only=True)

    split_data = {}
    for split_name in target_splits:
        split_data[split_name] = load_test_split(split_name, tokenizer)

    base_model, _ = load_base_model()

    json_path = OUTPUT_DIR / "perplexity_results.json"
    if json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        logger.info("已加载历史结果: %s", list(all_results.keys()))
    else:
        all_results = {}

    for exp_name in target_models:
        logger.info("=" * 60)
        if not force and exp_name in all_results:
            logger.info("跳过 %s（已有结果）", exp_name)
            continue
        logger.info("评估: %s", exp_name)
        adapter_rel_path = EXPERIMENT_MAP.get(exp_name)

        if adapter_rel_path is None:
            model = base_model
        else:
            full_adapter_path = outputs_root / adapter_rel_path
            model = load_lora_model(base_model, str(full_adapter_path))

        exp_results = {"exp_name": exp_name}
        for split_name in target_splits:
            exp_results[split_name] = evaluate_model_on_split(
                model, split_data[split_name], split_name, exp_name
            )

        all_results[exp_name] = exp_results

        if adapter_rel_path is not None:
            base_model = model.get_base_model()
            del model
            for a in ("peft_config", "active_adapter", "active_adapters"):
                try: delattr(base_model, a)
                except AttributeError: pass
            gc.collect()
            torch.cuda.empty_cache()

        _save_results(all_results, OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("评估完成！")
    _save_results(all_results, OUTPUT_DIR)
    return all_results


def _save_results(all_results, output_dir):
    json_path = output_dir / "perplexity_results.json"
    tmp_path  = json_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    tmp_path.replace(json_path)

    rows = []
    for exp_name, exp_data in all_results.items():
        for split_name in TEST_SPLITS:
            if split_name not in exp_data:
                continue
            r = exp_data[split_name]
            rows.append({
                "model": exp_name, "split": split_name,
                "mean_loss": r.get("mean_loss"), "std_loss": r.get("std_loss"),
                "perplexity": r.get("perplexity"), "n_valid": r.get("n_valid"),
            })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_dir / "perplexity_results.csv", index=False)
        logger.info("CSV 已保存")


# ── 论文级插图 ────────────────────────────────────────────────────────────
def generate_publication_figures(output_dir: Path):
    """读取 perplexity_results.csv 生成论文级插图。"""
    csv_path = output_dir / "perplexity_results.csv"
    if not csv_path.exists():
        logger.warning("perplexity_results.csv 不存在，跳过绘图")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
        "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.linewidth": 0.8, "axes.grid": True, "grid.alpha": 0.25,
    })

    df = pd.read_csv(csv_path)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Figure: Scaling Curve（500 → 1000 → 2000）──────────────────────────
    fig, ax = plt.subplots(figsize=(5.0, 3.5))
    scales = [500, 1000, 2000]

    for strategy, exps, color, marker in [
        ("Random",  ["exp1", "exp2", "exp14"], "#1f77b4", "s"),
        ("TopQ",    ["exp3", "exp4", "exp15"], "#ff7f0e", "o"),
    ]:
        gold_losses = []
        available_scales = []
        for exp, scale in zip(exps, scales):
            row = df[(df["model"] == exp) & (df["split"] == "gold")]
            if not row.empty:
                gold_losses.append(float(row["mean_loss"].iloc[0]))
                available_scales.append(scale)
        if gold_losses:
            ax.plot(available_scales, gold_losses, f"-{marker}",
                    color=color, linewidth=1.5, markersize=7,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=strategy, zorder=3)

    # Resolved 系列（如有）
    for strategy, exps, color, marker in [
        ("Resolved", ["exp5", "exp6"], "#2ca02c", "^"),
    ]:
        gold_losses = []
        available_scales = []
        for exp, scale in zip(exps, [500, 1000]):
            row = df[(df["model"] == exp) & (df["split"] == "gold")]
            if not row.empty:
                gold_losses.append(float(row["mean_loss"].iloc[0]))
                available_scales.append(scale)
        if gold_losses:
            ax.plot(available_scales, gold_losses, f"--{marker}",
                    color=color, linewidth=1.2, markersize=6,
                    markeredgecolor="white", markeredgewidth=0.5,
                    label=strategy, alpha=0.8, zorder=2)

    # Baseline 水平线
    bl_row = df[(df["model"] == "baseline") & (df["split"] == "gold")]
    if not bl_row.empty:
        bl_loss = float(bl_row["mean_loss"].iloc[0])
        ax.axhline(y=bl_loss, color="#d62728", ls=":", lw=1.0, alpha=0.6, label="Baseline (no SFT)")

    ax.set_xlabel("Number of Training Trajectories")
    ax.set_ylabel("Cross-Entropy Loss (Gold Test Set)")
    ax.set_title("Scaling Behavior: Random vs TopQ", fontweight="bold")
    ax.set_xticks(scales)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlim(350, 2150)

    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_scaling_curve.{fmt}")
    plt.close(fig)
    logger.info("Scaling curve 已保存")

    # ── Figure: B2-Only vs Composite Ablation ──────────────────────────────
    ablation_models = ["exp3", "exp16"]   # Composite-Top500 vs B2Only-Top500
    ablation_data = {}
    for m in ablation_models:
        for split in ["gold", "random", "low_q"]:
            row = df[(df["model"] == m) & (df["split"] == split)]
            if not row.empty:
                ablation_data.setdefault(m, {})[split] = float(row["mean_loss"].iloc[0])

    if len(ablation_data) == 2:
        fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
        splits_display = ["Gold", "Random", "Low-Q"]
        x = np.arange(3)
        w = 0.35

        for i, (m, color, label) in enumerate([
            ("exp3",  "#ff7f0e", "Composite-Top500"),
            ("exp16", "#9467bd", "B2Only-Top500"),
        ]):
            vals = [ablation_data[m].get(s, 0) for s in ["gold", "random", "low_q"]]
            ax2.bar(x + (i - 0.5) * w, vals, w, color=color, label=label,
                    edgecolor="white", linewidth=0.5)

        ax2.set_xticks(x)
        ax2.set_xticklabels(splits_display)
        ax2.set_ylabel("Cross-Entropy Loss")
        ax2.set_title("Composite Score vs B2-Only Filtering", fontweight="bold")
        ax2.legend(frameon=False)

        plt.tight_layout()
        for fmt in ("png", "pdf"):
            fig2.savefig(fig_dir / f"fig_b2only_ablation.{fmt}")
        plt.close(fig2)
        logger.info("B2-Only ablation 已保存")
    else:
        logger.info("B2-Only ablation 数据不足，跳过")

    # ── Figure: 全模型 Loss 对比（含新实验）──────────────────────────────────
    gold_df = df[df["split"] == "gold"].copy()
    if len(gold_df) > 0:
        gold_df["display"] = gold_df["model"].map(DISPLAY_NAMES).fillna(gold_df["model"])
        gold_df = gold_df.sort_values("mean_loss", ascending=True)

        fig3, ax3 = plt.subplots(figsize=(7.0, 4.0))
        colors = []
        for m in gold_df["model"]:
            if m == "baseline":
                colors.append("#d62728")
            elif m in ("exp14", "exp15", "exp16"):
                colors.append("#9467bd")   # 新实验紫色
            elif "exp7" == m:
                colors.append("#7f7f7f")
            else:
                colors.append("#1f77b4")

        bars = ax3.barh(range(len(gold_df)), gold_df["mean_loss"], color=colors,
                        edgecolor="white", linewidth=0.5)
        ax3.set_yticks(range(len(gold_df)))
        ax3.set_yticklabels(gold_df["display"])
        ax3.set_xlabel("Cross-Entropy Loss (Gold Test Set)")
        ax3.set_title("All Models: CE Loss Comparison", fontweight="bold")
        ax3.invert_yaxis()

        # 标注数值
        for bar, val in zip(bars, gold_df["mean_loss"]):
            ax3.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                     f"{val:.4f}", va="center", fontsize=8)

        plt.tight_layout()
        for fmt in ("png", "pdf"):
            fig3.savefig(fig_dir / f"fig_all_models_loss.{fmt}")
        plt.close(fig3)
        logger.info("All models loss 已保存")


# ── Entry Point ────────────────────────────────────────────────────────────
def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser(
        description="Perplexity 评估（含 exp14-16 新实验）"
    )
    parser.add_argument("--models", nargs="+",
                        default=["exp14", "exp15", "exp16"],
                        help="要评估的模型（默认: exp14 exp15 exp16）")
    parser.add_argument("--splits", nargs="+", default=TEST_SPLITS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--outputs-dir", type=Path, default=None,
                        help="LoRA adapter 根目录")
    parser.add_argument("--force", action="store_true",
                        help="强制重新评估")
    parser.add_argument("--plot-only", action="store_true",
                        help="只生成图表（不跑评估）")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size
    outputs_root = args.outputs_dir or OUTPUTS_ROOT

    if not args.plot_only:
        invalid = [m for m in args.models if m not in EXPERIMENT_MAP]
        if invalid:
            logger.error("未知模型: %s\n可用: %s", invalid, list(EXPERIMENT_MAP.keys()))
            return
        run_evaluation(args.models, args.splits, outputs_root, force=args.force)

    generate_publication_figures(OUTPUT_DIR)


if __name__ == "__main__":
    main()
