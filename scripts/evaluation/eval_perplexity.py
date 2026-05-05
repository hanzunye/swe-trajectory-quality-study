"""
Step 2: 计算各模型在测试集上的Perplexity/Loss。

A100 80GB优化策略：
  - BF16推理（无需4-bit量化，14GB即可装载7B模型）
  - Flash Attention 2
  - 按trajectory逐条计算loss（确保per-sample粒度）
  - 批量模型切换（base → LoRA adapter hot-swap）

输出：
  /workspace/eval_results/perplexity_results.json
  /workspace/eval_results/perplexity_results.csv

Usage:
    python eval_perplexity.py
    python eval_perplexity.py --models exp1 exp3 exp5
    python eval_perplexity.py --batch-size 4 --splits gold random
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
OUTPUTS_ROOT = WORKSPACE / "outputs"   # LoRA adapter存放处

# ── A100 80GB推理配置 ──────────────────────────────────────────────────────
BASE_MODEL   = "Qwen/Qwen2.5-Coder-7B-Instruct"
TORCH_DTYPE  = torch.bfloat16           # A100原生支持BF16
DEVICE       = "cuda:0"
MAX_SEQ_LEN  = 32768
BATCH_SIZE   = 2                        # 批量推理，充分利用GPU

# 实验矩阵（exp_name → LoRA adapter路径）— v3 实验设计
EXPERIMENT_MAP = {
    "baseline":  None,                  # 基础模型，无LoRA
    "exp1":      "exp1/final",          # Random-500
    "exp2":      "exp2/final",          # Random-1000
    "exp3":      "exp3/final",          # TopQ-500
    "exp4":      "exp4/final",          # TopQ-1000
    "exp5":      "exp5/final",          # ResolvedOnly-500
    "exp6":      "exp6/final",          # ResolvedOnly-1000
    "exp7":      "exp7/final",          # BottomQ-500
    "exp8":      "exp8/final",          # Ablation-NoEfficiency-500
    "exp9":      "exp9/final",          # Ablation-NoStyle-500
    "exp10":     "exp10/final",         # Ablation-NoB2-500
    "exp11":     "exp11/final",         # Ablation-NoB3-500
    "exp12":     "exp12/final",         # Ablation-NoC2-500
    "exp13":     "exp13/final",         # Ablation-NoC3-500
}

TEST_SPLITS = ["gold", "random", "low_q"]

# ── 数据格式工具（复用prepare_data.py逻辑）────────────────────────────────
def trajectory_to_chatml(row: dict, tokenizer) -> str | None:
    """将raw轨迹行转换为ChatML格式文本。"""
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
    """
    Tokenize文本并构建assistant-only labels（非assistant位置为-100）。
    复用prepare_data.py中的tokenize_and_label逻辑。
    """
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
    """单条轨迹的 ChatML 转换 + tokenization（供并行调用）。"""
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
    """
    加载并tokenize测试集（JSONL格式，每行是一条raw trajectory）。
    - 优先从磁盘缓存（.cache.pt）加载，避免重复处理。
    - 首次处理时并行tokenize，利用多核 CPU 加速。
    返回: list of {trajectory_id, input_ids, attention_mask, labels}
    """
    jsonl_path = EVAL_DATA / f"{split_name}.jsonl"
    cache_path = EVAL_DATA / f"{split_name}.cache.pt"

    if not jsonl_path.exists():
        raise FileNotFoundError(f"测试集文件不存在: {jsonl_path}\n请先运行 build_test_set.py")

    # ── 命中缓存：直接加载 ────────────────────────────────────────────────
    if cache_path.exists() and cache_path.stat().st_mtime >= jsonl_path.stat().st_mtime:
        logger.info("[%s] 从缓存加载: %s", split_name, cache_path)
        try:
            samples = torch.load(cache_path, weights_only=True)
            if not isinstance(samples, list):
                raise TypeError(f"unexpected cache type: {type(samples)!r}")
            logger.info("[%s] 缓存命中，加载 %d 条有效轨迹", split_name, len(samples))
            return samples
        except Exception as exc:
            logger.warning("[%s] 缓存读取失败，重新生成: %s", split_name, exc)

    # ── 缓存未命中：并行处理 ──────────────────────────────────────────────
    logger.info("[%s] 未找到缓存，开始并行预处理...", split_name)
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

    # ── 保存缓存 ──────────────────────────────────────────────────────────
    torch.save(samples, cache_path)
    logger.info("[%s] 预处理完成，%d 条有效轨迹已缓存至: %s", split_name, len(samples), cache_path)
    return samples


# ── 模型加载 ──────────────────────────────────────────────────────────────
def load_base_model(tokenizer_only: bool = False):
    """加载BF16基础模型（A100上约14GB）。"""
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
    model.config.use_cache = True   # 推理时开启KV cache
    model.eval()

    mem_gb = torch.cuda.memory_allocated(0) / 1024**3
    logger.info("模型加载完成，显存占用: %.2f GB", mem_gb)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True

    return model, tokenizer


def load_lora_model(base_model, adapter_path: str):
    """在已加载基础模型上挂载LoRA adapter。"""
    full_path = OUTPUTS_ROOT / adapter_path
    if not full_path.exists():
        raise FileNotFoundError(f"LoRA adapter不存在: {full_path}")

    logger.info("挂载LoRA adapter: %s", full_path)
    model = PeftModel.from_pretrained(
        base_model,
        str(full_path),
        torch_dtype=TORCH_DTYPE,
        is_trainable=False,
    )
    model.eval()
    return model


# ── Dataset / collate ─────────────────────────────────────────────────────
class TrajectoryDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: list[dict]) -> dict:
    """将变长序列 padding 到 batch 内最长，labels padding 位置填 -100。"""
    max_len = max(s["input_ids"].size(0) for s in batch)
    input_ids_list, attn_mask_list, labels_list, tids = [], [], [], []

    for s in batch:
        pad = max_len - s["input_ids"].size(0)
        input_ids_list.append(torch.cat([s["input_ids"],
                                         torch.zeros(pad, dtype=torch.long)]))
        attn_mask_list.append(torch.cat([s["attention_mask"],
                                          torch.zeros(pad, dtype=torch.long)]))
        labels_list.append(torch.cat([s["labels"],
                                       torch.full((pad,), -100, dtype=torch.long)]))
        tids.append(s["trajectory_id"])

    return {
        "trajectory_ids":   tids,
        "input_ids":        torch.stack(input_ids_list),
        "attention_mask":   torch.stack(attn_mask_list),
        "labels":           torch.stack(labels_list),
    }


# ── 核心Loss计算（批量版）─────────────────────────────────────────────────
@torch.no_grad()
def compute_batch_loss(
    model,
    input_ids:      torch.Tensor,   # [B, seq_len]
    attention_mask: torch.Tensor,   # [B, seq_len]
    labels:         torch.Tensor,   # [B, seq_len]
) -> list[tuple[float, int]]:
    """
    批量计算每条轨迹的平均 cross-entropy loss（仅 assistant token）。
    Returns: list of (avg_loss, num_assistant_tokens) per sample
    """
    input_ids      = input_ids.to(DEVICE, non_blocking=True)
    attention_mask = attention_mask.to(DEVICE, non_blocking=True)
    labels         = labels.to(DEVICE, non_blocking=True)

    with torch.amp.autocast(device_type="cuda", dtype=TORCH_DTYPE):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    # 保持 BF16，逐样本切片计算 loss，避免一次性分配巨型 float32 张量
    shift_logits = outputs.logits[:, :-1, :]   # BF16 [B, seq_len-1, vocab]
    shift_labels = labels[:, 1:].contiguous()  # [B, seq_len-1]

    loss_fct = nn.CrossEntropyLoss(reduction="none")
    results = []

    for i in range(shift_logits.size(0)):
        # 仅对单条样本升精度，显存占用降低 B 倍
        token_loss = loss_fct(shift_logits[i].float(), shift_labels[i])  # [seq_len-1]
        mask       = (shift_labels[i] != -100).float()
        num_tokens = int(mask.sum().item())
        if num_tokens == 0:
            results.append((float("nan"), 0))
        else:
            results.append(((token_loss * mask).sum().item() / num_tokens, num_tokens))

    del shift_logits, outputs   # 立即释放显存
    return results


@torch.no_grad()
def evaluate_model_on_split(model, samples: list[dict], split_name: str, exp_name: str) -> dict:
    """
    在单个测试集上批量评估模型，DataLoader 多进程预取 + pin_memory 加速传输。
    """
    losses   = []
    n_tokens = []
    failed   = 0
    processed = 0

    loader = DataLoader(
        TrajectoryDataset(samples),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,          # CPU 预取线程
        pin_memory=True,        # 锁页内存，加速 CPU→GPU 传输
        collate_fn=collate_fn,
    )

    t0 = time.time()
    for batch in loader:
        batch_results = compute_batch_loss(
            model,
            batch["input_ids"],
            batch["attention_mask"],
            batch["labels"],
        )
        for loss, n in batch_results:
            if not np.isnan(loss):
                losses.append(loss)
                n_tokens.append(n)
            else:
                failed += 1

        processed += len(batch_results)
        if processed % 50 < BATCH_SIZE:
            elapsed = time.time() - t0
            logger.info(
                "  [%s/%s] %d/%d  avg_loss=%.4f  elapsed=%.0fs",
                exp_name, split_name, processed, len(samples),
                float(np.mean(losses)) if losses else float("nan"),
                elapsed,
            )

    if not losses:
        return {"mean_loss": float("nan"), "std_loss": float("nan"),
                "perplexity": float("nan"), "n_valid": 0, "n_failed": failed}

    mean_loss = float(np.mean(losses))
    std_loss  = float(np.std(losses))
    ppl       = float(np.exp(mean_loss))

    logger.info(
        "[%s/%s] 完成: mean_loss=%.4f  std=%.4f  PPL=%.2f  valid=%d  failed=%d",
        exp_name, split_name, mean_loss, std_loss, ppl, len(losses), failed,
    )

    return {
        "mean_loss":  mean_loss,
        "std_loss":   std_loss,
        "perplexity": ppl,
        "losses":     losses,       # per-trajectory loss列表，用于统计检验
        "n_valid":    len(losses),
        "n_failed":   failed,
        "mean_assistant_tokens": float(np.mean(n_tokens)),
    }


# ── 主评估循环 ────────────────────────────────────────────────────────────
def run_evaluation(target_models: list[str], target_splits: list[str], force: bool = False):
    """
    依次评估所有模型在所有测试集上的loss。

    内存策略：
    1. 加载base model (BF16, ~14GB)
    2. 对baseline直接评估
    3. 对每个LoRA: load adapter → 评估 → unload adapter → 重复
    避免重复加载base model（80GB可以常驻）
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载tokenizer（用于数据预处理）
    logger.info("加载Tokenizer...")
    _, tokenizer = load_base_model(tokenizer_only=True)

    # 预处理并缓存所有测试集
    logger.info("预处理测试集...")
    split_data = {}
    for split_name in target_splits:
        if split_name not in split_data:
            split_data[split_name] = load_test_split(split_name, tokenizer)

    # 加载基础模型（常驻显存）
    logger.info("加载基础模型（常驻显存）...")
    base_model, _ = load_base_model()

    # 加载已有结果，避免覆盖之前跑完的模型
    json_path = OUTPUT_DIR / "perplexity_results.json"
    if json_path.exists():
        with open(json_path) as f:
            all_results = json.load(f)
        logger.info("已加载历史结果，包含模型: %s", list(all_results.keys()))
    else:
        all_results = {}

    for exp_name in target_models:
        logger.info("=" * 60)
        # 断点续跑：已有结果且未强制重跑时跳过
        if not force and exp_name in all_results:
            logger.info("跳过 %s（历史结果已存在）。如需重跑请加 --force", exp_name)
            continue
        logger.info("评估模型: %s", exp_name)
        adapter_rel_path = EXPERIMENT_MAP.get(exp_name)

        # 加载模型（baseline用base，其他加载LoRA）
        if adapter_rel_path is None:
            model = base_model
            logger.info("使用基础模型（无LoRA）")
        else:
            model = load_lora_model(base_model, adapter_rel_path)

        # 在各测试集上评估
        exp_results = {"exp_name": exp_name}
        for split_name in target_splits:
            samples = split_data[split_name]
            result  = evaluate_model_on_split(model, samples, split_name, exp_name)
            exp_results[split_name] = result

        all_results[exp_name] = exp_results

        # 卸载LoRA（保留base model）
        if adapter_rel_path is not None:
            # get_base_model() 返回原始 AutoModelForCausalLM
            base_model = model.get_base_model()
            del model
            # 清除 PEFT 遗留属性，防止下次加载时出现 multiple adapters 警告
            for _attr in ("peft_config", "active_adapter", "active_adapters"):
                try:
                    delattr(base_model, _attr)
                except AttributeError:
                    pass
            gc.collect()
            torch.cuda.empty_cache()
            mem_gb = torch.cuda.memory_allocated(0) / 1024**3
            logger.info("LoRA已卸载，当前显存: %.2f GB", mem_gb)

        # 中间保存（防止意外中断）
        _save_results(all_results, OUTPUT_DIR)

    logger.info("=" * 60)
    logger.info("所有模型评估完成!")
    _save_results(all_results, OUTPUT_DIR)
    return all_results


def _save_results(all_results: dict, output_dir: Path):
    """保存JSON + CSV格式结果。原子写入：先写临时文件再替换，防止写入中断导致文件损坏。"""
    # JSON（含per-trajectory losses列表）
    json_path = output_dir / "perplexity_results.json"
    tmp_path  = json_path.with_suffix(".tmp.json")
    with open(tmp_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    tmp_path.replace(json_path)   # 原子替换，写完再覆盖

    # CSV（汇总统计，不含losses列表）
    rows = []
    for exp_name, exp_data in all_results.items():
        for split_name in TEST_SPLITS:
            if split_name not in exp_data:
                continue
            r = exp_data[split_name]
            rows.append({
                "model":      exp_name,
                "split":      split_name,
                "mean_loss":  r.get("mean_loss"),
                "std_loss":   r.get("std_loss"),
                "perplexity": r.get("perplexity"),
                "n_valid":    r.get("n_valid"),
            })

    if rows:
        df = pd.DataFrame(rows)
        csv_path = output_dir / "perplexity_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("结果已保存: %s", csv_path)


# ── Entry Point ────────────────────────────────────────────────────────────
def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser(description="计算各模型在测试集上的Perplexity")
    parser.add_argument("--models",  nargs="+",
                        default=list(EXPERIMENT_MAP.keys()),
                        help="要评估的模型名称 (e.g. baseline exp1 exp3)")
    parser.add_argument("--splits",  nargs="+",
                        default=TEST_SPLITS,
                        help="要评估的测试集 (gold random low_q)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="批处理大小（默认2）")
    parser.add_argument("--force", action="store_true",
                        help="强制重新评估（忽略已有结果，会覆盖）")
    args = parser.parse_args()

    BATCH_SIZE = args.batch_size

    # 验证模型名称
    invalid = [m for m in args.models if m not in EXPERIMENT_MAP]
    if invalid:
        logger.error("未知模型名称: %s\n可用: %s", invalid, list(EXPERIMENT_MAP.keys()))
        return

    run_evaluation(args.models, args.splits, force=args.force)


if __name__ == "__main__":
    main()
