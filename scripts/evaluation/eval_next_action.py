"""
Step 3: Next-Action Prediction Accuracy（指标②）。

原理：给定轨迹前N步，模型能否预测第N+1步的action type。

评估：
  - Action Type Accuracy: bash/edit/search等类型匹配
  - Format Compliance  : tool_call格式是否合法JSON
  - BLEU/ROUGE分数     : 生成内容与真实内容的相似度

A100优化：vLLM批量推理（如已启动vLLM服务）或直接HF推理。

Usage:
    python eval_next_action.py --models baseline exp1 exp3
    python eval_next_action.py --use-vllm --vllm-base-url http://localhost:8000
    python eval_next_action.py --truncate-steps 3 5 7
"""

import argparse
import gc
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────
WORKSPACE    = Path(os.environ.get("WORKSPACE", "/workspace"))
EVAL_DATA    = WORKSPACE / "eval_data"
OUTPUT_DIR   = WORKSPACE / "eval_results"
OUTPUTS_ROOT = WORKSPACE / "outputs"

BASE_MODEL  = "Qwen/Qwen2.5-Coder-7B-Instruct"
TORCH_DTYPE = torch.bfloat16
DEVICE      = "cuda:0"

# 评估时截断到前N步，预测第N+1步
TRUNCATE_AT_STEPS = [3, 5, 7]
MAX_NEW_TOKENS    = 256    # 生成最大token数
NUM_SAMPLES          = 100    # 每个模型评估100条轨迹（节省时间）
GENERATE_BATCH_SIZE  = 4     # 批量生成，充分利用GPU

# 实验矩阵（exp_name → LoRA adapter路径）— v3 实验设计
EXPERIMENT_MAP = {
    "baseline": None,                   # 基础模型，无LoRA
    "exp1":  "exp1/final",              # Random-500
    "exp2":  "exp2/final",              # Random-1000
    "exp3":  "exp3/final",              # TopQ-500
    "exp4":  "exp4/final",              # TopQ-1000
    "exp5":  "exp5/final",              # ResolvedOnly-500
    "exp6":  "exp6/final",              # ResolvedOnly-1000
    "exp7":  "exp7/final",              # BottomQ-500
    "exp8":  "exp8/final",              # Ablation-NoEfficiency-500
    "exp9":  "exp9/final",              # Ablation-NoStyle-500
    "exp10": "exp10/final",             # Ablation-NoB2-500
    "exp11": "exp11/final",             # Ablation-NoB3-500
    "exp12": "exp12/final",             # Ablation-NoC2-500
    "exp13": "exp13/final",             # Ablation-NoC3-500
}

# ── Action类型解析 ────────────────────────────────────────────────────────
ACTION_PATTERNS = {
    "bash":       re.compile(r'"name"\s*:\s*"(execute_bash|bash|run_command)"'),
    "edit":       re.compile(r'"name"\s*:\s*"(str_replace_editor|write_file|edit_file|create_file|str_replace)"'),
    "search":     re.compile(r'"name"\s*:\s*"(search|grep|find|read_file|view)"'),
    "think":      re.compile(r'<think>|<reasoning>'),
    "finish":     re.compile(r'"name"\s*:\s*"(finish|submit|exit)"'),
    "tool_call":  re.compile(r'<tool_call>'),
}


def parse_action_type(text: str) -> str:
    """从生成文本中解析action类型。"""
    text = text.strip()
    for action_type, pattern in ACTION_PATTERNS.items():
        if action_type == "tool_call":
            continue
        if pattern.search(text):
            return action_type
    # 通用tool_call
    if ACTION_PATTERNS["tool_call"].search(text):
        return "tool_call_other"
    return "text"


def check_format_compliance(text: str) -> dict:
    """
    检查生成文本的格式合规性：
    1. 是否包含<tool_call>标签
    2. tool_call内容是否为合法JSON
    3. 是否包含必要的name和arguments字段
    """
    result = {
        "has_tool_call": False,
        "valid_json":    False,
        "has_name":      False,
        "has_arguments": False,
        "format_score":  0.0,   # 0-1
    }

    tc_match = re.search(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL)
    if not tc_match:
        # 检查是否有半格式化的tool call
        if "<tool_call>" in text:
            result["has_tool_call"] = True
            result["format_score"]  = 0.25
        return result

    result["has_tool_call"] = True
    tc_content = tc_match.group(1).strip()

    try:
        parsed = json.loads(tc_content)
        result["valid_json"] = True
        if "name" in parsed:
            result["has_name"] = True
        if "arguments" in parsed:
            result["has_arguments"] = True
        score = 0.5 + 0.25 * result["has_name"] + 0.25 * result["has_arguments"]
        result["format_score"] = score
    except json.JSONDecodeError:
        result["format_score"] = 0.3

    return result


def compute_bleu_rouge(hypothesis: str, reference: str) -> dict:
    """计算BLEU-4和ROUGE-L分数。"""
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from rouge_score import rouge_scorer

        hyp_tokens = hypothesis.split()
        ref_tokens = reference.split()

        # BLEU-4
        smoothie = SmoothingFunction().method1
        bleu4 = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

        # ROUGE-L
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        rouge_l = scorer.score(reference, hypothesis)["rougeL"].fmeasure

        return {"bleu4": float(bleu4), "rouge_l": float(rouge_l)}
    except ImportError:
        return {"bleu4": float("nan"), "rouge_l": float("nan")}


# ── 轨迹截断工具 ──────────────────────────────────────────────────────────
def parse_trajectory_messages(row: dict) -> list[dict] | None:
    """解析原始行的trajectory字段为messages列表。"""
    trajectory = row.get("trajectory", [])
    if isinstance(trajectory, str):
        try:
            trajectory = json.loads(trajectory)
        except Exception:
            return None

    if not trajectory:
        return None

    messages = []
    for msg in trajectory:
        role    = msg.get("role", "")
        content = msg.get("content", "") or ""
        tool_calls = msg.get("tool_calls") or []

        if role == "system":
            messages.append({"role": "system", "content": content})
        elif role == "user":
            messages.append({"role": "user", "content": content})
        elif role == "assistant":
            if tool_calls:
                tc_parts = []
                for tc in tool_calls:
                    func = tc.get("function", {})
                    args = func.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            pass
                    args_str = json.dumps(args, ensure_ascii=False) if isinstance(args, dict) else str(args)
                    tc_parts.append(
                        f'<tool_call>\n{{"name": "{func.get("name","")}", "arguments": {args_str}}}\n</tool_call>'
                    )
                full_content = (content + "\n" + "\n".join(tc_parts)) if content else "\n".join(tc_parts)
                messages.append({"role": "assistant", "content": full_content})
            elif content:
                messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})

    return messages if len(messages) >= 3 else None


def truncate_to_n_steps(messages: list[dict], n_steps: int) -> tuple[list[dict], dict | None]:
    """
    截断messages到前n_steps个"轮次"（一个用户消息+一个assistant响应 = 1步）。
    返回: (截断后的messages, 第n_steps+1步的assistant消息)
    """
    # 收集 (user, assistant) 对，跳过system消息
    system_msgs = [m for m in messages if m["role"] == "system"]
    turns = []   # (user_msg, assistant_msg) pairs

    i = 0
    non_system = [m for m in messages if m["role"] != "system"]
    while i < len(non_system) - 1:
        if non_system[i]["role"] == "user" and non_system[i+1]["role"] == "assistant":
            turns.append((non_system[i], non_system[i+1]))
            i += 2
        else:
            i += 1

    if len(turns) < n_steps + 1:
        return None, None   # 轨迹太短

    # 构建截断messages（保留前n_steps轮）
    truncated = system_msgs.copy()
    for user_msg, asst_msg in turns[:n_steps]:
        truncated.append(user_msg)
        truncated.append(asst_msg)
    # 加入第n_steps+1步的用户消息（作为prompt）
    truncated.append(turns[n_steps][0])   # next user turn

    ground_truth_asst = turns[n_steps][1]  # 真实的第n_steps+1步assistant回复

    return truncated, ground_truth_asst


# ── HF推理 ────────────────────────────────────────────────────────────────
def batch_generate(model, tokenizer, messages_list: list[list[dict]]) -> list[str]:
    """
    批量生成：对多条 messages 并行 generate，GPU 利用率接近线性提升。
    使用 left-padding（causal LM 生成的标准做法）。
    """
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=32768 - MAX_NEW_TOKENS,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokenizer.padding_side = orig_padding_side

    # 只取各序列新生成的 token（去掉 padded input 部分）
    padded_input_len = inputs["input_ids"].shape[1]
    return [
        tokenizer.decode(outputs[i][padded_input_len:], skip_special_tokens=True)
        for i in range(len(messages_list))
    ]


# ── 评估单个模型 ──────────────────────────────────────────────────────────
def evaluate_next_action(
    model, tokenizer,
    test_samples: list[dict],
    exp_name: str,
    truncate_steps: list[int],
) -> dict:
    """
    评估模型的next-action预测能力。
    策略：先收集所有 (sample, n_steps) 对，再批量 generate，最后后处理。
    """
    # ── Step 1: 预处理，收集所有待生成的请求 ─────────────────────────────
    pending = []   # (tid, n_steps, truncated_messages, ground_truth_content)
    for i, row in enumerate(test_samples[:NUM_SAMPLES]):
        tid = row.get("trajectory_id", f"row_{i}")
        messages = parse_trajectory_messages(row)
        if messages is None:
            continue
        for n_steps in truncate_steps:
            truncated, ground_truth = truncate_to_n_steps(messages, n_steps)
            if truncated is None:
                continue
            pending.append((tid, n_steps, truncated, ground_truth["content"]))

    logger.info("  [%s] 共 %d 条生成请求，batch=%d", exp_name, len(pending), GENERATE_BATCH_SIZE)

    # ── Step 2: 批量 generate ─────────────────────────────────────────────
    generated_texts = []
    for start in range(0, len(pending), GENERATE_BATCH_SIZE):
        batch = pending[start:start + GENERATE_BATCH_SIZE]
        generated_texts.extend(batch_generate(model, tokenizer, [p[2] for p in batch]))
        done = min(start + GENERATE_BATCH_SIZE, len(pending))
        if done % 50 < GENERATE_BATCH_SIZE or done == len(pending):
            logger.info("  [%s] 已生成 %d/%d", exp_name, done, len(pending))

    # ── Step 3: 后处理，统计指标 ─────────────────────────────────────────
    results_by_step = {n: [] for n in truncate_steps}
    for (tid, n_steps, _, gt_content), generated in zip(pending, generated_texts):
        true_action  = parse_action_type(gt_content)
        pred_action  = parse_action_type(generated)
        fmt          = check_format_compliance(generated)
        metrics      = compute_bleu_rouge(generated[:512], gt_content[:512])

        results_by_step[n_steps].append({
            "trajectory_id": tid,
            "action_match":  (true_action == pred_action),
            "true_action":   true_action,
            "pred_action":   pred_action,
            "format_score":  fmt["format_score"],
            "bleu4":         metrics["bleu4"],
            "rouge_l":       metrics["rouge_l"],
        })

    # 汇总统计
    summary = {}
    for n_steps, records in results_by_step.items():
        if not records:
            continue
        summary[f"step_{n_steps}"] = {
            "action_accuracy": float(np.mean([r["action_match"] for r in records])),
            "format_score":    float(np.mean([r["format_score"] for r in records])),
            "bleu4":           float(np.nanmean([r["bleu4"] for r in records])),
            "rouge_l":         float(np.nanmean([r["rouge_l"] for r in records])),
            "n_samples":       len(records),
            "action_dist_pred": _count_action_dist([r["pred_action"] for r in records]),
            "action_dist_true": _count_action_dist([r["true_action"] for r in records]),
        }
        logger.info(
            "[%s] step=%d: action_acc=%.3f  format=%.3f  bleu4=%.3f  rouge_l=%.3f",
            exp_name, n_steps,
            summary[f"step_{n_steps}"]["action_accuracy"],
            summary[f"step_{n_steps}"]["format_score"],
            summary[f"step_{n_steps}"]["bleu4"],
            summary[f"step_{n_steps}"]["rouge_l"],
        )

    return summary


def _count_action_dist(actions: list[str]) -> dict:
    dist = {}
    for a in actions:
        dist[a] = dist.get(a, 0) + 1
    total = len(actions)
    return {k: round(v / total, 3) for k, v in dist.items()}


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    global NUM_SAMPLES
    parser = argparse.ArgumentParser(description="Next-Action Prediction Accuracy")
    parser.add_argument("--models", nargs="+",
                        default=["baseline", "exp1", "exp3", "exp5", "exp7"],
                        help="要评估的模型 (subset of experiment names)")
    parser.add_argument("--splits", nargs="+", default=["gold"],
                        help="测试集 (gold random low_q)")
    parser.add_argument("--truncate-steps", nargs="+", type=int, default=TRUNCATE_AT_STEPS,
                        help="截断步数列表 (e.g. 3 5 7)")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES,
                        help="每个模型评估的样本数")
    args = parser.parse_args()

    NUM_SAMPLES = args.num_samples

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 加载tokenizer和base model
    logger.info("加载Tokenizer和基础模型...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=TORCH_DTYPE,
        device_map={"": DEVICE},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    base_model.config.use_cache = True
    base_model.eval()

    # 加载测试数据（raw JSONL）
    all_test_rows = []
    for split_name in args.splits:
        jsonl_path = EVAL_DATA / f"{split_name}.jsonl"
        if not jsonl_path.exists():
            logger.warning("测试集不存在: %s，跳过", jsonl_path)
            continue
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        logger.info("[%s] 加载 %d 条原始轨迹", split_name, len(rows))
        all_test_rows.extend(rows)

    if not all_test_rows:
        logger.error("无可用测试数据，请先运行 build_test_set.py")
        return

    out_path = OUTPUT_DIR / "next_action_results.json"
    if out_path.exists():
        with open(out_path) as f:
            all_results = json.load(f)
        logger.info("已加载历史结果，包含模型: %s", list(all_results.keys()))
    else:
        all_results = {}

    for exp_name in args.models:
        logger.info("=" * 60)
        logger.info("评估模型: %s", exp_name)

        if exp_name not in EXPERIMENT_MAP:
            logger.error("未知模型名称: %s，跳过。可用: %s", exp_name, list(EXPERIMENT_MAP.keys()))
            continue

        adapter_rel = EXPERIMENT_MAP.get(exp_name)
        if adapter_rel is not None:
            model = PeftModel.from_pretrained(
                base_model, str(OUTPUTS_ROOT / adapter_rel),
                torch_dtype=TORCH_DTYPE, is_trainable=False,
            )
            model.eval()
        else:
            model = base_model

        result = evaluate_next_action(
            model, tokenizer,
            all_test_rows, exp_name,
            args.truncate_steps,
        )
        all_results[exp_name] = result

        # 中间保存（防止意外中断丢失结果）
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        logger.info("中间结果已保存: %s", out_path)

        if adapter_rel is not None:
            base_model = model.get_base_model()
            del model
            for _attr in ("peft_config", "active_adapter", "active_adapters"):
                try:
                    delattr(base_model, _attr)
                except AttributeError:
                    pass
            gc.collect()
            torch.cuda.empty_cache()

    # 保存结果
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info("Next-Action结果已保存: %s", out_path)


if __name__ == "__main__":
    main()
