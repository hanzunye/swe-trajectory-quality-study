"""
eval_first_action_new.py — First-Action Evaluation（含 exp14/15/16 新实验）
A100 服务器版本

在原 eval_first_action.py 基础上扩展：
  exp14 — Random-2000      (Experiment B: scale extension)
  exp15 — TopQ-2000        (Experiment B: scale extension)
  exp16 — B2Only-Top500    (Experiment C: B2-only vs composite ablation)

新增特性：
  - 完整实验矩阵（baseline + exp1-16）
  - --outputs-dir 参数支持自定义 adapter 路径
  - --plot-only 模式，仅从已有 JSON 结果重新生成论文级插图
  - 增量合并已有 first_action_results.json
  - 3 组论文级插图：
    1. fig_proxy_validation.pdf — CE Loss vs First-Action Quality (scatter)
    2. fig_first_action_bars.pdf — Grouped bar chart (all checkpoints)
    3. fig_fa_scaling_curve.pdf — Random vs TopQ scaling (500→1000→2000)
    4. fig_fa_b2only_ablation.pdf — Composite-Top500 vs B2Only-Top500

Usage:
    # 只评估新实验
    python eval_first_action_new.py --models exp14 exp15 exp16

    # 全量评估
    python eval_first_action_new.py --models baseline exp1 exp2 exp3 exp4 exp14 exp15 exp16

    # 自定义 adapter 路径
    python eval_first_action_new.py --models exp14 exp15 exp16 --outputs-dir /workspace/outputs

    # 只重新生成图表（不跑推理）
    python eval_first_action_new.py --plot-only
"""

import argparse
import gc
import json
import logging
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).parent
EVAL_DATA    = SCRIPT_DIR / "eval_data"
OUTPUT_DIR   = SCRIPT_DIR / "eval_results"
OUTPUTS_ROOT = SCRIPT_DIR / "outputs"   # LoRA adapter 默认目录

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

# 完整实验矩阵（exp1-13 原有 + exp14-16 新增）
EXPERIMENT_MAP = {
    "baseline": None,                  # 基础模型，无 LoRA
    # ── Block 1: Core ──
    "exp1":     "exp1/final",          # Random-500
    "exp2":     "exp2/final",          # Random-1000
    "exp3":     "exp3/final",          # TopQ-500
    "exp4":     "exp4/final",          # TopQ-1000
    "exp5":     "exp5/final",          # ResolvedOnly-500
    "exp6":     "exp6/final",          # ResolvedOnly-1000
    "exp7":     "exp7/final",          # BottomQ-500
    # ── Block 2: Ablation ──
    "exp8":     "exp8/final",          # Ablation-NoEfficiency-500
    "exp9":     "exp9/final",          # Ablation-NoStyle-500
    # ── Block 3: Sub-dimension ──
    "exp10":    "exp10/final",         # Ablation-NoB2-500
    "exp11":    "exp11/final",         # Ablation-NoB3-500
    "exp12":    "exp12/final",         # Ablation-NoC2-500
    "exp13":    "exp13/final",         # Ablation-NoC3-500
    # ── NEW: Experiment B (Scale) + C (B2-only) ──
    "exp14":    "exp14/final",         # Random-2000
    "exp15":    "exp15/final",         # TopQ-2000
    "exp16":    "exp16/final",         # B2Only-Top500
}

# 显示名称映射（用于图表）
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "exp1": "Random-500",   "exp2": "Random-1000",  "exp14": "Random-2000",
    "exp3": "TopQ-500",     "exp4": "TopQ-1000",    "exp15": "TopQ-2000",
    "exp5": "Resolved-500", "exp6": "Resolved-1000",
    "exp7": "BottomQ-500",
    "exp8": "NoEfficiency", "exp9": "NoStyle",
    "exp10": "NoB2", "exp11": "NoB3", "exp12": "NoC2", "exp13": "NoC3",
    "exp16": "B2Only-500",
}

# 每个 checkpoint 的 marker + 颜色（用于 scatter plot）
STYLE_MAP = {
    "baseline":  ("Baseline",       "D", "#d62728"),
    "exp1":      ("Random-500",     "s", "#1f77b4"),
    "exp2":      ("Random-1000",    "^", "#2ca02c"),
    "exp3":      ("TopQ-500",       "o", "#ff7f0e"),
    "exp4":      ("TopQ-1000",      "P", "#9467bd"),
    "exp5":      ("Resolved-500",   "v", "#8c564b"),
    "exp6":      ("Resolved-1000",  "<", "#e377c2"),
    "exp7":      ("BottomQ-500",    "X", "#7f7f7f"),
    "exp8":      ("NoEfficiency",   "h", "#bcbd22"),
    "exp9":      ("NoStyle",        "p", "#17becf"),
    "exp10":     ("NoB2",           "d", "#aec7e8"),
    "exp11":     ("NoB3",           "*", "#ffbb78"),
    "exp12":     ("NoC2",           "H", "#98df8a"),
    "exp13":     ("NoC3",           "+", "#ff9896"),
    "exp14":     ("Random-2000",    "s", "#393b79"),
    "exp15":     ("TopQ-2000",      "o", "#e6550d"),
    "exp16":     ("B2Only-500",     "D", "#756bb1"),
}

MAX_NEW_TOKENS      = 300    # 第一步 action 一般不超过 300 token
MAX_SEQ_LEN         = 32768  # A100 40/80GB + Flash Attention 2
GENERATE_BATCH_SIZE = 4      # A100 显存充裕，可调至 8


# ── Action 类型解析 ──────────────────────────────────────────────────────
ACTION_PATTERNS = {
    "bash":   re.compile(r'"name"\s*:\s*"(execute_bash|bash|run_command)"'),
    "edit":   re.compile(r'"name"\s*:\s*"(str_replace_editor|write_file|edit_file|create_file|str_replace)"'),
    "search": re.compile(r'"name"\s*:\s*"(search|grep|find|read_file|view|open_file)"'),
    "think":  re.compile(r'<think>|<reasoning>'),
    "finish": re.compile(r'"name"\s*:\s*"(finish|submit|exit)"'),
    "tool_call": re.compile(r'<tool_call>'),
}

FILE_PATH_RE = re.compile(
    r'(?:'
    r'"(?:path|file_path|filename|target_file)"\s*:\s*"([^"]+)"'
    r'|'
    r'(?:^|[\s"])(/[\w./\-]+\.\w+)'
    r'|'
    r'(?:^|[\s"])([\w./\-]+/[\w./\-]+\.\w+)'
    r')',
    re.MULTILINE,
)


# ─────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────

def detect_device(force_device: str | None = None) -> torch.device:
    if force_device:
        dev = torch.device(force_device)
        logger.info("设备已强制指定：%s", dev)
        return dev
    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        logger.info("检测到 CUDA GPU：%s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        dev = torch.device("mps")
        logger.info("检测到 Apple MPS")
    else:
        dev = torch.device("cpu")
        logger.info("使用 CPU 推理")
    return dev


def get_torch_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
    gc.collect()


def parse_action_type(text: str) -> str:
    text = text.strip()
    for action_type, pattern in ACTION_PATTERNS.items():
        if action_type == "tool_call":
            continue
        if pattern.search(text):
            return action_type
    if ACTION_PATTERNS["tool_call"].search(text):
        return "tool_call_other"
    return "text"


def extract_file_paths(text: str) -> set[str]:
    paths = set()
    for m in FILE_PATH_RE.finditer(text):
        for group in m.groups():
            if group:
                p = group.strip().strip('"')
                if len(p) >= 3 and "/" in p:
                    paths.add(p)
    return paths


def file_path_match(generated: str, ground_truth: str) -> float:
    gt_paths   = extract_file_paths(ground_truth)
    pred_paths = extract_file_paths(generated)
    if not gt_paths and not pred_paths:
        return 1.0
    if not gt_paths or not pred_paths:
        return 0.0
    intersection = gt_paths & pred_paths
    union        = gt_paths | pred_paths
    return len(intersection) / len(union)


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        score  = scorer.score(reference[:1000], hypothesis[:1000])
        return float(score["rougeL"].fmeasure)
    except ImportError:
        h_tokens = hypothesis.split()[:200]
        r_tokens = reference.split()[:200]
        m, n = len(h_tokens), len(r_tokens)
        if m == 0 or n == 0:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if h_tokens[i-1] == r_tokens[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        prec = lcs / m if m else 0.0
        rec  = lcs / n if n else 0.0
        return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────
# 轨迹解析
# ─────────────────────────────────────────────────────────────────────────

def parse_first_action_pair(row: dict) -> tuple[list[dict], str] | tuple[None, None]:
    """
    从一条 JSONL 行中提取：
      - prompt_messages: [system, user]
      - gt_first_action: ground truth 第一步 assistant 内容
    """
    trajectory = row.get("trajectory", [])
    if isinstance(trajectory, str):
        try:
            trajectory = json.loads(trajectory)
        except Exception:
            return None, None
    if not trajectory:
        return None, None

    messages: list[dict] = []
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
                    args_str = (
                        json.dumps(args, ensure_ascii=False)
                        if isinstance(args, dict) else str(args)
                    )
                    tc_parts.append(
                        f'<tool_call>\n{{"name": "{func.get("name","")}", '
                        f'"arguments": {args_str}}}\n</tool_call>'
                    )
                full_content = (
                    (content + "\n" + "\n".join(tc_parts)) if content
                    else "\n".join(tc_parts)
                )
                messages.append({"role": "assistant", "content": full_content})
            elif content:
                messages.append({"role": "assistant", "content": content})
        elif role == "tool":
            messages.append({"role": "user", "content": f"<tool_response>\n{content}\n</tool_response>"})

    system_msgs = [m for m in messages if m["role"] == "system"]
    non_system  = [m for m in messages if m["role"] != "system"]

    if not non_system or non_system[0]["role"] != "user":
        return None, None

    prompt_messages = system_msgs + [non_system[0]]

    gt_msg = next((m for m in non_system if m["role"] == "assistant"), None)
    if gt_msg is None:
        return None, None

    return prompt_messages, gt_msg["content"]


# ─────────────────────────────────────────────────────────────────────────
# 模型加载与推理
# ─────────────────────────────────────────────────────────────────────────

def load_base_model(base_model_name: str, device: torch.device, dtype: torch.dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("加载 tokenizer: %s", base_model_name)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("加载 base model（dtype=%s）...", dtype)
    load_kwargs = dict(
        torch_dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if device.type == "cuda":
        load_kwargs["device_map"] = "auto"
        load_kwargs["attn_implementation"] = "flash_attention_2"
        logger.info("启用 Flash Attention 2")
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(base_model_name, **load_kwargs)
        model = model.to(device)

    model.config.use_cache = True
    model.eval()
    logger.info("base model 加载完成，设备：%s", device)
    return tokenizer, model


def load_lora_model(base_model, adapter_path: str, device: torch.device, dtype: torch.dtype):
    from peft import PeftModel
    logger.info("加载 LoRA adapter: %s", adapter_path)
    model = PeftModel.from_pretrained(
        base_model, adapter_path,
        torch_dtype=dtype, is_trainable=False,
    )
    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    return model


def unload_lora(model, device: torch.device):
    base = model.get_base_model()
    for attr in ("peft_config", "active_adapter", "active_adapters"):
        try:
            delattr(base, attr)
        except AttributeError:
            pass
    del model
    empty_cache(device)
    return base


def batch_generate(
    model, tokenizer,
    messages_list: list[list[dict]],
    device: torch.device,
) -> list[str]:
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    texts = [
        tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        for msgs in messages_list
    ]
    inputs = tokenizer(
        texts, return_tensors="pt",
        truncation=True, max_length=MAX_SEQ_LEN - MAX_NEW_TOKENS,
        padding=True,
    )
    input_ids      = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    tokenizer.padding_side = orig_padding_side
    padded_len = input_ids.shape[1]
    return [
        tokenizer.decode(output_ids[i][padded_len:], skip_special_tokens=True)
        for i in range(len(messages_list))
    ]


# ─────────────────────────────────────────────────────────────────────────
# 评估单个 checkpoint
# ─────────────────────────────────────────────────────────────────────────

def evaluate_checkpoint(
    model, tokenizer,
    samples: list[dict],
    exp_name: str,
    device: torch.device,
    num_samples: int,
    batch_size: int = GENERATE_BATCH_SIZE,
) -> dict:
    pending = []
    n_skipped = 0
    for i, row in enumerate(samples[:num_samples]):
        tid = row.get("trajectory_id", f"row_{i}")
        prompt_msgs, gt_action = parse_first_action_pair(row)
        if prompt_msgs is None:
            n_skipped += 1
        else:
            pending.append((tid, prompt_msgs, gt_action))

    logger.info("[%s] 有效样本 %d 条，batch_size=%d", exp_name, len(pending), batch_size)

    records = []
    t0 = time.time()

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        msgs_batch = [p[1] for p in batch]

        try:
            generated_batch = batch_generate(model, tokenizer, msgs_batch, device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                logger.warning("[%s] batch %d OOM，降级逐条推理...", exp_name, start)
                empty_cache(device)
                generated_batch = []
                for single_msgs in msgs_batch:
                    try:
                        out = batch_generate(model, tokenizer, [single_msgs], device)
                        generated_batch.extend(out)
                    except Exception:
                        logger.warning("[%s] 单条推理仍 OOM，跳过", exp_name)
                        generated_batch.append("")
                        n_skipped += 1
                        empty_cache(device)
            else:
                logger.warning("[%s] 批量推理失败（batch %d）: %s", exp_name, start, e)
                n_skipped += len(batch)
                continue
        except Exception as e:
            logger.warning("[%s] 批量推理失败（batch %d）: %s", exp_name, start, e)
            n_skipped += len(batch)
            continue

        for (tid, _, gt_action), generated in zip(batch, generated_batch):
            if not generated:
                continue
            type_match  = (parse_action_type(generated) == parse_action_type(gt_action))
            file_score  = file_path_match(generated, gt_action)
            rouge_l     = compute_rouge_l(generated, gt_action)
            exact_match = (generated.strip() == gt_action.strip())
            records.append({
                "trajectory_id":     tid,
                "action_type_match": type_match,
                "pred_action_type":  parse_action_type(generated),
                "gt_action_type":    parse_action_type(gt_action),
                "file_match_score":  file_score,
                "rouge_l":           rouge_l,
                "exact_match":       exact_match,
                "generated":         generated[:500],
                "ground_truth":      gt_action[:500],
            })

        done = min(start + batch_size, len(pending))
        elapsed = time.time() - t0
        if done % 20 < batch_size or done == len(pending):
            logger.info(
                "[%s] %d/%d  type_acc=%.3f  file=%.3f  rouge_l=%.3f  (%.0fs)",
                exp_name, done, len(pending),
                np.mean([r["action_type_match"] for r in records]) if records else 0,
                np.mean([r["file_match_score"] for r in records]) if records else 0,
                np.mean([r["rouge_l"] for r in records]) if records else 0,
                elapsed,
            )

    if not records:
        logger.error("[%s] 没有有效样本！", exp_name)
        return {}

    summary = {
        "n_evaluated":      len(records),
        "n_skipped":        n_skipped,
        "action_type_acc":  float(np.mean([r["action_type_match"] for r in records])),
        "file_match_score": float(np.mean([r["file_match_score"] for r in records])),
        "rouge_l":          float(np.mean([r["rouge_l"] for r in records])),
        "exact_match_acc":  float(np.mean([r["exact_match"] for r in records])),
        "action_type_dist_pred": _count_dist([r["pred_action_type"] for r in records]),
        "action_type_dist_gt":   _count_dist([r["gt_action_type"] for r in records]),
        "per_instance":     records,
    }
    logger.info(
        "[%s] 完成  type_acc=%.3f  file=%.3f  rouge_l=%.3f  n=%d",
        exp_name, summary["action_type_acc"], summary["file_match_score"],
        summary["rouge_l"], summary["n_evaluated"],
    )
    return summary


def _count_dist(items: list[str]) -> dict:
    dist: dict[str, int] = {}
    for x in items:
        dist[x] = dist.get(x, 0) + 1
    total = len(items)
    return {k: round(v / total, 3) for k, v in sorted(dist.items(), key=lambda x: -x[1])}


# ─────────────────────────────────────────────────────────────────────────
# CE Loss 加载
# ─────────────────────────────────────────────────────────────────────────

def load_ce_loss(csv_path: Path, models: list[str], split: str = "gold") -> dict[str, float]:
    if not csv_path.exists():
        logger.warning("perplexity_results.csv 不存在：%s", csv_path)
        return {}
    df = pd.read_csv(csv_path)
    df_split = df[df["split"] == split]
    result = {}
    for m in models:
        row = df_split[df_split["model"] == m]
        if row.empty:
            logger.warning("CE loss 未找到：model=%s, split=%s", m, split)
        else:
            result[m] = float(row["mean_loss"].iloc[0])
    return result


# ─────────────────────────────────────────────────────────────────────────
# 相关性分析
# ─────────────────────────────────────────────────────────────────────────

def correlation_analysis(
    summary_data: dict[str, dict],
    ce_loss_map: dict[str, float],
) -> dict:
    common_models = [m for m in summary_data if m in ce_loss_map and summary_data[m]]
    if len(common_models) < 3:
        logger.warning("可用数据点 %d 个（至少需要 3 个），跳过相关性分析。", len(common_models))
        return {}

    losses      = [ce_loss_map[m] for m in common_models]
    type_accs   = [summary_data[m]["action_type_acc"]  for m in common_models]
    file_scores = [summary_data[m]["file_match_score"] for m in common_models]
    rouge_ls    = [summary_data[m]["rouge_l"]          for m in common_models]

    corr_results = {}
    for metric_name, values in [
        ("action_type_acc",  type_accs),
        ("file_match_score", file_scores),
        ("rouge_l",          rouge_ls),
    ]:
        rho, p = stats.spearmanr(losses, values)
        corr_results[metric_name] = {
            "spearman_rho": float(rho) if not np.isnan(rho) else float("nan"),
            "p_value":      float(p) if not np.isnan(p) else float("nan"),
            "n":            len(common_models),
            "interpretation": (
                "strong negative" if rho < -0.7 else
                "moderate negative" if rho < -0.4 else
                "weak" if abs(rho) < 0.4 else
                "moderate positive" if rho < 0.7 else
                "strong positive"
            ) if not np.isnan(rho) else "N/A (constant)",
        }
        logger.info(
            "Spearman(CE loss, %s): rho=%.3f, p=%.4f  [%s]",
            metric_name, rho, p, corr_results[metric_name]["interpretation"],
        )

    return corr_results


# ─────────────────────────────────────────────────────────────────────────
# 论文级插图
# ─────────────────────────────────────────────────────────────────────────

def _setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    plt.rcParams.update({
        "font.family":       "serif",
        "font.serif":        ["Times New Roman", "DejaVu Serif"],
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "figure.dpi":        300,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "axes.linewidth":    0.8,
        "axes.grid":         True,
        "grid.alpha":        0.25,
        "grid.linewidth":    0.5,
    })
    return plt, mticker


def _get_style(model: str):
    return STYLE_MAP.get(model, (DISPLAY_NAMES.get(model, model), "o", "#333333"))


def generate_publication_figures(
    summary_data: dict[str, dict],
    ce_loss_map: dict[str, float],
    corr_results: dict,
    output_dir: Path,
):
    """生成全部论文级插图。"""
    try:
        plt, mticker = _setup_matplotlib()
    except Exception as e:
        logger.warning("matplotlib 初始化失败: %s", e)
        return

    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    common_models = [m for m in summary_data if m in ce_loss_map and summary_data[m]]
    if not common_models:
        logger.warning("无有效数据，跳过绘图")
        return

    losses      = [ce_loss_map[m] for m in common_models]
    type_accs   = [summary_data[m]["action_type_acc"]  for m in common_models]
    file_scores = [summary_data[m]["file_match_score"] for m in common_models]
    rouge_ls    = [summary_data[m]["rouge_l"]          for m in common_models]

    # numpy RankWarning 兼容性
    _RankWarning = getattr(np, "RankWarning", None) or getattr(
        np.polynomial.polynomial, "RankWarning", Warning
    )

    # ================================================================
    # Figure 1 — Scatter: CE Loss vs First-Action Quality
    # ================================================================
    metrics_cfg = [
        ("action_type_acc",  type_accs,   "Action Type Accuracy"),
        ("file_match_score", file_scores, "Target File Match"),
        ("rouge_l",          rouge_ls,    "ROUGE-L"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))
    panel_labels = ["(a)", "(b)", "(c)"]

    for idx, (ax, (metric_key, values, ylabel)) in enumerate(zip(axes, metrics_cfg)):
        rho = corr_results.get(metric_key, {}).get("spearman_rho", float("nan"))
        pv  = corr_results.get(metric_key, {}).get("p_value", float("nan"))

        for model, x, y in zip(common_models, losses, values):
            label, marker, color = _get_style(model)
            ax.scatter(x, y, marker=marker, color=color, s=60,
                       edgecolors="white", linewidths=0.5, zorder=4, label=label)

        # 线性回归 + 95% CI
        if len(losses) >= 3:
            xs = np.array(losses)
            ys = np.array(values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", _RankWarning)
                z = np.polyfit(xs, ys, 1)
            p_fn = np.poly1d(z)
            x_line = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 200)
            ax.plot(x_line, p_fn(x_line), color="#555555", ls="--", lw=1.0, zorder=2)

            n_boot = 2000
            boot_preds = np.zeros((n_boot, len(x_line)))
            rng = np.random.default_rng(42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", _RankWarning)
                for b in range(n_boot):
                    idx_b = rng.choice(len(xs), size=len(xs), replace=True)
                    if len(set(idx_b)) < 2:
                        boot_preds[b] = p_fn(x_line)
                        continue
                    zb = np.polyfit(xs[idx_b], ys[idx_b], 1)
                    boot_preds[b] = np.poly1d(zb)(x_line)
            lo = np.clip(np.percentile(boot_preds, 2.5,  axis=0), 0, 1)
            hi = np.clip(np.percentile(boot_preds, 97.5, axis=0), 0, 1)
            ax.fill_between(x_line, lo, hi, color="#cccccc", alpha=0.35, zorder=1)

        y_margin = max((max(values) - min(values)) * 0.25, 0.05)
        ax.set_ylim(max(0, min(values) - y_margin), min(1.0, max(values) + y_margin))
        ax.set_xlabel("Cross-Entropy Loss")
        ax.set_ylabel(ylabel)

        if pv < 0.01:
            sig = "***"
        elif pv < 0.05:
            sig = "**"
        elif pv < 0.1:
            sig = "*"
        else:
            sig = "" if not np.isnan(pv) else " (N/A)"
        rho_str = f"{rho:.2f}" if not np.isnan(rho) else "N/A"
        ax.text(
            0.03, 0.95,
            f"{panel_labels[idx]}  $\\rho$ = {rho_str}{sig}",
            transform=ax.transAxes, va="top", ha="left",
            fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               ncol=min(len(common_models), 6), frameon=False,
               bbox_to_anchor=(0.5, -0.08), columnspacing=1.0)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    for fmt in ("png", "pdf"):
        fig.savefig(fig_dir / f"fig_proxy_validation.{fmt}")
    plt.close(fig)
    logger.info("Figure 1 (scatter) saved: fig_proxy_validation.png/pdf")

    # ================================================================
    # Figure 2 — Grouped Bar: First-Action Metrics by Checkpoint
    # ================================================================
    bar_metrics = ["action_type_acc", "file_match_score", "rouge_l"]
    bar_labels  = ["Action Type Acc.", "File Match", "ROUGE-L"]

    fig2, ax2 = plt.subplots(figsize=(max(5.0, len(common_models) * 0.8), 3.0))

    n_models  = len(common_models)
    n_metrics = len(bar_metrics)
    bar_w     = 0.22
    x_pos     = np.arange(n_models)

    for j, (bm, bl) in enumerate(zip(bar_metrics, bar_labels)):
        means, ci_lo, ci_hi = [], [], []
        for m in common_models:
            per_inst = summary_data[m].get("per_instance", [])
            if per_inst and bm != "file_match_score":
                key = "action_type_match" if bm == "action_type_acc" else bm
                vals = [r[key] for r in per_inst if key in r]
                if vals:
                    mean_v = float(np.mean(vals))
                    rng2 = np.random.default_rng(42)
                    boots = [float(np.mean(rng2.choice(vals, size=len(vals), replace=True)))
                             for _ in range(2000)]
                    lo_v = np.percentile(boots, 2.5)
                    hi_v = np.percentile(boots, 97.5)
                    means.append(mean_v)
                    ci_lo.append(mean_v - lo_v)
                    ci_hi.append(hi_v - mean_v)
                    continue
            means.append(summary_data[m].get(bm, 0.0))
            ci_lo.append(0)
            ci_hi.append(0)

        color = ["#1f77b4", "#ff7f0e", "#2ca02c"][j]
        ax2.bar(
            x_pos + (j - 1) * bar_w, means, bar_w,
            yerr=[ci_lo, ci_hi], capsize=3,
            label=bl, color=color, edgecolor="white", linewidth=0.5,
            error_kw=dict(lw=0.8, capthick=0.8),
        )

    display_names = [_get_style(m)[0] for m in common_models]
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(display_names, rotation=20, ha="right")
    ax2.set_ylabel("Score")
    ax2.set_ylim(0, 1.0)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax2.legend(frameon=False, loc="upper left")
    ax2.set_title("First-Action Quality by Checkpoint", fontweight="bold")

    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig2.savefig(fig_dir / f"fig_first_action_bars.{fmt}")
    plt.close(fig2)
    logger.info("Figure 2 (bar chart) saved: fig_first_action_bars.png/pdf")

    # ================================================================
    # Figure 3 — ROUGE-L Scaling Curve (500 → 1000 → 2000)
    # ================================================================
    fig3, ax3 = plt.subplots(figsize=(5.0, 3.5))
    scales = [500, 1000, 2000]

    for strategy, exps, color, marker in [
        ("Random", ["exp1", "exp2", "exp14"], "#1f77b4", "s"),
        ("TopQ",   ["exp3", "exp4", "exp15"], "#ff7f0e", "o"),
    ]:
        rouge_vals = []
        available_scales = []
        for exp, scale in zip(exps, scales):
            if exp in summary_data and summary_data[exp]:
                rouge_vals.append(summary_data[exp]["rouge_l"])
                available_scales.append(scale)
        if rouge_vals:
            ax3.plot(available_scales, rouge_vals, f"-{marker}",
                     color=color, linewidth=1.5, markersize=7,
                     markeredgecolor="white", markeredgewidth=0.5,
                     label=strategy, zorder=3)

    # Baseline 水平线
    if "baseline" in summary_data and summary_data["baseline"]:
        bl_rouge = summary_data["baseline"]["rouge_l"]
        ax3.axhline(y=bl_rouge, color="#d62728", ls=":", lw=1.0, alpha=0.6,
                     label="Baseline (no SFT)")

    ax3.set_xlabel("Number of Training Trajectories")
    ax3.set_ylabel("ROUGE-L Score")
    ax3.set_title("First-Action ROUGE-L: Scaling Behavior", fontweight="bold")
    ax3.set_xticks(scales)
    ax3.set_xlim(350, 2150)
    ax3.legend(frameon=False, loc="upper left")

    plt.tight_layout()
    for fmt in ("png", "pdf"):
        fig3.savefig(fig_dir / f"fig_fa_scaling_curve.{fmt}")
    plt.close(fig3)
    logger.info("Figure 3 (scaling curve) saved: fig_fa_scaling_curve.png/pdf")

    # ================================================================
    # Figure 4 — B2-Only vs Composite Ablation (ROUGE-L + File Match)
    # ================================================================
    ablation_models = ["exp3", "exp16"]  # Composite-Top500 vs B2Only-Top500
    ablation_data = {}
    for m in ablation_models:
        if m in summary_data and summary_data[m]:
            ablation_data[m] = summary_data[m]

    if len(ablation_data) == 2:
        fig4, ax4 = plt.subplots(figsize=(4.0, 3.0))
        abl_metrics = ["rouge_l", "file_match_score", "action_type_acc"]
        abl_labels  = ["ROUGE-L", "File Match", "Action Type Acc."]
        x = np.arange(len(abl_metrics))
        w = 0.35

        for i, (m, color, label) in enumerate([
            ("exp3",  "#ff7f0e", "Composite-Top500"),
            ("exp16", "#756bb1", "B2Only-Top500"),
        ]):
            vals = [ablation_data[m].get(metric, 0.0) for metric in abl_metrics]
            ax4.bar(x + (i - 0.5) * w, vals, w, color=color, label=label,
                    edgecolor="white", linewidth=0.5)

        ax4.set_xticks(x)
        ax4.set_xticklabels(abl_labels)
        ax4.set_ylabel("Score")
        ax4.set_ylim(0, 1.0)
        ax4.set_title("First-Action: Composite vs B2-Only", fontweight="bold")
        ax4.legend(frameon=False)

        plt.tight_layout()
        for fmt in ("png", "pdf"):
            fig4.savefig(fig_dir / f"fig_fa_b2only_ablation.{fmt}")
        plt.close(fig4)
        logger.info("Figure 4 (B2-only ablation) saved: fig_fa_b2only_ablation.png/pdf")
    else:
        logger.info("B2-Only ablation 数据不足（需要 exp3 和 exp16），跳过")


# ─────────────────────────────────────────────────────────────────────────
# 报告生成
# ─────────────────────────────────────────────────────────────────────────

def write_report(
    summary_data: dict[str, dict],
    ce_loss_map: dict[str, float],
    corr_results: dict,
    output_dir: Path,
) -> None:
    lines = [
        "# First-Action Evaluation Report (Extended)",
        "",
        "## Summary Table",
        "",
        "| Checkpoint | Display Name | CE Loss | Type Acc | File Match | ROUGE-L | N |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, s in summary_data.items():
        if not s:
            continue
        ce = ce_loss_map.get(m, float("nan"))
        display = DISPLAY_NAMES.get(m, m)
        lines.append(
            f"| {m} | {display} | {ce:.4f} | {s['action_type_acc']:.3f} | "
            f"{s['file_match_score']:.3f} | {s['rouge_l']:.3f} | {s['n_evaluated']} |"
        )

    lines += [
        "",
        "## Spearman Correlation (CE Loss vs Action Quality)",
        "",
        "| Metric | Spearman rho | p-value | Interpretation |",
        "|---|---|---|---|",
    ]
    for metric, cr in corr_results.items():
        rho_str = f"{cr['spearman_rho']:.3f}" if not np.isnan(cr['spearman_rho']) else "N/A"
        p_str   = f"{cr['p_value']:.4f}" if not np.isnan(cr['p_value']) else "N/A"
        lines.append(f"| {metric} | {rho_str} | {p_str} | {cr['interpretation']} |")

    lines += [
        "",
        "## Notes",
        "- CE Loss from `eval_results/perplexity_results.csv` (gold split)",
        "- Spearman rho < 0 is expected: lower loss -> higher action quality",
        "- rho < -0.7 indicates strong proxy validity",
        f"- Models evaluated: {', '.join(summary_data.keys())}",
        "",
    ]

    report_path = output_dir / "first_action_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("报告已保存: %s", report_path)


# ─────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="First-Action Evaluation（含 exp14-16 新实验）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--models", nargs="+",
        default=["exp14", "exp15", "exp16"],
        help="要评估的 checkpoint（默认: exp14 exp15 exp16）",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["gold"],
        choices=["gold", "random", "low_q"],
    )
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=GENERATE_BATCH_SIZE)
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument(
        "--outputs-dir", type=Path, default=None,
        help="LoRA adapter 根目录（默认: <脚本目录>/outputs）",
    )
    parser.add_argument("--device", default=None, choices=["mps", "cuda", "cpu"])
    parser.add_argument("--ce-loss-split", default="gold", choices=["gold", "random", "low_q"])
    parser.add_argument(
        "--skip-inference", action="store_true",
        help="跳过推理，直接读取已有结果做分析和绘图",
    )
    parser.add_argument(
        "--plot-only", action="store_true",
        help="只生成图表（等同 --skip-inference，语义更清晰）",
    )
    parser.add_argument("--force", action="store_true", help="强制重新评估（忽略已有结果）")
    parser.add_argument("--eval-data-dir", type=Path, default=None)
    args = parser.parse_args()

    outputs_root = args.outputs_dir or OUTPUTS_ROOT
    eval_data    = args.eval_data_dir or EVAL_DATA
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    result_path = OUTPUT_DIR / "first_action_results.json"
    skip_inference = args.skip_inference or args.plot_only

    # ── 推理阶段 ──────────────────────────────────────────────────────────
    if not skip_inference:
        device = detect_device(args.device)
        dtype  = get_torch_dtype(device)

        all_samples: list[dict] = []
        for split in args.splits:
            jsonl_path = eval_data / f"{split}.jsonl"
            if not jsonl_path.exists():
                logger.error("测试集文件不存在: %s", jsonl_path)
                continue
            with open(jsonl_path, encoding="utf-8") as f:
                rows = [json.loads(line) for line in f if line.strip()]
            logger.info("[%s] 加载 %d 条轨迹", split, len(rows))
            all_samples.extend(rows)

        if not all_samples:
            logger.error("没有可用测试数据，退出。")
            return

        if result_path.exists() and not args.force:
            with open(result_path, encoding="utf-8") as f:
                all_results: dict = json.load(f)
            logger.info("已加载历史结果，包含: %s", list(all_results.keys()))
        else:
            all_results = {}

        tokenizer, base_model = load_base_model(args.base_model, device, dtype)

        for exp_name in args.models:
            if not args.force and exp_name in all_results:
                logger.info("[%s] 已有结果，跳过推理。（使用 --force 重跑）", exp_name)
                continue

            if exp_name not in EXPERIMENT_MAP:
                logger.error("未知 checkpoint: %s，可用: %s", exp_name, list(EXPERIMENT_MAP.keys()))
                continue

            logger.info("=" * 55)
            logger.info("评估 checkpoint: %s", exp_name)

            adapter_rel = EXPERIMENT_MAP[exp_name]
            if adapter_rel is not None:
                adapter_path = outputs_root / adapter_rel
                if not adapter_path.exists():
                    logger.error(
                        "LoRA adapter 目录不存在: %s\n"
                        "  -> 请通过 --outputs-dir 指定 adapter 根目录",
                        adapter_path,
                    )
                    continue
                model = load_lora_model(base_model, str(adapter_path), device, dtype)
            else:
                model = base_model

            summary = evaluate_checkpoint(
                model, tokenizer, all_samples, exp_name,
                device, args.num_samples, batch_size=args.batch_size,
            )
            all_results[exp_name] = summary

            with open(result_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            logger.info("中间结果已保存: %s", result_path)

            if adapter_rel is not None:
                base_model = unload_lora(model, device)

        del base_model, tokenizer
        empty_cache(device)

    else:
        if not result_path.exists():
            logger.error("找不到 %s，请先运行推理", result_path)
            return
        with open(result_path, encoding="utf-8") as f:
            all_results = json.load(f)
        logger.info("跳过推理，使用已有结果: %s", list(all_results.keys()))

    # ── 相关性分析 ────────────────────────────────────────────────────────
    # 收集所有有结果的模型名
    evaluated_models = [m for m in all_results if all_results[m]]
    ce_loss_map = load_ce_loss(
        OUTPUT_DIR / "perplexity_results.csv",
        evaluated_models,
        split=args.ce_loss_split,
    )
    logger.info("CE loss 数据: %s", {k: f"{v:.4f}" for k, v in ce_loss_map.items()})

    summary_only = {
        m: {k: v for k, v in s.items() if k != "per_instance"}
        for m, s in all_results.items() if s
    }

    # Summary CSV
    rows = []
    for m, s in summary_only.items():
        row = {"model": m, "display_name": DISPLAY_NAMES.get(m, m),
               "ce_loss": ce_loss_map.get(m, float("nan"))}
        row.update(s)
        rows.append(row)
    if rows:
        df_summary = pd.DataFrame(rows)
        csv_path = OUTPUT_DIR / "first_action_summary.csv"
        df_summary.to_csv(csv_path, index=False)
        logger.info("Summary CSV 已保存: %s", csv_path)
        cols = ["model", "display_name", "ce_loss", "action_type_acc",
                "file_match_score", "rouge_l", "n_evaluated"]
        print("\n" + df_summary[[c for c in cols if c in df_summary.columns]].to_string(index=False))

    corr_results = correlation_analysis(all_results, ce_loss_map)

    if corr_results:
        corr_path = OUTPUT_DIR / "first_action_correlation.json"
        with open(corr_path, "w", encoding="utf-8") as f:
            json.dump(corr_results, f, indent=2, ensure_ascii=False)
        logger.info("相关性结果已保存: %s", corr_path)

    # ── 绘图 ──────────────────────────────────────────────────────────────
    generate_publication_figures(all_results, ce_loss_map, corr_results, OUTPUT_DIR)

    write_report(summary_only, ce_loss_map, corr_results, OUTPUT_DIR)

    logger.info("=" * 55)
    logger.info("First-Action Evaluation 完成！")
    logger.info("输出文件：")
    logger.info("  推理结果:   %s", result_path)
    logger.info("  汇总 CSV:   %s", OUTPUT_DIR / "first_action_summary.csv")
    logger.info("  相关系数:   %s", OUTPUT_DIR / "first_action_correlation.json")
    logger.info("  报告:       %s", OUTPUT_DIR / "first_action_report.md")
    logger.info("  Figures:    %s", OUTPUT_DIR / "figures")


if __name__ == "__main__":
    main()
