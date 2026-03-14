"""
Step 1: Build evaluation test sets.

从67k轨迹数据中构建三个不重叠的测试集：
  - Gold Test Set   : 200条高质量轨迹 (composite_score 前10% + resolved==1)
  - Random Test Set : 200条随机轨迹
  - Low-Q Test Set  : 200条低质量轨迹 (composite_score 后10% + error/timeout)

确保与所有训练子集无重叠（按trajectory_id去重）。

Usage:
    # 自动使用本地 trajectory_scored_v3.csv（推荐）
    python build_test_set.py

    # 手动指定打分文件
    python build_test_set.py --scores-file /workspace/trajectory_scored_v3.csv

    # 先用 generate_quality_scores.py 转换格式，再传入
    python generate_quality_scores.py  &&  python build_test_set.py --scores-file quality_scores.csv
"""

import argparse
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from huggingface_hub import login

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────
WORKSPACE = Path(os.environ.get("WORKSPACE", Path(__file__).parent))
EVAL_DATA_DIR = WORKSPACE / "eval_data"

# 数据源
SUBSET_HF_REPO   = "davongluck/swe-bench-trajectory-quality-subsets"
RAW_HF_DATASET   = "nebius/SWE-rebench-openhands-trajectories"

# 所有训练子集名称（需从测试集中排除）— v3 实验矩阵
ALL_TRAINING_SUBSETS = [
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
]

TEST_SET_SIZE = 200
RANDOM_SEED   = 42


# ── Step A: 收集所有训练集中使用的trajectory_id（用于排除）─────────────
def get_all_training_ids() -> set:
    """从HF加载所有训练子集的trajectory_id，用于测试集去重。"""
    all_ids = set()
    for subset_name in ALL_TRAINING_SUBSETS:
        try:
            ds = load_dataset(SUBSET_HF_REPO, subset_name, split="train")
            ids = set(ds["trajectory_id"])
            all_ids.update(ids)
            logger.info("  [%s] %d IDs", subset_name, len(ids))
        except Exception as e:
            logger.warning("  [%s] 加载失败: %s", subset_name, e)
    logger.info("训练集共用ID总数: %d", len(all_ids))
    return all_ids


# ── Step B: 加载质量得分数据 ──────────────────────────────────────────────
def load_quality_scores(scores_repo: str = None, scores_file: str = None) -> pd.DataFrame:
    """
    加载轨迹质量得分。返回 DataFrame，必须包含列：
      trajectory_id, quality_score, exit_status

    优先级: scores_file > scores_repo > 从raw数据推断exit_status
    """
    if scores_file and Path(scores_file).exists():
        logger.info("从本地文件加载质量得分: %s", scores_file)
        if scores_file.endswith(".csv"):
            df = pd.read_csv(scores_file)
        else:
            df = pd.read_json(scores_file)
        # 兼容 v3 列名（composite_score）和旧列名（composite_Q）
        if "quality_score" not in df.columns:
            if "composite_score" in df.columns:
                df = df.rename(columns={"composite_score": "quality_score"})
            elif "composite_Q" in df.columns:
                df = df.rename(columns={"composite_Q": "quality_score"})
        logger.info("加载 %d 条得分记录", len(df))
        return df

    if scores_repo:
        logger.info("从HF加载质量得分: %s", scores_repo)
        try:
            ds = load_dataset(scores_repo, split="train")
            df = ds.to_pandas()
            # 兼容 v3 列名（composite_score）和旧列名（composite_Q）
            if "quality_score" not in df.columns:
                if "composite_score" in df.columns:
                    df = df.rename(columns={"composite_score": "quality_score"})
                elif "composite_Q" in df.columns:
                    df = df.rename(columns={"composite_Q": "quality_score"})
            logger.info("加载 %d 条得分记录", len(df))
            return df
        except Exception as e:
            logger.warning("HF加载失败: %s，将从raw数据推断", e)

    # Fallback: 从raw数据中直接流式提取exit_status，无quality_score
    logger.info("从raw数据中推断exit_status（无质量得分，将用随机采样替代Gold/Low-Q分层）")
    return _extract_exit_status_from_raw()


def _extract_exit_status_from_raw(max_scan: int = 70000) -> pd.DataFrame:
    """
    流式扫描原始数据集，提取trajectory_id和exit_status。
    当无质量得分时的fallback方案。
    """
    logger.info("流式扫描 %s ...", RAW_HF_DATASET)
    raw_ds = load_dataset(RAW_HF_DATASET, split="train", streaming=True)

    records = []
    for i, row in enumerate(raw_ds):
        if i >= max_scan:
            break
        tid = row.get("trajectory_id", "")
        exit_status = row.get("exit_status", row.get("status", "unknown"))
        records.append({"trajectory_id": tid, "exit_status": str(exit_status), "quality_score": None})
        if (i + 1) % 10000 == 0:
            logger.info("  已扫描 %d 条", i + 1)

    df = pd.DataFrame(records)
    logger.info("共扫描 %d 条轨迹", len(df))
    return df


# ── Step C: 构建三个测试集 ────────────────────────────────────────────────
def build_test_sets(df: pd.DataFrame, training_ids: set) -> dict:
    """
    从质量得分DataFrame构建 Gold / Random / Low-Q 三个测试集。

    Args:
        df: 含 trajectory_id, quality_score, exit_status 的DataFrame
        training_ids: 所有训练集ID，需排除

    Returns:
        dict with keys 'gold', 'random', 'low_q', each is a list of trajectory_ids
    """
    rng = random.Random(RANDOM_SEED)

    # 排除训练集ID
    df_eligible = df[~df["trajectory_id"].isin(training_ids)].copy()
    logger.info("排除训练集后，可用轨迹数: %d", len(df_eligible))

    has_scores = df_eligible["quality_score"].notna().any()

    # ── Gold Test Set ──────────────────────────────────────────────────────
    if has_scores:
        # Q(t) 前10% + exit_status=submit
        q90 = df_eligible["quality_score"].quantile(0.90)
        gold_pool = df_eligible[
            (df_eligible["quality_score"] >= q90) &
            (df_eligible["exit_status"].str.lower().isin(["submit", "success", "resolved"]))
        ]
        if len(gold_pool) < TEST_SET_SIZE:
            logger.warning("Gold候选不足 (%d < %d)，放宽条件只用Q前10%%", len(gold_pool), TEST_SET_SIZE)
            gold_pool = df_eligible[df_eligible["quality_score"] >= q90]
    else:
        # Fallback: exit_status=submit的轨迹作为Gold
        gold_pool = df_eligible[
            df_eligible["exit_status"].str.lower().isin(["submit", "success", "resolved"])
        ]

    gold_ids = gold_pool["trajectory_id"].tolist()
    rng.shuffle(gold_ids)
    gold_selected = gold_ids[:TEST_SET_SIZE]
    logger.info("Gold Test Set: %d/%d 候选中选取 %d 条", len(gold_pool), len(df_eligible), len(gold_selected))

    # ── Low-Q Test Set ──────────────────────────────────────────────────────
    exclude_for_lowq = training_ids | set(gold_selected)
    df_for_lowq = df[~df["trajectory_id"].isin(exclude_for_lowq)].copy()

    if has_scores:
        q10 = df_for_lowq["quality_score"].quantile(0.10)
        lowq_pool = df_for_lowq[
            (df_for_lowq["quality_score"] <= q10) &
            (df_for_lowq["exit_status"].str.lower().isin(["error", "timeout", "failed", "exit_cost"]))
        ]
        if len(lowq_pool) < TEST_SET_SIZE:
            logger.info(
                "Low-Q：exit_status 过滤后候选不足 (%d)，改用纯质量排序后10%% "
                "（trajectory_scored_v3.csv 仅含 resolved 轨迹，此为预期行为）",
                len(lowq_pool),
            )
            lowq_pool = df_for_lowq[df_for_lowq["quality_score"] <= q10]
    else:
        lowq_pool = df_for_lowq[
            df_for_lowq["exit_status"].str.lower().isin(["error", "timeout", "failed", "exit_cost"])
        ]

    lowq_ids = lowq_pool["trajectory_id"].tolist()
    rng.shuffle(lowq_ids)
    lowq_selected = lowq_ids[:TEST_SET_SIZE]
    logger.info("Low-Q Test Set: %d/%d 候选中选取 %d 条", len(lowq_pool), len(df_for_lowq), len(lowq_selected))

    # ── Random Test Set ──────────────────────────────────────────────────────
    exclude_for_rand = training_ids | set(gold_selected) | set(lowq_selected)
    df_for_rand = df[~df["trajectory_id"].isin(exclude_for_rand)].copy()
    rand_ids = df_for_rand["trajectory_id"].tolist()
    rng.shuffle(rand_ids)
    rand_selected = rand_ids[:TEST_SET_SIZE]
    logger.info("Random Test Set: 从 %d 可用中随机选取 %d 条", len(df_for_rand), len(rand_selected))

    return {
        "gold":   gold_selected,
        "random": rand_selected,
        "low_q":  lowq_selected,
    }


# ── Step D: 从raw数据中提取选中轨迹的完整内容 ────────────────────────────
def fetch_trajectories(target_ids: set, output_path: Path, split_name: str):
    """
    从raw数据集流式提取目标轨迹，保存为JSONL。
    """
    output_path.mkdir(parents=True, exist_ok=True)
    out_file = output_path / f"{split_name}.jsonl"

    if out_file.exists():
        existing = sum(1 for _ in open(out_file))
        if existing >= len(target_ids):
            logger.info("[%s] 已存在 %d 条，跳过", split_name, existing)
            return out_file

    logger.info("正在提取 [%s] %d 条轨迹...", split_name, len(target_ids))
    raw_ds = load_dataset(RAW_HF_DATASET, split="train", streaming=True)

    found = 0
    with open(out_file, "w") as f:
        for row in raw_ds:
            tid = row.get("trajectory_id", "")
            if tid in target_ids:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                found += 1
                if found % 50 == 0:
                    logger.info("  [%s] 已找到 %d/%d", split_name, found, len(target_ids))
                if found >= len(target_ids):
                    break

    logger.info("[%s] 共提取 %d 条轨迹 → %s", split_name, found, out_file)
    return out_file


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    global TEST_SET_SIZE
    parser = argparse.ArgumentParser(description="构建评估测试集")
    parser.add_argument("--scores-repo",  type=str, default=None,
                        help="HF质量得分数据集路径（多配置dataset需搭配config使用）")
    parser.add_argument("--scores-file",  type=str, default=None,
                        help="本地打分文件路径 (CSV/JSON)，支持 trajectory_scored_v3.csv 直接传入；"
                             "若不指定，自动查找 trajectory_scored_v3.csv / quality_scores.csv")
    parser.add_argument("--no-fetch",     action="store_true",
                        help="只生成ID列表，不从HF提取完整轨迹内容")
    parser.add_argument("--size",         type=int, default=TEST_SET_SIZE,
                        help="每个测试集大小 (默认200)")
    args = parser.parse_args()

    TEST_SET_SIZE = args.size

    # 自动检测本地打分文件（未手动指定时）
    if not args.scores_file and not args.scores_repo:
        for candidate in ["trajectory_scored_v3.csv", "quality_scores.csv"]:
            candidate_path = WORKSPACE / candidate
            if candidate_path.exists():
                args.scores_file = str(candidate_path)
                logger.info("自动检测到本地打分文件: %s", args.scores_file)
                break

    # HF登录
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)

    EVAL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step A: 收集训练集ID
    logger.info("=" * 60)
    logger.info("Step A: 收集所有训练集ID（用于排除）")
    training_ids = get_all_training_ids()

    # Step B: 加载质量得分
    logger.info("=" * 60)
    logger.info("Step B: 加载质量得分数据")
    df_scores = load_quality_scores(args.scores_repo, args.scores_file)

    # Step C: 构建测试集ID列表
    logger.info("=" * 60)
    logger.info("Step C: 构建三个测试集")
    test_sets = build_test_sets(df_scores, training_ids)

    # 保存ID列表
    id_file = EVAL_DATA_DIR / "test_set_ids.json"
    with open(id_file, "w") as f:
        json.dump({
            "gold":   test_sets["gold"],
            "random": test_sets["random"],
            "low_q":  test_sets["low_q"],
            "total_training_ids": len(training_ids),
            "seed": RANDOM_SEED,
        }, f, indent=2)
    logger.info("ID列表已保存: %s", id_file)

    # Step D: 提取完整轨迹（可选）
    if not args.no_fetch:
        logger.info("=" * 60)
        logger.info("Step D: 从原始数据集提取完整轨迹内容")
        for split_name, ids in test_sets.items():
            fetch_trajectories(set(ids), EVAL_DATA_DIR, split_name)

    logger.info("=" * 60)
    logger.info("测试集构建完成!")
    logger.info("  Gold   : %d 条 → %s/gold.jsonl",   len(test_sets["gold"]),   EVAL_DATA_DIR)
    logger.info("  Random : %d 条 → %s/random.jsonl", len(test_sets["random"]), EVAL_DATA_DIR)
    logger.info("  Low-Q  : %d 条 → %s/low_q.jsonl",  len(test_sets["low_q"]),  EVAL_DATA_DIR)


if __name__ == "__main__":
    main()
