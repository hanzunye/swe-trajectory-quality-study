"""
Step 4: 统计分析与可视化。

生成以下图表：
  1. Bar chart : 各模型在gold/random/low-Q测试集上的mean_loss
  2. Ablation heatmap : Block2/3消融系列各模型的loss差异（相对TopQ-500）
  3. 核心假设验证图 : H1/H2/H4（Gate效果、评分效果、Sanity check）
  4. 统计显著性检验 : Mann-Whitney U test
  5. Per-trajectory loss分布箱线图

输出：
  /workspace/eval_results/figures/*.png
  /workspace/eval_results/stats_report.md

Usage:
    python analyze_results.py
    python analyze_results.py --results /workspace/eval_results/perplexity_results.json
"""

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # 无头环境（RunPod）
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── 路径配置 ──────────────────────────────────────────────────────────────
WORKSPACE   = Path(os.environ.get("WORKSPACE", Path(__file__).resolve().parents[2]))
RESULTS_DIR = WORKSPACE / "data" / "eval_results"
FIGURES_DIR = RESULTS_DIR / "figures"

# 模型标签（用于图表显示）— v3 实验设计
MODEL_LABELS = {
    "baseline":  "Baseline\n(no SFT)",
    "exp1":      "Random\n500",
    "exp2":      "Random\n1000",
    "exp3":      "TopQ\n500",
    "exp4":      "TopQ\n1000",
    "exp5":      "ResolvedOnly\n500",
    "exp6":      "ResolvedOnly\n1000",
    "exp7":      "BottomQ\n500",
    "exp8":      "Ablation\nNoEfficiency",
    "exp9":      "Ablation\nNoStyle",
    "exp10":     "Ablation\nNoB2",
    "exp11":     "Ablation\nNoB3",
    "exp12":     "Ablation\nNoC2",
    "exp13":     "Ablation\nNoC3",
}

SPLIT_COLORS = {
    "gold":   "#2ecc71",   # 绿色
    "random": "#3498db",   # 蓝色
    "low_q":  "#e74c3c",   # 红色
}

SPLIT_LABELS = {
    "gold":   "Gold (High-Q)",
    "random": "Random",
    "low_q":  "Low-Q",
}

plt.rcParams.update({
    "figure.dpi":       150,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  9,
})


# ── 数据加载 ──────────────────────────────────────────────────────────────
def load_results(results_path: Path) -> tuple[dict, pd.DataFrame]:
    """加载JSON结果，返回原始dict和汇总DataFrame。"""
    with open(results_path) as f:
        raw = json.load(f)

    rows = []
    for exp_name, exp_data in raw.items():
        for split_name in ["gold", "random", "low_q"]:
            if split_name not in exp_data:
                continue
            r = exp_data[split_name]
            rows.append({
                "model":      exp_name,
                "split":      split_name,
                "mean_loss":  r.get("mean_loss"),
                "std_loss":   r.get("std_loss"),
                "perplexity": r.get("perplexity"),
                "n_valid":    r.get("n_valid", 0),
                "losses":     r.get("losses", []),
            })

    df = pd.DataFrame(rows)
    return raw, df


# ── 图表1: 各模型在三个测试集上的Loss对比（分组柱状图）──────────────────
def plot_loss_comparison(df: pd.DataFrame, save_path: Path):
    """
    分组柱状图：X轴=模型，每组3条柱（gold/random/low_q），Y轴=mean_loss。
    """
    # 只保留有数据的模型，按exp编号排序
    models_ordered = [m for m in ["baseline","exp1","exp2","exp3","exp4","exp5","exp6",
                                   "exp7","exp8","exp9","exp10","exp11","exp12","exp13"]
                      if m in df["model"].values]
    splits = ["gold", "random", "low_q"]

    n_models = len(models_ordered)
    n_splits = len(splits)
    bar_width = 0.25
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(12, n_models * 1.2), 6))

    for i, split in enumerate(splits):
        split_df = df[df["split"] == split].set_index("model")
        means = [split_df.loc[m, "mean_loss"] if m in split_df.index else np.nan
                 for m in models_ordered]
        stds  = [split_df.loc[m, "std_loss"] if m in split_df.index else 0
                 for m in models_ordered]
        offset = (i - n_splits / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset, means,
            width=bar_width, label=SPLIT_LABELS[split],
            color=SPLIT_COLORS[split], alpha=0.85,
            yerr=stds, capsize=3, error_kw={"linewidth": 1},
        )

    ax.set_xlabel("Model")
    ax.set_ylabel("Mean Cross-Entropy Loss (assistant tokens only)")
    ax.set_title("Perplexity Evaluation: Loss Comparison Across Models and Test Sets\n"
                 "(Lower is better; ↓ means model fits the test distribution better)")
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_ordered], ha="center")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.4)

    # 标注baseline参考线
    if "baseline" in df["model"].values:
        for split in splits:
            row = df[(df["model"] == "baseline") & (df["split"] == split)]
            if not row.empty:
                ax.axhline(
                    row.iloc[0]["mean_loss"],
                    color=SPLIT_COLORS[split], linestyle="--", alpha=0.4, linewidth=1,
                )

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("图1已保存: %s", save_path)


# ── 图表2: Ablation Heatmap（相对baseline的loss变化）────────────────────
def plot_ablation_heatmap(df: pd.DataFrame, save_path: Path):
    """
    热力图：行=消融模型(exp8-exp13), 列=测试集, 值=相对TopQ-500(exp3)的loss变化（%）。
    Block 2: exp8/exp9（大维度消融）; Block 3: exp10-exp13（子维度消融）
    """
    # Block2 + Block3 消融模型（exp7=BottomQ 是 sanity check，不属于消融）
    ablation_models = [m for m in ["exp8","exp9","exp10","exp11","exp12","exp13"]
                       if m in df["model"].values]
    splits = ["gold", "random", "low_q"]

    # 基准：TopQ-500（exp3），而非 baseline
    ref_model = "exp3"
    ref_loss = {}
    for split in splits:
        row = df[(df["model"] == ref_model) & (df["split"] == split)]
        ref_loss[split] = row.iloc[0]["mean_loss"] if not row.empty else np.nan

    # v3 消融标签
    ablation_labels = {
        "exp8":  "Block2: No Efficiency\n(Style only)",
        "exp9":  "Block2: No Style\n(Efficiency only)",
        "exp10": "Block3: No B2\n(error_retry removed)",
        "exp11": "Block3: No B3\n(step_count removed)",
        "exp12": "Block3: No C2\n(action_diversity removed)",
        "exp13": "Block3: No C3\n(obs_utilization removed)",
    }

    matrix = []
    row_labels = []
    for m in ablation_models:
        row_labels.append(ablation_labels.get(m, m))
        row_data = []
        for split in splits:
            cell = df[(df["model"] == m) & (df["split"] == split)]
            if cell.empty or np.isnan(ref_loss.get(split, np.nan)):
                row_data.append(np.nan)
            else:
                delta_pct = (cell.iloc[0]["mean_loss"] - ref_loss[split]) / ref_loss[split] * 100
                row_data.append(delta_pct)
        matrix.append(row_data)

    matrix = np.array(matrix)
    col_labels = [SPLIT_LABELS[s] for s in splits]

    if matrix.size == 0 or np.all(np.isnan(matrix)):
        logger.warning("无Ablation数据，跳过热力图")
        return

    fig, ax = plt.subplots(figsize=(9, max(5, len(ablation_models) * 1.1)))
    vmax = max(abs(np.nanmin(matrix)), abs(np.nanmax(matrix)), 0.1)
    annot_labels = np.where(
        np.isnan(matrix), "",
        np.vectorize(lambda v: f"{v:+.1f}%")(matrix)
    )
    sns.heatmap(
        matrix,
        annot=annot_labels, fmt="",
        xticklabels=col_labels, yticklabels=row_labels,
        cmap="RdYlGn_r",
        center=0, vmin=-vmax, vmax=vmax,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 10},
    )
    ax.set_title("Ablation Heatmap: Loss Change (%) vs TopQ-500\n"
                 "(Red = worse than full composite score, Green = better)")
    ax.set_xlabel("Test Set")
    ax.set_ylabel("Removed Quality Dimension")
    ax.tick_params(axis="both", labelsize=9)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("图2已保存: %s", save_path)


# ── 图表3: 核心模型对比（H1/H2假设验证）────────────────────────────────
def plot_hypothesis_comparison(df: pd.DataFrame, save_path: Path):
    """
    核心假设验证图（v3）：
      H1 (Gate效果):  ResolvedOnly-500 < Random-500
      H2 (评分效果):  TopQ-500 < ResolvedOnly-500
      H4 (Sanity):    TopQ-500 << BottomQ-500
    对比模型：Baseline / Random-500(exp1) / ResolvedOnly-500(exp5) / TopQ-500(exp3) / BottomQ-500(exp7)
    """
    # exp1=Random-500, exp5=ResolvedOnly-500, exp3=TopQ-500, exp7=BottomQ-500
    key_models = ["baseline", "exp1", "exp5", "exp3", "exp7"]
    splits = ["gold", "random", "low_q"]
    colors = ["#95a5a6", "#3498db", "#f39c12", "#2ecc71", "#e74c3c"]
    labels = [
        "Baseline",
        "Random-500",
        "ResolvedOnly-500\n(H1: Gate)",
        "TopQ-500\n(H2: Score)",
        "BottomQ-500\n(H4: Sanity)",
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)

    for ax, split in zip(axes, splits):
        split_df = df[df["split"] == split].set_index("model")
        for m, color, label in zip(key_models, colors, labels):
            if m not in split_df.index:
                continue
            row  = split_df.loc[m]
            mean = row["mean_loss"]
            std  = row["std_loss"] if not pd.isna(row["std_loss"]) else 0
            ax.bar(label, mean, color=color, alpha=0.8, yerr=std, capsize=4,
                   error_kw={"linewidth": 1.5})

        ax.set_title(f"{SPLIT_LABELS[split]}\nTest Set")
        ax.set_ylabel("Mean Loss" if split == "gold" else "")
        ax.tick_params(axis="x", labelsize=7.5)
        ax.grid(axis="y", alpha=0.4)

    fig.suptitle(
        "Hypothesis Validation (v3)\n"
        "H1: ResolvedOnly < Random (Gate effect)  |  "
        "H2: TopQ < ResolvedOnly (Score effect)  |  "
        "H4: TopQ << BottomQ (Sanity check)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("图3已保存: %s", save_path)


# ── 图表4: Loss分布箱线图 ────────────────────────────────────────────────
def plot_loss_distribution(raw_results: dict, save_path: Path):
    """
    箱线图：展示per-trajectory loss分布（不仅仅是均值）。
    核心对比：Baseline / Random-500 / ResolvedOnly-500 / TopQ-500 / BottomQ-500
    """
    key_models = ["baseline", "exp1", "exp5", "exp3", "exp7"]
    split = "gold"

    data  = []
    names = []
    for m in key_models:
        if m not in raw_results:
            continue
        split_data = raw_results[m].get(split, {})
        losses = split_data.get("losses", [])
        if losses:
            data.append(losses)
            names.append(MODEL_LABELS.get(m, m).replace("\n", " "))

    if not data:
        logger.warning("无per-trajectory losses数据（确保eval_perplexity.py保存了losses列表）")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, labels=names, patch_artist=True, notch=True,
                    medianprops={"color": "black", "linewidth": 2})
    colors = ["#95a5a6", "#3498db", "#f39c12", "#2ecc71", "#e74c3c"][:len(data)]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Model")
    ax.set_ylabel("Per-Trajectory Loss (Gold Test Set)")
    ax.set_title("Loss Distribution on Gold Test Set\n"
                 "(Lower median = better fit to high-quality trajectories)")
    ax.grid(axis="y", alpha=0.4)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    logger.info("图4已保存: %s", save_path)


# ── 统计检验 ──────────────────────────────────────────────────────────────
def run_statistical_tests(raw_results: dict) -> str:
    """
    Mann-Whitney U 检验：比较各模型在gold测试集上的loss分布。
    验证 v3 研究假设 H1–H7。返回 Markdown 格式报告。
    """
    def get_losses(exp_name: str) -> list:
        return raw_results.get(exp_name, {}).get("gold", {}).get("losses", [])

    def mwu_less(a_losses, b_losses, label_a, label_b):
        """单侧检验：a < b（a 的 loss 更低）"""
        if not a_losses or not b_losses:
            return f"- 数据不足，跳过检验"
        u, p = stats.mannwhitneyu(a_losses, b_losses, alternative="less")
        supported = "**Supported** ✓" if p < 0.05 else "Not supported ✗"
        return (
            f"- Mann-Whitney U={u:.0f}, p={p:.4f} → {supported}\n"
            f"- Mean loss: {label_a}={np.mean(a_losses):.4f}, {label_b}={np.mean(b_losses):.4f}"
        )

    lines = ["# Statistical Significance Report (v3)\n"]

    # ── 全模型 vs Baseline 汇总表 ────────────────────────────────────────────
    lines += [
        "## All Models vs Baseline (Gold Test Set, two-sided)\n",
        "| Model | Direction | U-stat | p-value | Significant | Effect Size r |",
        "|-------|-----------|--------|---------|-------------|---------------|",
    ]
    baseline_losses = get_losses("baseline")
    if not baseline_losses:
        lines.append("_Baseline losses not available._\n")
    else:
        for exp_name in ["exp1","exp2","exp3","exp4","exp5","exp6",
                         "exp7","exp8","exp9","exp10","exp11","exp12","exp13"]:
            model_losses = get_losses(exp_name)
            if not model_losses:
                continue
            u_stat, p_val = stats.mannwhitneyu(baseline_losses, model_losses, alternative="two-sided")
            r = u_stat / (len(baseline_losses) * len(model_losses))
            sig = "**YES**" if p_val < 0.05 else "no"
            direction = "↓ better" if np.mean(model_losses) < np.mean(baseline_losses) else "↑ worse"
            label = MODEL_LABELS.get(exp_name, exp_name).replace("\n", " ")
            lines.append(f"| {label} | {direction} | {u_stat:.0f} | {p_val:.4f} | {sig} | {r:.3f} |")

    # ── Block 1 假设检验 ─────────────────────────────────────────────────────
    lines.append("\n---\n## Block 1: Gate & Score Effect\n")

    rand_l      = get_losses("exp1")   # Random-500
    resolved_l  = get_losses("exp5")   # ResolvedOnly-500
    topq_l      = get_losses("exp3")   # TopQ-500
    bottomq_l   = get_losses("exp7")   # BottomQ-500

    lines.append("### H1: Gate 有效性 — ResolvedOnly-500 < Random-500 (Gold set)\n")
    lines.append(mwu_less(resolved_l, rand_l, "ResolvedOnly-500", "Random-500"))

    lines.append("\n### H2: 评分有效性 — TopQ-500 < ResolvedOnly-500 (Gold set)\n")
    lines.append(mwu_less(topq_l, resolved_l, "TopQ-500", "ResolvedOnly-500"))

    lines.append("\n### H4: Sanity Check — TopQ-500 < BottomQ-500 (Gold set)\n")
    lines.append(mwu_less(topq_l, bottomq_l, "TopQ-500", "BottomQ-500"))

    # H3: Scaling（仅报告均值，不做检验）
    lines.append("\n### H3: Scaling — 500 → 1000 均值对比\n")
    lines.append("| Strategy | 500 mean loss | 1000 mean loss | Δ |")
    lines.append("|----------|--------------|----------------|---|")
    for name, e500, e1000 in [
        ("Random",      "exp1", "exp2"),
        ("TopQ",        "exp3", "exp4"),
        ("ResolvedOnly","exp5", "exp6"),
    ]:
        l500  = get_losses(e500)
        l1000 = get_losses(e1000)
        if l500 and l1000:
            m500, m1000 = np.mean(l500), np.mean(l1000)
            lines.append(f"| {name} | {m500:.4f} | {m1000:.4f} | {m1000-m500:+.4f} |")

    # ── Block 2 假设检验 ─────────────────────────────────────────────────────
    lines.append("\n---\n## Block 2: Efficiency vs Style\n")
    noeff_l   = get_losses("exp8")   # Ablation-NoEfficiency
    nostyle_l = get_losses("exp9")   # Ablation-NoStyle

    lines.append("### H5: Efficiency 贡献 > Style 贡献 — NoEfficiency > NoStyle on Gold set\n")
    if noeff_l and nostyle_l:
        u, p = stats.mannwhitneyu(noeff_l, nostyle_l, alternative="greater")
        supported = "**Supported** ✓" if p < 0.05 else "Not supported ✗"
        lines.append(
            f"- Mann-Whitney U={u:.0f}, p={p:.4f} → {supported}\n"
            f"- Mean loss: NoEfficiency={np.mean(noeff_l):.4f}, NoStyle={np.mean(nostyle_l):.4f}"
        )
    else:
        lines.append("- 数据不足，跳过检验")

    # ── Block 3 假设检验 ─────────────────────────────────────────────────────
    lines.append("\n---\n## Block 3: Sub-dimension Ablation\n")
    nob2_l  = get_losses("exp10")   # Ablation-NoB2
    nob3_l  = get_losses("exp11")   # Ablation-NoB3
    noc2_l  = get_losses("exp12")   # Ablation-NoC2
    noc3_l  = get_losses("exp13")   # Ablation-NoC3

    lines.append("### H6: B2(error_retry) 贡献 > B3(step_count) — NoB2 > NoB3 on Gold set\n")
    if nob2_l and nob3_l:
        u, p = stats.mannwhitneyu(nob2_l, nob3_l, alternative="greater")
        supported = "**Supported** ✓" if p < 0.05 else "Not supported ✗"
        lines.append(
            f"- Mann-Whitney U={u:.0f}, p={p:.4f} → {supported}\n"
            f"- Mean loss: NoB2={np.mean(nob2_l):.4f}, NoB3={np.mean(nob3_l):.4f}"
        )
    else:
        lines.append("- 数据不足，跳过检验")

    lines.append("\n### H7: C3(obs_utilization) 贡献 > C2(action_diversity) — NoC3 > NoC2 on Gold set\n")
    if noc3_l and noc2_l:
        u, p = stats.mannwhitneyu(noc3_l, noc2_l, alternative="greater")
        supported = "**Supported** ✓" if p < 0.05 else "Not supported ✗"
        lines.append(
            f"- Mann-Whitney U={u:.0f}, p={p:.4f} → {supported}\n"
            f"- Mean loss: NoC3={np.mean(noc3_l):.4f}, NoC2={np.mean(noc2_l):.4f}"
        )
    else:
        lines.append("- 数据不足，跳过检验")

    # ── 子维度 impact 排名 ────────────────────────────────────────────────────
    lines.append("\n### 子维度 Impact 排名（Δ vs TopQ-500，gold set）\n")
    lines.append("| Ablation | Mean Loss | Δ vs TopQ | Rank |")
    lines.append("|----------|-----------|-----------|------|")
    topq_mean = np.mean(topq_l) if topq_l else np.nan
    impacts = []
    for exp_name, dim_name in [
        ("exp8",  "No Efficiency"),
        ("exp9",  "No Style"),
        ("exp10", "No B2 (error_retry)"),
        ("exp11", "No B3 (step_count)"),
        ("exp12", "No C2 (action_div)"),
        ("exp13", "No C3 (obs_util)"),
    ]:
        l = get_losses(exp_name)
        if l:
            m = np.mean(l)
            impacts.append((dim_name, m, m - topq_mean))
    impacts.sort(key=lambda x: x[2], reverse=True)
    for rank, (dim_name, m, delta) in enumerate(impacts, 1):
        lines.append(f"| {dim_name} | {m:.4f} | {delta:+.4f} | #{rank} |")

    if impacts:
        lines.append(f"\n**最关键维度**: 移除 **{impacts[0][0]}** 导致 loss 增幅最大。")

    # ── H8: 测试集质量梯度验证 ───────────────────────────────────────────────
    lines.append("\n---\n## H8: 测试集质量梯度（所有模型应满足 gold < random < low_q）\n")
    lines.append("| Model | Gold | Random | Low-Q | H8 Satisfied |")
    lines.append("|-------|------|--------|-------|--------------|")
    for exp_name in ["baseline","exp1","exp3","exp5","exp7"]:
        g = raw_results.get(exp_name, {}).get("gold",   {}).get("mean_loss")
        r = raw_results.get(exp_name, {}).get("random", {}).get("mean_loss")
        l = raw_results.get(exp_name, {}).get("low_q",  {}).get("mean_loss")
        if g is None or r is None or l is None:
            continue
        satisfied = "✓" if g < r < l else "✗"
        label = MODEL_LABELS.get(exp_name, exp_name).replace("\n", " ")
        lines.append(f"| {label} | {g:.4f} | {r:.4f} | {l:.4f} | {satisfied} |")

    return "\n".join(lines)


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="统计分析与可视化")
    parser.add_argument("--results", type=str,
                        default=str(RESULTS_DIR / "perplexity_results.json"),
                        help="perplexity_results.json路径")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        logger.error("结果文件不存在: %s\n请先运行 eval_perplexity.py", results_path)
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("加载评估结果: %s", results_path)
    raw_results, df = load_results(results_path)

    logger.info("数据概览:\n%s", df[["model", "split", "mean_loss", "perplexity", "n_valid"]].to_string())

    # 生成图表
    plot_loss_comparison(df,        FIGURES_DIR / "fig1_loss_comparison.png")
    plot_ablation_heatmap(df,       FIGURES_DIR / "fig2_ablation_heatmap.png")
    plot_hypothesis_comparison(df,  FIGURES_DIR / "fig3_hypothesis.png")
    plot_loss_distribution(raw_results, FIGURES_DIR / "fig4_loss_distribution.png")

    # 统计检验
    logger.info("运行统计显著性检验...")
    report = run_statistical_tests(raw_results)
    report_path = RESULTS_DIR / "stats_report.md"
    with open(report_path, "w") as f:
        f.write(report)
    logger.info("统计报告已保存: %s", report_path)
    print("\n" + report)

    # 保存汇总CSV
    summary_csv = RESULTS_DIR / "summary_table.csv"
    df.drop(columns=["losses"], errors="ignore").to_csv(summary_csv, index=False)
    logger.info("汇总CSV已保存: %s", summary_csv)

    logger.info("\n所有分析完成!")
    logger.info("图表目录: %s", FIGURES_DIR)
    logger.info("统计报告: %s", report_path)


if __name__ == "__main__":
    main()
