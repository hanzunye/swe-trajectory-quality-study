#!/usr/bin/env bash
# run_pipeline.sh — 从 exp4 开始完整评估流水线
# 用法: bash run_pipeline.sh [--force]
#
# 步骤:
#   1. eval_perplexity.py  -- exp4..exp13 (全部 splits)
#   2. eval_next_action.py -- baseline exp1 exp3 exp5 exp7, split=gold
#   3. analyze_results.py  -- 可视化 & 统计报告

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
WORKSPACE="${WORKSPACE:-${REPO_ROOT}}"
RESULTS_DIR="${WORKSPACE}/data/eval_results"
LOG_DIR="${RESULTS_DIR}/logs"
export WORKSPACE

# ── 颜色输出 ──────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

FORCE_FLAG=""
if [[ "${1:-}" == "--force" ]]; then
    FORCE_FLAG="--force"
    warn "已启用 --force，将覆盖已有结果"
fi

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# ── 工具函数：带日志和错误检测的 python 调用 ──────────────────────────────
run_step() {
    local step_name="$1"; shift
    local log_file="$LOG_DIR/${TIMESTAMP}_${step_name}.log"
    info "=========================================="
    info "开始: $step_name"
    info "命令: python3 $*"
    info "日志: $log_file"
    info "=========================================="

    # tee: 同时输出到终端和日志文件
    if python3 "$@" 2>&1 | tee "$log_file"; then
        info "完成: $step_name"
    else
        error "$step_name 失败，退出码 $?"
        error "查看日志: $log_file"
        exit 1
    fi
}

# ── Step 1: Perplexity 评估（exp4 ~ exp13）────────────────────────────────
PERP_MODELS="exp4 exp5 exp6 exp7 exp8 exp9 exp10 exp11 exp12 exp13"

run_step "perplexity" "${SCRIPT_DIR}/eval_perplexity.py" \
    --models $PERP_MODELS \
    $FORCE_FLAG

# ── Step 2: Next-Action Prediction（指定模型 & gold split）────────────────
run_step "next_action" "${SCRIPT_DIR}/eval_next_action.py" \
    --models baseline exp1 exp3 exp5 exp7 \
    --splits gold

# ── Step 3: 可视化 & 统计分析 ─────────────────────────────────────────────
run_step "analyze" "${SCRIPT_DIR}/analyze_results.py"

# ── 完成 ──────────────────────────────────────────────────────────────────
info "=========================================="
info "全部流水线执行完毕！"
info "结果目录: ${RESULTS_DIR}/"
info "  figures/       — 图表 (fig1~fig4)"
info "  stats_report.md — 统计报告"
info "  logs/          — 各步骤日志"
info "=========================================="
