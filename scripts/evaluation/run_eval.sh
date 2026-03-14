#!/bin/bash
# ============================================================
# A100 评估流水线 — 完整执行脚本
# plan.md v3 实验设计（13组 + 1 baseline）
#
# 预估总耗时: ~10-14小时 (单张A100 80GB)
#   Step 1 (构建测试集):   1-2h  (CPU, HF数据流式下载)
#   Step 2 (Perplexity):   6-8h  (GPU, 14模型×3测试集)
#   Step 3 (Next-Action):  1-2h  (GPU, 5核心模型×100样本)
#   Step 4 (分析):         <5min (CPU)
#
# 用法:
#   bash run_eval.sh                    # 运行完整流水线
#   bash run_eval.sh --step 2           # 只运行Step 2
#   bash run_eval.sh --skip-step1       # 跳过测试集构建（已存在时）
#   bash run_eval.sh --models baseline exp1 exp3 exp5 exp7
# ============================================================

set -e

# ── 颜色 ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
log_info()  { echo -e "${GREEN}[INFO]${NC}  $1"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }
log_step()  { echo -e "\n${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; \
              echo -e "${CYAN}[STEP $1]${NC} $2"; \
              echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"; }

# ── 默认配置 ─────────────────────────────────────────────────────────────
WORKSPACE="${WORKSPACE:-/workspace}"
EVAL_SCRIPT_DIR="${EVAL_SCRIPT_DIR:-$WORKSPACE/eval}"
ONLY_STEP=""
SKIP_STEP1=false
MODELS_OVERRIDE=""
SCORES_REPO=""   # HF质量得分repo（可选）
SCORES_FILE=""   # 本地质量得分文件（可选）

# ── 参数解析 ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)           ONLY_STEP="$2"; shift 2;;
        --skip-step1)     SKIP_STEP1=true; shift;;
        --models)         shift; MODELS_OVERRIDE="$@"; break;;
        --scores-repo)    SCORES_REPO="$2"; shift 2;;
        --scores-file)    SCORES_FILE="$2"; shift 2;;
        -h|--help)
            echo "Usage: bash run_eval.sh [--step N] [--skip-step1] [--models m1 m2 ...]"
            exit 0;;
        *) log_warn "未知参数: $1"; shift;;
    esac
done

# ── 环境检查 ─────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  A100 评估流水线  |  plan.md § 方案A: Proxy Metrics${NC}"
echo -e "${CYAN}============================================================${NC}"
echo ""

log_info "检查GPU环境..."
if ! nvidia-smi &>/dev/null; then
    log_error "未检测到NVIDIA GPU，请确认运行在GPU实例上"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

log_info "检查Python依赖..."
python3 -c "
import torch, transformers, peft, datasets
print(f'  PyTorch       : {torch.__version__}')
print(f'  Transformers  : {transformers.__version__}')
print(f'  PEFT          : {peft.__version__}')
print(f'  CUDA BF16     : {torch.cuda.is_bf16_supported()}')
if torch.cuda.is_available():
    mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f'  GPU Memory    : {mem:.1f} GB')
"

# 安装可能缺失的依赖
pip install --quiet scipy matplotlib seaborn pandas rouge-score nltk 2>/dev/null || true

# 下载NLTK数据（用于BLEU）
python3 -c "import nltk; nltk.download('punkt', quiet=True)" 2>/dev/null || true

log_info "工作目录: $WORKSPACE"
log_info "评估脚本目录: $EVAL_SCRIPT_DIR"
log_info "开始时间: $(date)"

# ── 确保脚本在eval目录下 ─────────────────────────────────────────────────
mkdir -p "$EVAL_SCRIPT_DIR"
# 将脚本复制到workspace（如果从本地上传）
SCRIPT_SOURCE="$(dirname "$0")"
for f in build_test_set.py eval_perplexity.py eval_next_action.py analyze_results.py; do
    if [ -f "$SCRIPT_SOURCE/$f" ] && [ ! -f "$EVAL_SCRIPT_DIR/$f" ]; then
        cp "$SCRIPT_SOURCE/$f" "$EVAL_SCRIPT_DIR/$f"
        log_info "已复制 $f → $EVAL_SCRIPT_DIR/"
    fi
done

cd "$EVAL_SCRIPT_DIR"

# ── 模型配置 ─────────────────────────────────────────────────────────────
# Step 2使用的模型列表（默认全部14个，含 baseline）
# exp1=Random-500   exp2=Random-1000
# exp3=TopQ-500     exp4=TopQ-1000
# exp5=ResolvedOnly-500  exp6=ResolvedOnly-1000
# exp7=BottomQ-500
# exp8=Ablation-NoEfficiency-500  exp9=Ablation-NoStyle-500
# exp10=Ablation-NoB2-500  exp11=Ablation-NoB3-500
# exp12=Ablation-NoC2-500  exp13=Ablation-NoC3-500
if [ -n "$MODELS_OVERRIDE" ]; then
    PERPLEXITY_MODELS="$MODELS_OVERRIDE"
else
    PERPLEXITY_MODELS="baseline exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9 exp10 exp11 exp12 exp13"
fi

# Step 3只评估核心模型（节省时间）：
#   baseline / Random-500 / TopQ-500 / ResolvedOnly-500 / BottomQ-500
NEXT_ACTION_MODELS="baseline exp1 exp3 exp5 exp7"


# ════════════════════════════════════════════════════════════
# STEP 1: 构建测试集
# ════════════════════════════════════════════════════════════
should_run_step() { [ -z "$ONLY_STEP" ] || [ "$ONLY_STEP" = "$1" ]; }

if should_run_step 1; then
    log_step "1/4" "构建测试集 (Gold / Random / Low-Q)"

    if [ "$SKIP_STEP1" = true ] && [ -f "$WORKSPACE/eval_data/test_set_ids.json" ]; then
        log_info "测试集已存在，跳过Step 1"
    else
        STEP1_ARGS=""
        [ -n "$SCORES_REPO" ]  && STEP1_ARGS="$STEP1_ARGS --scores-repo $SCORES_REPO"
        [ -n "$SCORES_FILE" ]  && STEP1_ARGS="$STEP1_ARGS --scores-file $SCORES_FILE"

        log_info "运行 build_test_set.py ..."
        python3 build_test_set.py $STEP1_ARGS

        # 验证输出
        for split in gold random low_q; do
            count=$(wc -l < "$WORKSPACE/eval_data/${split}.jsonl" 2>/dev/null || echo 0)
            log_info "  [${split}] ${count} 条轨迹"
            if [ "$count" -lt 100 ]; then
                log_warn "  [${split}] 轨迹数量不足100，请检查数据源"
            fi
        done

        log_info "Step 1 完成 ✓"
    fi
fi


# ════════════════════════════════════════════════════════════
# STEP 2: Perplexity 评估（核心指标①）
# ════════════════════════════════════════════════════════════
if should_run_step 2; then
    log_step "2/4" "Perplexity 评估 (14模型 × 3测试集)"
    log_info "模型列表: $PERPLEXITY_MODELS"
    log_warn "预计耗时: 4-6小时，建议在tmux/screen中运行"

    # 开始前清空GPU缓存
    python3 -c "import torch; torch.cuda.empty_cache(); print('GPU缓存已清空')"

    START_TIME=$(date +%s)
    python3 eval_perplexity.py \
        --models $PERPLEXITY_MODELS \
        --splits gold random low_q

    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    log_info "Step 2 完成 ✓  耗时: $((ELAPSED/3600))h $((ELAPSED%3600/60))m"

    # 快速预览结果
    if [ -f "$WORKSPACE/eval_results/perplexity_results.csv" ]; then
        log_info "Perplexity结果预览:"
        python3 -c "
import pandas as pd
df = pd.read_csv('$WORKSPACE/eval_results/perplexity_results.csv')
pivot = df.pivot_table(values='mean_loss', index='model', columns='split')
print(pivot.round(4).to_string())
"
    fi
fi


# ════════════════════════════════════════════════════════════
# STEP 3: Next-Action Prediction（指标②）
# ════════════════════════════════════════════════════════════
if should_run_step 3; then
    log_step "3/4" "Next-Action Prediction Accuracy（5核心模型）"
    log_info "评估模型: $NEXT_ACTION_MODELS"
    log_warn "预计耗时: 1-2小时"

    python3 eval_next_action.py \
        --models $NEXT_ACTION_MODELS \
        --splits gold \
        --truncate-steps 3 5 7 \
        --num-samples 100

    log_info "Step 3 完成 ✓"
fi


# ════════════════════════════════════════════════════════════
# STEP 4: 统计分析与可视化
# ════════════════════════════════════════════════════════════
if should_run_step 4; then
    log_step "4/4" "统计分析与可视化"

    if [ ! -f "$WORKSPACE/eval_results/perplexity_results.json" ]; then
        log_error "找不到 perplexity_results.json，请先完成Step 2"
        exit 1
    fi

    python3 analyze_results.py \
        --results "$WORKSPACE/eval_results/perplexity_results.json"

    log_info "Step 4 完成 ✓"

    # 列出所有输出文件
    log_info "生成的文件:"
    ls -lh "$WORKSPACE/eval_results/figures/" 2>/dev/null || true
    ls -lh "$WORKSPACE/eval_results/"*.{json,csv,md} 2>/dev/null || true
fi


# ── 总结 ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  评估流水线完成!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
echo ""
log_info "结果目录: $WORKSPACE/eval_results/"
log_info "  perplexity_results.json  — 完整数据（含per-trajectory losses）"
log_info "  perplexity_results.csv   — 汇总表格"
log_info "  stats_report.md          — 统计检验报告"
log_info "  figures/                 — 图表目录"
log_info ""
log_info "完成时间: $(date)"
