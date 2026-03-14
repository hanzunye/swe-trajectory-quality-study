#!/bin/bash
# ============================================================
# RunPod Environment Setup Script
# Base image: runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
# Target: QLoRA fine-tuning Qwen2.5-Coder-7B-Instruct on A100 80G
# ============================================================
set -euo pipefail

echo "============================================"
echo "  RunPod Training Environment Setup"
echo "============================================"

# ── Workspace paths ──────────────────────────────────────
# RunPod persistent volume is mounted at /workspace
WORKSPACE="/workspace"
CHECKPOINT_DIR="${WORKSPACE}/checkpoints"
DATA_DIR="${WORKSPACE}/data"
OUTPUT_DIR="${WORKSPACE}/outputs"

mkdir -p "${CHECKPOINT_DIR}" "${DATA_DIR}" "${OUTPUT_DIR}"

# ── System dependencies ──────────────────────────────────
echo "[1/4] Installing system dependencies..."
apt-get update -qq && apt-get install -y -qq \
    git-lfs \
    tmux \
    htop \
    nvtop \
    rsync \
    2>/dev/null

git lfs install --skip-smudge 2>/dev/null || true

# ── Python training dependencies ─────────────────────────
echo "[2/4] Installing Python training packages..."
pip install --quiet --upgrade pip

pip install --quiet \
    "transformers>=4.46.0,<5.0.0" \
    "datasets>=2.19.0" \
    "accelerate>=0.34.0" \
    "peft>=0.13.0" \
    "trl>=0.12.0" \
    "bitsandbytes>=0.44.0" \
    "flash-attn>=2.6.0" --no-build-isolation \
    "wandb>=0.18.0" \
    "huggingface-hub>=0.25.0" \
    "tiktoken>=0.7.0" \
    "orjson>=3.10.0" \
    "scipy>=1.14.0"

# ── Verify GPU & CUDA ───────────────────────────────────
echo "[3/4] Verifying GPU setup..."
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
    print(f'BF16 support: {torch.cuda.is_bf16_supported()}')
"

# ── Verify key packages ─────────────────────────────────
echo "[4/4] Verifying training packages..."
python3 -c "
from transformers import __version__ as tv; print(f'transformers: {tv}')
from peft import __version__ as pv; print(f'peft: {pv}')
from trl import __version__ as trlv; print(f'trl: {trlv}')
import bitsandbytes; print(f'bitsandbytes: OK')
try:
    import flash_attn; print(f'flash_attn: {flash_attn.__version__}')
except: print('flash_attn: not available (will use sdpa)')
import wandb; print(f'wandb: OK')
"

echo ""
echo "============================================"
echo "  Setup complete!"
echo "  Checkpoints:  ${CHECKPOINT_DIR}"
echo "  Data cache:   ${DATA_DIR}"
echo "============================================"
echo ""
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Next steps:"
echo "  1. Set HF_TOKEN:    export HF_TOKEN=your_token"
echo "  2. Set WANDB_KEY:   export WANDB_API_KEY=your_key  (optional)"
echo "  3. Prepare data:    python3 ${SCRIPT_DIR}/prepare_data.py --subset Random-500"
echo "  4. Run training:    python3 ${SCRIPT_DIR}/train.py --experiment exp1"
echo "  5. Run all exps:    bash ${SCRIPT_DIR}/run_experiments.sh"
