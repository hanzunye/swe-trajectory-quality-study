#!/bin/bash
# ============================================================
# Run All Experiments Sequentially with Auto-Resume
#
# This script runs all 16 experiments in order, automatically
# resuming from checkpoints if a spot instance was interrupted.
#
#   Block A (exp1–exp13): core matrix (Random/TopQ/ResolvedOnly,
#                         BottomQ sanity check, group + dim ablations)
#   Block B (exp14–exp15): scale extension to 2000 (Random/TopQ)
#   Block C (exp16):       single-dimension ranking (B2Only-Top500)
#
# Usage:
#   bash run_experiments.sh              # Run all from exp1
#   bash run_experiments.sh exp3         # Start from exp3
#   bash run_experiments.sh exp7 exp13   # Run exp7 through exp13
#   bash run_experiments.sh exp14 exp16  # Run only the new Block B/C
#
# Optional env vars:
#   WORKSPACE    Override workspace root (default: /workspace on cloud)
#   SUBSET_DIR   Path to local Subset/output/ for prepare_data.py
#                (only needed for local runs; cloud loads IDs from HF)
# ============================================================
set -euo pipefail

WORKSPACE="${WORKSPACE:-/workspace}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_DIR="${WORKSPACE}/checkpoints"
OUTPUT_DIR="${WORKSPACE}/outputs"
STATUS_FILE="${WORKSPACE}/experiment_status.json"

# Optional: local Subset/output/ directory for trajectory IDs
SUBSET_DIR="${SUBSET_DIR:-}"

# Ordered experiment list
ALL_EXPERIMENTS=(exp1 exp2 exp3 exp4 exp5 exp6 exp7 exp8 exp9 exp10 exp11 exp12 exp13 exp14 exp15 exp16)

# Parse args: optional start and end experiment
START_EXP="${1:-exp1}"
END_EXP="${2:-exp16}"

# ── Helper functions ─────────────────────────────────────

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

is_experiment_done() {
    local exp="$1"
    local summary="${OUTPUT_DIR}/${exp}/training_summary.json"
    [ -f "$summary" ]
}

has_checkpoint() {
    local exp="$1"
    local ckpt_dir="${CHECKPOINT_DIR}/${exp}"
    [ -d "$ckpt_dir" ] && [ "$(ls -d "${ckpt_dir}"/checkpoint-* 2>/dev/null | wc -l)" -gt 0 ]
}

update_status() {
    local exp="$1"
    local status="$2"
    python3 -c "
import json, os
path = '${STATUS_FILE}'
data = {}
if os.path.exists(path):
    with open(path) as f: data = json.load(f)
data['$exp'] = {'status': '$status', 'timestamp': '$(date -Iseconds)'}
with open(path, 'w') as f: json.dump(data, f, indent=2)
"
}

# ── Main loop ────────────────────────────────────────────

log "=========================================="
log "  Experiment Runner — Spot-Safe"
log "  Start: ${START_EXP}  End: ${END_EXP}"
log "=========================================="

IN_RANGE=false

for exp in "${ALL_EXPERIMENTS[@]}"; do
    # Check if we've reached the start experiment
    if [ "$exp" = "$START_EXP" ]; then
        IN_RANGE=true
    fi

    if [ "$IN_RANGE" = false ]; then
        continue
    fi

    log "──────────────────────────────────────"
    log "Experiment: ${exp}"

    # Skip if already completed
    if is_experiment_done "$exp"; then
        log "  ✓ Already completed, skipping."
        update_status "$exp" "completed"
        # Check if we've passed the end
        if [ "$exp" = "$END_EXP" ]; then
            break
        fi
        continue
    fi

    # Determine if we should resume
    RESUME_FLAG=""
    if has_checkpoint "$exp"; then
        log "  ↻ Found checkpoint, will resume."
        RESUME_FLAG="--resume"
    fi

    # Prepare data if not ready (Arrow directory format)
    SUBSET=$(python3 -c "import sys; sys.path.insert(0, '${SCRIPT_DIR}'); from configs import get_experiment; print(get_experiment('$exp').subset)")
    DATA_DIR="${WORKSPACE}/data/${SUBSET}"
    if [ ! -d "$DATA_DIR" ] || [ ! -f "${DATA_DIR}/dataset_info.json" ]; then
        log "  Preparing data for subset: ${SUBSET}"
        python3 "${SCRIPT_DIR}/prepare_data.py" --subset "$SUBSET"
    else
        log "  Data already prepared: ${SUBSET}"
    fi

    # Run training
    update_status "$exp" "running"
    log "  Starting training..."

    if python3 "${SCRIPT_DIR}/train.py" --experiment "$exp" $RESUME_FLAG; then
        log "  ✓ ${exp} completed successfully!"
        update_status "$exp" "completed"
    else
        EXIT_CODE=$?
        if [ $EXIT_CODE -eq 130 ] || [ $EXIT_CODE -eq 137 ] || [ $EXIT_CODE -eq 143 ]; then
            log "  ⚠ ${exp} interrupted (signal). Checkpoint saved."
            update_status "$exp" "interrupted"
            log "  Re-run this script to resume from checkpoint."
            exit 0
        else
            log "  ✗ ${exp} failed with exit code ${EXIT_CODE}"
            update_status "$exp" "failed"
            exit $EXIT_CODE
        fi
    fi

    # Check if we've passed the end
    if [ "$exp" = "$END_EXP" ]; then
        break
    fi
done

log ""
log "=========================================="
log "  All experiments complete!"
log "  Results in: ${OUTPUT_DIR}"
log "=========================================="
