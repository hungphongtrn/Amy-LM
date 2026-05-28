#!/bin/bash
#SBATCH --job-name=amy-dpo-27
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/dpo_27_%j.log
#SBATCH --error=logs/dpo_27_%j.err
#
# Issue #27 — Full DPO Training on NVTTS FACodec dataset
#
# Submit:
#   sbatch scripts/slurm/train_27_dpo.slurm
#
# Prerequisites: run scripts/slurm/setup_env.sh first (or set HF_TOKEN here).
# The script auto-runs setup if .setup_done marker is missing.

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/Amy-LM}"
DPO_CONFIG="${DPO_CONFIG:-configs/dpo/base.yaml}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB="${WANDB:-0}"

cd "$PROJECT_DIR"

# Auto-setup if not done
if [ ! -f "$PROJECT_DIR/.setup_done" ]; then
    echo "--- Running setup_env.sh (first time) ---"
    bash scripts/slurm/setup_env.sh
fi

echo "=== Issue #27: Full DPO Training ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Config: $DPO_CONFIG"
echo "Start: $(date)"

WANDB_FLAG="--no-wandb"
if [ "$WANDB" = "1" ]; then
    WANDB_FLAG="--wandb-project amy-lm-dpo --wandb-run-name dpo-$(date +%Y%m%d_%H%M%S)"
fi

uv run python scripts/train_amy_dpo.py \
    --config "$DPO_CONFIG" \
    $WANDB_FLAG

echo "#27 DPO training completed at $(date)"
