#!/bin/bash
#SBATCH --job-name=amy-faco-29
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/amy_29_%j.log
#SBATCH --error=logs/amy_29_%j.err
#
# Issue #29 — Amy Classifier (Prosody+Timbre) saturation on MUStARD
#
# Submit:
#   sbatch scripts/slurm/train_29_amy.slurm

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/Amy-LM}"
HF_TOKEN="${HF_TOKEN:-}"
WANDB="${WANDB:-0}"

cd "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/.setup_done" ]; then
    echo "--- Running setup_env.sh (first time) ---"
    bash scripts/slurm/setup_env.sh
fi

echo "=== Issue #29: Amy Classifier (Prosody+Timbre) Saturation ==="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

DATA="data/processed/mustard-processed/train.parquet"

WANDB_FLAG=""
if [ "$WANDB" = "1" ]; then
    WANDB_FLAG="--wandb --wandb-project amy-lm-pilot"
fi

uv run python scripts/train_amy_classifier.py \
    --data-path "$DATA" \
    --mode amy \
    --device cuda \
    --seed 42 \
    --facodec-checkpoint checkpoints/facodec/ns3_facodec_decoder.bin \
    --checkpoint-dir checkpoints/training_amy \
    --output-dir outputs/training_amy \
    $WANDB_FLAG

echo "#29 Amy classifier completed at $(date)"
