#!/bin/bash
#SBATCH --job-name=amy-lm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%A_%a.log
#SBATCH --error=logs/%x_%A_%a.err
#
# Amy-LM Training — Issues #27, #28, #29
#
# Submit with:
#   sbatch scripts/slurm/submit.slurm           # run all 3 sequentially
#   ISSUES=27 scripts/slurm/submit.slurm         # run #27 only (DPO)
#   ISSUES=28,29 scripts/slurm/submit.slurm      # run #28 + #29 (classifiers)
#
# Or submit 3 parallel jobs:
#   for i in 27 28 29; do sbatch --job-name=amy-lm-$i scripts/slurm/submit.slurm; done
#
# ── Hyperparameters (override via SBATCH export) ──

# Which issues to run (comma-separated: 27,28,29 or "all")
ISSUES="${ISSUES:-all}"

# HF token (set yours or pass via: export HF_TOKEN=...)
HF_TOKEN="${HF_TOKEN:-}"

# Repo branch
BRANCH="${BRANCH:-exp/amylm-facodec}"

# WandB (set to "1" to enable)
WANDB="${WANDB:-0}"

# DPO config (for #27)
DPO_CONFIG="${DPO_CONFIG:-configs/dpo/base.yaml}"

# ── Early exit if no GPU ──
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. This script requires CUDA."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
echo "GPU: ${GPU_NAME:-unknown}"

# ── Paths ──
PROJECT_DIR="${PROJECT_DIR:-$HOME/Amy-LM}"
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

set -euo pipefail
echo "=== Amy-LM Training Setup ==="
echo "Start: $(date)"
echo "Issues: $ISSUES"
echo "Project: $PROJECT_DIR"

# ── Step 1: System dependencies ──
echo "--- Installing system dependencies ---"
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq git cmake ffmpeg libsndfile1 curl 2>&1 | tail -3
fi

# ── Step 2: Clone repo ──
if [ ! -d "$PROJECT_DIR" ]; then
    echo "--- Cloning repo ---"
    git clone https://github.com/hungphongtrn/Amy-LM.git "$PROJECT_DIR"
else
    echo "--- Repo exists, fetching latest ---"
fi
cd "$PROJECT_DIR"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH" 2>/dev/null || true
echo "Branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

# ── Step 3: Install uv + Python deps ──
if ! command -v uv &>/dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "--- Syncing Python deps ---"
uv sync 2>&1 | tail -5
echo "uv version: $(uv --version)"
echo "python: $(uv run python --version)"

# ── Step 4: Login to HF ──
if [ -n "$HF_TOKEN" ]; then
    uv run python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>&1 | tail -1
    echo "HF authenticated as: $(uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])')"
else
    echo "WARNING: HF_TOKEN not set. Model/data download may fail."
fi

# ── Step 5: Download FACodec checkpoints ──
echo "--- Downloading FACodec checkpoints (398MB) ---"
mkdir -p checkpoints/facodec
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('hungphongtrn/facodec-checkpoints', local_dir='checkpoints/facodec', repo_type='model')
print('Done.')
" 2>&1 | tail -5
ls -lh checkpoints/facodec/ns3_facodec_*.bin

# ── Step 6: Run training ──

run_issue_27() {
    echo ""
    echo "=== Issue #27: Full DPO Training ==="
    echo "Config: $DPO_CONFIG"
    WANDB_FLAG="--no-wandb"
    if [ "$WANDB" = "1" ]; then
        WANDB_FLAG="--wandb-project amy-lm-dpo --wandb-run-name dpo-$(date +%Y%m%d_%H%M%S)"
    fi
    uv run python scripts/train_amy_dpo.py \
        --config "$DPO_CONFIG" \
        $WANDB_FLAG
    echo "#27 DPO training completed at $(date)"
}

run_issue_28() {
    echo ""
    echo "=== Issue #28: Baseline Classifier Saturation ==="
    DATA="$PROJECT_DIR/data/processed/mustard-processed/train.parquet"
    if [ ! -f "$DATA" ]; then
        echo "Downloading MUStARD preprocessed data (304MB)..."
        mkdir -p "$(dirname "$DATA")"
        uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('hungphongtrn/mustard-facodec', local_dir='$(dirname "$DATA")', repo_type='dataset')
print('Done.')
" 2>&1 | tail -3
    fi
    WANDB_FLAG=""
    if [ "$WANDB" = "1" ]; then
        WANDB_FLAG="--wandb --wandb-project amy-lm-pilot"
    fi
    uv run python scripts/train_amy_classifier.py \
        --data-path "$DATA" \
        --mode baseline \
        --device cuda \
        --seed 42 \
        --checkpoint-dir checkpoints/training_baseline \
        --output-dir outputs/training_baseline \
        $WANDB_FLAG
    echo "#28 Baseline classifier completed at $(date)"
}

run_issue_29() {
    echo ""
    echo "=== Issue #29: Amy Classifier (Prosody+Timbre) Saturation ==="
    DATA="$PROJECT_DIR/data/processed/mustard-processed/train.parquet"
    if [ ! -f "$DATA" ]; then
        echo "Downloading MUStARD preprocessed data (304MB)..."
        mkdir -p "$(dirname "$DATA")"
        uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('hungphongtrn/mustard-facodec', local_dir='$(dirname "$DATA")', repo_type='dataset')
print('Done.')
" 2>&1 | tail -3
    fi
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
}

if [ "$ISSUES" = "all" ]; then
    run_issue_27
    run_issue_28
    run_issue_29
else
    IFS=',' read -ra ISSUE_LIST <<< "$ISSUES"
    for issue in "${ISSUE_LIST[@]}"; do
        case "$issue" in
            27) run_issue_27 ;;
            28) run_issue_28 ;;
            29) run_issue_29 ;;
            *) echo "ERROR: Unknown issue $issue. Use 27, 28, 29, or 'all'." ;;
        esac
    done
fi

echo ""
echo "=== All training complete ==="
echo "End: $(date)"
echo "Outputs:"
ls -la output/ 2>/dev/null || echo "  (none)"
ls -la checkpoints/training*/ 2>/dev/null || echo "  (none)"
