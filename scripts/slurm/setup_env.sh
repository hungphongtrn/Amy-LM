#!/bin/bash
# Amy-LM Environment Setup
#
# Run once on the fresh machine before submitting training jobs:
#   HF_TOKEN="..." bash scripts/slurm/setup_env.sh
#
# Sets up: system deps, uv, Python env, HF login, downloads checkpoints + data.

set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$HOME/Amy-LM}"
BRANCH="${BRANCH:-exp/amylm-facodec}"
HF_TOKEN="${HF_TOKEN:-}"
MARKER="$PROJECT_DIR/.setup_done"

if [ -f "$MARKER" ]; then
    echo "Setup already completed at $(cat "$MARKER"). Delete $MARKER to re-run."
    exit 0
fi

echo "=== Amy-LM Setup: $(date) ==="
echo "Project: $PROJECT_DIR"
echo "Branch:  $BRANCH"

# ── System deps ──
echo "--- System dependencies ---"
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq
    sudo apt-get install -y -qq git cmake ffmpeg libsndfile1 curl 2>&1 | tail -3
fi

# ── Clone repo ──
if [ ! -d "$PROJECT_DIR" ]; then
    echo "--- Cloning repo ---"
    git clone https://github.com/hungphongtrn/Amy-LM.git "$PROJECT_DIR"
fi
cd "$PROJECT_DIR"
git fetch origin
git checkout "$BRANCH"
git pull origin "$BRANCH" 2>/dev/null || true
echo "Branch: $(git branch --show-current) @ $(git rev-parse --short HEAD)"

# ── uv + Python deps ──
if ! command -v uv &>/dev/null; then
    echo "--- Installing uv ---"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "--- Syncing Python deps ---"
uv sync 2>&1 | tail -5
echo "uv:    $(uv --version)"
echo "python: $(uv run python --version)"

# ── HF login ──
if [ -n "$HF_TOKEN" ]; then
    echo "--- HF login ---"
    uv run python -c "from huggingface_hub import login; login(token='$HF_TOKEN')" 2>&1 | tail -1
    HF_USER=$(uv run python -c 'from huggingface_hub import whoami; print(whoami()["name"])')
    echo "Authenticated as: $HF_USER"
fi

# ── FACodec checkpoints ──
echo "--- FACodec checkpoints (398MB) ---"
mkdir -p checkpoints/facodec
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('hungphongtrn/facodec-checkpoints', local_dir='checkpoints/facodec', repo_type='model')
print('Done.')
" 2>&1 | tail -3
ls -lh checkpoints/facodec/ns3_facodec_*.bin

# ── MUStARD data ──
echo "--- MUStARD preprocessed (304MB) ---"
mkdir -p data/processed/mustard-processed
uv run python -c "
from huggingface_hub import snapshot_download
snapshot_download('hungphongtrn/mustard-facodec', local_dir='data/processed/mustard-processed', repo_type='dataset')
print('Done.')
" 2>&1 | tail -3
ls -lh data/processed/mustard-processed/train.parquet

# ── Mark done ──
date > "$MARKER"
echo ""
echo "=== Setup complete: $(date) ==="
echo "Ready to submit: sbatch scripts/slurm/train_27_dpo.slurm"
