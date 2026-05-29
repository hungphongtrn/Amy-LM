#!/usr/bin/env python3
"""Issue #28 — Baseline classifier (MOSS-Audio only) saturation on MUStARD.

Usage:
    python scripts/train_28_baseline.py            # defaults
    python scripts/train_28_baseline.py --wandb    # with W&B logging

Prerequisites:
    python scripts/setup.py        # one-time: download checkpoints + MUStARD data

Output:
    checkpoints/training_baseline/best_model.pt
    outputs/training_baseline/results.json
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_mustard_data() -> Path:
    local_dir = PROJECT_ROOT / "data" / "processed" / "mustard-processed"
    parquet = local_dir / "train.parquet"
    if not parquet.exists():
        print("Downloading MUStARD preprocessed data (304MB)...")
        local_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download("hungphongtrn/mustard-facodec", local_dir=str(local_dir),
                          repo_type="dataset")
    return parquet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #28 — Baseline Classifier Saturation")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--data-path", default=None, help="Override path to train.parquet")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Issue #28: Baseline Classifier Saturation ===\n")

    data_path = args.data_path or str(ensure_mustard_data())

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_amy_classifier.py"),
        "--data-path", data_path,
        "--mode", "baseline",
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--seed", str(args.seed),
        "--checkpoint-dir", str(PROJECT_ROOT / "checkpoints" / "training_baseline"),
        "--output-dir", str(PROJECT_ROOT / "outputs" / "training_baseline"),
        "--patience", "0",
        "--checkpoint-metric", "val_f1",
        "--wandb" if args.wandb else "",
    ]
    cmd = [a for a in cmd if a]

    print(f"Data: {data_path}")
    print(f"Running: {' '.join(cmd[1:])}\n")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
