#!/usr/bin/env python3
"""Issue #29 / #31 — Amy classifier (Prosody+Timbre) proof experiment on MUStARD.

Usage:
    python scripts/train_29_amy.py            # defaults (warm-started prosody, val_f1 checkpoint)
    python scripts/train_29_amy.py --wandb    # with W&B logging
    python scripts/train_29_amy.py --facodec-control shuffled  # negative control

Prerequisites:
    python scripts/setup.py        # one-time: download checkpoints + MUStARD data

Output:
    checkpoints/training_amy/best_model.pt
    outputs/training_amy/results.json  (includes lambda_p, lambda_t convergence)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ensure_assets() -> tuple[Path, Path]:
    facodec_dir = PROJECT_ROOT / "checkpoints" / "facodec"
    decoder = facodec_dir / "ns3_facodec_decoder.bin"
    if not decoder.exists():
        print("Downloading FACodec checkpoints (398MB)...")
        facodec_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download("hungphongtrn/facodec-checkpoints", local_dir=str(facodec_dir),
                          repo_type="model")

    mustard_dir = PROJECT_ROOT / "data" / "processed" / "mustard-processed"
    parquet = mustard_dir / "train.parquet"
    if not parquet.exists():
        print("Downloading MUStARD preprocessed data (304MB)...")
        mustard_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download("hungphongtrn/mustard-facodec", local_dir=str(mustard_dir),
                          repo_type="dataset")

    return decoder, parquet


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #29 / #31 — Amy Classifier Proof")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--data-path", default=None, help="Override path to train.parquet")
    p.add_argument("--facodec-checkpoint", default=None, help="Override path to decoder checkpoint")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument(
        "--facodec-control",
        type=str,
        choices=["aligned", "shuffled"],
        default="aligned",
        help="FACodec feature alignment: 'aligned' (standard) or 'shuffled' (negative control)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Issue #29 / #31: Amy Classifier (Prosody+Timbre) Proof ===\n")

    decoder, parquet = ensure_assets()
    data_path = args.data_path or str(parquet)
    facodec_ckpt = args.facodec_checkpoint or str(decoder)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_amy_classifier.py"),
        "--data-path", data_path,
        "--mode", "amy",
        "--device", args.device,
        "--facodec-checkpoint", facodec_ckpt,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--seed", str(args.seed),
        "--checkpoint-dir", str(PROJECT_ROOT / "checkpoints" / "training_amy"),
        "--output-dir", str(PROJECT_ROOT / "outputs" / "training_amy"),
        "--patience", str(args.patience),
        "--checkpoint-metric", "val_f1",
        "--facodec-control", args.facodec_control,
        "--wandb" if args.wandb else "",
    ]
    cmd = [a for a in cmd if a]

    print(f"FACodec decoder: {facodec_ckpt}")
    print(f"Data: {data_path}")
    print(f"FACodec control: {args.facodec_control}")
    print(f"Running: {' '.join(cmd[1:])}\n")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
