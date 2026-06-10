#!/usr/bin/env python3
"""Issue #33 — LoRA Amy classifier experiment on MUStARD (H2 gradient-flow test).

LoRA adapters on audio_adapter + language_model projection layers.
FACodec modules + classifier fully trainable via modules_to_save.

Usage:
    python scripts/train_33_lora_amy.py                                     # LoRA + aligned FACodec (no W&B)
    python scripts/train_33_lora_amy.py --wandb                             # with W&B logging
    python scripts/train_33_lora_amy.py --facodec-control shuffled --wandb  # LoRA-only isolation (shuffled FACodec)

Prerequisites:
    python scripts/setup.py        # one-time: download checkpoints + MUStARD data

Output:
    checkpoints/training_lora_amy/best_model.pt          (aligned)
    checkpoints/training_lora_amy_shuffled/best_model.pt (shuffled)
    Outputs include lambda_p, lambda_t, LoRA config, facodec_control
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
    p = argparse.ArgumentParser(description="Issue #33 — LoRA Amy Classifier (H2 Gradient-Flow Test)")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--data-path", default=None, help="Override path to train.parquet")
    p.add_argument("--facodec-checkpoint", default=None, help="Override path to decoder checkpoint")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--patience", type=int, default=0)
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--facodec-control",
        type=str,
        choices=["aligned", "shuffled"],
        default="aligned",
        help="FACodec feature alignment (default: aligned). Use 'shuffled' to isolate LoRA-only gains from FACodec+LoRA.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Issue #33: LoRA Amy Classifier (H2 Gradient-Flow Test) ===\n")

    decoder, parquet = ensure_assets()
    data_path = args.data_path or str(parquet)
    facodec_ckpt = args.facodec_checkpoint or str(decoder)

    suffix = "_shuffled" if args.facodec_control == "shuffled" else ""
    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_amy_classifier.py"),
        "--data-path", data_path,
        "--mode", "amy",
        "--device", args.device,
        "--facodec-checkpoint", facodec_ckpt,
        "--facodec-control", args.facodec_control,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--batch-size", str(args.batch_size),
        "--grad-accum", str(args.grad_accum),
        "--seed", str(args.seed),
        "--checkpoint-dir", str(PROJECT_ROOT / "checkpoints" / f"training_lora_amy{suffix}"),
        "--output-dir", str(PROJECT_ROOT / "outputs" / f"training_lora_amy{suffix}"),
        "--patience", str(args.patience),
        "--checkpoint-metric", "val_f1",
        "--use-lora",
        "--lora-r", str(args.lora_r),
        "--lora-alpha", str(args.lora_alpha),
        "--lora-dropout", str(args.lora_dropout),
        "--wandb" if args.wandb else "",
    ]
    cmd = [a for a in cmd if a]

    print(f"FACodec control: {args.facodec_control} ({'LoRA-only isolation' if args.facodec_control == 'shuffled' else 'LoRA + FACodec signal'})")
    print(f"FACodec decoder: {facodec_ckpt}")
    print(f"Data: {data_path}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Running: {' '.join(cmd[1:])}\n")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
