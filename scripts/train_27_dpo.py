#!/usr/bin/env python3
"""Issue #27 — Full DPO training on NVTTS FACodec dataset.

Usage:
    python scripts/train_27_dpo.py                              # defaults (base.yaml)
    python scripts/train_27_dpo.py --config configs/dpo/rtx3060_12gb.yaml
    python scripts/train_27_dpo.py --config configs/dpo/base.yaml --wandb

HF assets:
    hungphongtrn/nvtts_facodec           — DPO training data (~3600 samples)
    OpenMOSS-Team/MOSS-Audio-4B-Thinking — 4B backbone (loaded in bf16)

Output:
    output/amy_dpo/  — checkpoints + final model
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Issue #27 — DPO Training")
    p.add_argument("--config", default="configs/dpo/base.yaml",
                   help="YAML config (default: base.yaml for A100/H100)")
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument("--output-dir", default=None, help="Override output_dir")
    p.add_argument("--cosine-threshold", type=float, default=None)
    p.add_argument("--num-epochs", type=float, default=None)
    p.add_argument("--lora-r", type=int, default=None)
    p.add_argument("--optim", default=None)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print("=== Issue #27: Full DPO Training ===\n")

    cmd = [
        sys.executable, str(PROJECT_ROOT / "scripts" / "train_amy_dpo.py"),
        "--config", args.config,
        "--no-wandb" if not args.wandb else "",
    ]
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if args.cosine_threshold is not None:
        cmd.extend(["--cosine-threshold", str(args.cosine_threshold)])
    if args.num_epochs is not None:
        cmd.extend(["--num-epochs", str(args.num_epochs)])
    if args.lora_r is not None:
        cmd.extend(["--lora-r", str(args.lora_r)])
    if args.optim is not None:
        cmd.extend(["--optim", args.optim])
    cmd = [a for a in cmd if a]

    print(f"Running: {' '.join(cmd[1:])}\n")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


if __name__ == "__main__":
    raise SystemExit(main())
