#!/usr/bin/env python3
"""FACodec-encode adversarial preference pairs and push to HF Hub.

Loads adversarial JSONL pairs, joins with NVTTS audio by id, runs FACodec
encoding, splits into train/dev/test, and pushes to HuggingFace Hub.

Usage:
    # Full encode + push (GPU):
    uv run python scripts/encode_adversarial_pairs.py \\
        --pairs data/nvtts_adversarial/pairs.jsonl \\
        --nvtts data/nvtts_enriched/nvtts_enriched.parquet \\
        --output hungphongtrn/nvtts_facodec_adversarial

    # Dry run with mock encoder (CPU, no push):
    uv run python scripts/encode_adversarial_pairs.py \\
        --pairs data/nvtts_adversarial/pairs.jsonl \\
        --nvtts data/nvtts_enriched/nvtts_enriched.parquet \\
        --output ./local_adversarial \\
        --mock --local

    # Custom split sizes:
    uv run python scripts/encode_adversarial_pairs.py \\
        --pairs data/nvtts_adversarial/pairs.jsonl \\
        --nvtts data/nvtts_enriched/nvtts_enriched.parquet \\
        --train-size 800 --dev-size 100 --test-size 100
"""

from __future__ import annotations

import argparse
import os
import sys

from datasets import Dataset

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_PROJECT_ROOT, "src")
for _path in (_PROJECT_ROOT, _SRC):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.preprocessing.facodec_encoder import FACodecEncoder
from src.preprocessing.preference_dataset_processor import PreferenceDatasetProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="FACodec-encode adversarial pairs and push to HF Hub"
    )
    parser.add_argument(
        "--pairs", required=True,
        help="Path to adversarial JSONL file from generate_adversarial_pairs.py",
    )
    parser.add_argument(
        "--nvtts", required=True,
        help="Path to NVTTS enriched parquet (audio + metadata source)",
    )
    parser.add_argument(
        "--output", required=True,
        help="HF Hub repo ID (e.g. hungphongtrn/nvtts_facodec_adversarial) or local dir",
    )
    parser.add_argument(
        "--train-size", type=int, default=800,
        help="Number of training samples (default: 800)",
    )
    parser.add_argument(
        "--dev-size", type=int, default=100,
        help="Number of dev samples (default: 100)",
    )
    parser.add_argument(
        "--test-size", type=int, default=100,
        help="Number of test samples (default: 100)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="FACodec encoding batch size (default: 8)",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device for FACodec encoder (default: cuda)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Shuffle seed for reproducible splits (default: 42)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use mock FACodec encoder (for testing without checkpoints)",
    )
    parser.add_argument(
        "--local", action="store_true",
        help="Save locally instead of pushing to HF Hub",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.pairs):
        print(f"Error: pairs JSONL not found at {args.pairs}")
        sys.exit(1)
    if not os.path.exists(args.nvtts):
        print(f"Error: NVTTS parquet not found at {args.nvtts}")
        sys.exit(1)

    nvtts_ds = Dataset.from_parquet(args.nvtts)
    print(f"Loaded {len(nvtts_ds)} NVTTS samples from {args.nvtts}")

    encoder = FACodecEncoder(device=args.device, force_mock=args.mock)
    processor = PreferenceDatasetProcessor(encoder, batch_size=args.batch_size)

    print(f"Processing adversarial pairs from {args.pairs} ...")
    result = processor.process_adversarial_dataset(args.pairs, nvtts_ds)
    print(f"Encoded {len(result)} adversarial samples")

    if len(result) == 0:
        print("Error: no matching pairs found. Check that JSONL ids match NVTTS ids.")
        sys.exit(1)

    split_threshold = args.train_size + args.dev_size
    available = min(len(result), args.train_size + args.dev_size + args.test_size)

    result = result.shuffle(seed=args.seed)

    train_end = min(args.train_size, available)
    dev_end = min(args.train_size + args.dev_size, available)
    test_end = available

    train_ds = result.select(range(0, train_end))
    dev_ds = result.select(range(train_end, dev_end)) if dev_end > train_end else None
    test_ds = result.select(range(dev_end, test_end)) if test_end > dev_end else None

    print(f"\nSplit: train={len(train_ds)}, dev={len(dev_ds) if dev_ds else 0}, "
          f"test={len(test_ds) if test_ds else 0}")

    if args.local:
        output_dir = args.output
        os.makedirs(output_dir, exist_ok=True)
        train_ds.to_parquet(os.path.join(output_dir, "train.parquet"))
        if dev_ds is not None and len(dev_ds) > 0:
            dev_ds.to_parquet(os.path.join(output_dir, "dev.parquet"))
        if test_ds is not None and len(test_ds) > 0:
            test_ds.to_parquet(os.path.join(output_dir, "test.parquet"))
        print(f"Saved to {output_dir}")
    else:
        print(f"Pushing to {args.output} ...")
        train_ds.push_to_hub(args.output, split="train", private=False)
        if dev_ds is not None and len(dev_ds) > 0:
            dev_ds.push_to_hub(args.output, split="dev", private=False)
        if test_ds is not None and len(test_ds) > 0:
            test_ds.push_to_hub(args.output, split="test", private=False)
        print(f"Pushed to {args.output}")


if __name__ == "__main__":
    main()
