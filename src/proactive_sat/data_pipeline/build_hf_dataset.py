#!/usr/bin/env python3
"""
Proactive-SAT HuggingFace Dataset Builder

Builds a 200-sample HuggingFace DatasetDict from enriched JSONL with
stratified sampling option for balanced prosody style distribution.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def stratified_sample(
    samples: list[dict[str, Any]],
    n: int,
    seed: int,
    stratify_by: str = "prosody_style",
) -> list[dict[str, Any]]:
    """
    Perform stratified sampling to ensure balanced distribution across strata.

    Args:
        samples: List of sample dictionaries to sample from
        n: Number of samples to select
        seed: Random seed for reproducibility
        stratify_by: Field to stratify by (default: prosody_style)

    Returns:
        List of sampled dictionaries

    Raises:
        ValueError: If stratify_by field is missing or has too few samples
    """
    import random

    random.seed(seed)

    # Group samples by strata
    strata: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        strata_value = sample.get(stratify_by, "unknown")
        if strata_value is None:
            strata_value = "unknown"
        strata[strata_value].append(sample)

    # Check minimum samples per stratum
    min_samples = min(len(s) for s in strata.values())
    if min_samples < 2:
        raise ValueError(
            f"Stratum '{min(strata, key=lambda k: len(strata[k]))}' has only {min_samples} samples. "
            "Need at least 2 samples per stratum for stratified sampling."
        )

    # Calculate samples per stratum (proportional)
    total_samples = len(samples)
    sampled: list[dict[str, Any]] = []

    for stratum_name, stratum_samples in sorted(strata.items()):
        proportion = len(stratum_samples) / total_samples
        n_stratum = max(1, int(round(n * proportion)))

        # Ensure we don't oversample a stratum
        n_stratum = min(n_stratum, len(stratum_samples))

        selected = random.sample(stratum_samples, n_stratum)
        sampled.extend(selected)

    # Handle rounding: adjust to exactly n samples
    # Shuffle and trim/add as needed
    while len(sampled) > n:
        sampled.pop()
    while len(sampled) < n:
        # Add samples from largest stratum
        largest_stratum = max(strata.values(), key=len)
        remaining = [s for s in largest_stratum if s not in sampled]
        if remaining:
            sampled.append(random.choice(remaining))
        else:
            break

    # Shuffle final selection
    random.shuffle(sampled)

    return sampled


def random_sample(
    samples: list[dict[str, Any]],
    n: int,
    seed: int,
) -> list[dict[str, Any]]:
    """
    Perform simple random sampling with a fixed seed for determinism.

    Args:
        samples: List of sample dictionaries to sample from
        n: Number of samples to select
        seed: Random seed for reproducibility

    Returns:
        List of sampled dictionaries
    """
    import random

    random.seed(seed)

    if n > len(samples):
        raise ValueError(f"Requested {n} samples but only {len(samples)} available")

    return random.sample(samples, n)


def build_hf_dataset(
    in_path: Path,
    out_dir: Path,
    *,
    n: int = 200,
    seed: int = 42,
    stratify_by: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Build and save a HuggingFace dataset from enriched JSONL.

    Args:
        in_path: Path to enriched_samples.jsonl
        out_dir: Directory to save the dataset
        n: Number of samples to select (default: 200)
        seed: Random seed for deterministic selection (default: 42)
        stratify_by: If set, perform stratified sampling by this field

    Returns:
        tuple: (dataset_dict, stats) where stats contains selection and style counts

    Raises:
        FileNotFoundError: If in_path does not exist
        ValueError: If n exceeds available samples or stratify_by field is invalid
    """
    from datasets import Dataset, DatasetDict

    # Read input samples
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    samples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        raise ValueError("No samples found in input file")

    # Enforce unique sample_id
    seen_ids: set[str] = set()
    unique_samples: list[dict[str, Any]] = []
    for sample in samples:
        sample_id = sample.get("sample_id")
        if sample_id not in seen_ids:
            seen_ids.add(sample_id)
            unique_samples.append(sample)
        else:
            print(
                f"Warning: Duplicate sample_id '{sample_id}' skipped", file=sys.stderr
            )

    samples = unique_samples

    # Select samples
    if stratify_by:
        selected = stratified_sample(samples, n, seed, stratify_by)
    else:
        selected = random_sample(samples, n, seed)

    # Count prosody styles
    style_counts: dict[str, int] = defaultdict(int)
    for sample in selected:
        style = sample.get("prosody_style", "unknown")
        if style is None:
            style = "unknown"
        style_counts[style] += 1

    # Create dataset
    ds = Dataset.from_list(selected)
    ds_dict = DatasetDict({"train": ds})

    # Save to disk
    out_dir.mkdir(parents=True, exist_ok=True)
    ds_dict.save_to_disk(str(out_dir))

    # Build stats
    stats = {
        "total_samples": len(samples),
        "selected_samples": len(selected),
        "style_counts": dict(style_counts),
        "seed": seed,
        "stratify_by": stratify_by,
        "output_path": str(out_dir),
    }

    return ds_dict, stats


def main():
    """CLI entrypoint for HF dataset building."""
    parser = argparse.ArgumentParser(
        description="Build a HuggingFace dataset from enriched JSONL with stratified sampling."
    )
    parser.add_argument(
        "--in",
        default=".data/proactive_sat/enriched_samples.jsonl",
        dest="input_path",
        help="Input JSONL file (default: .data/proactive_sat/enriched_samples.jsonl)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of samples to select (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic selection (default: 42)",
    )
    parser.add_argument(
        "--out-dir",
        default=".data/proactive_sat/hf_dataset",
        help="Output directory for the dataset (default: .data/proactive_sat/hf_dataset)",
    )
    parser.add_argument(
        "--stratify-by",
        default=None,
        help="Field to stratify by (e.g., 'prosody_style') for balanced sampling",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print selection statistics",
    )

    args = parser.parse_args()

    try:
        in_path = Path(args.input_path)
        out_dir = Path(args.out_dir)
        n = args.n
        seed = args.seed
        stratify_by = args.stratify_by
        print_stats = args.print_stats

        ds_dict, stats = build_hf_dataset(
            in_path=in_path,
            out_dir=out_dir,
            n=n,
            seed=seed,
            stratify_by=stratify_by,
        )

        if print_stats:
            print(f"Total samples available: {stats['total_samples']}")
            print(f"Samples selected: {stats['selected_samples']}")
            print(f"Seed: {stats['seed']}")
            if stats["stratify_by"]:
                print(f"Stratified by: {stats['stratify_by']}")
            print("Style distribution:")
            for style, count in sorted(stats["style_counts"].items()):
                print(f"  {style}: {count}")
            print(f"Output: {stats['output_path']}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
