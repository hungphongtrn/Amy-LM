#!/usr/bin/env python3
"""
Proactive-SAT Phase 1 Pipeline Runner

One-command runner that orchestrates the full Phase 1 data pipeline:
parse_data → enrich_samples → build_hf_dataset

Produces a 200-sample HuggingFace dataset from source data files.
"""

import argparse
import sys
from pathlib import Path


def run_pipeline(
    *,
    limit: int | None = None,
    neutralizer: str = "rule_based",
    n: int = 200,
    seed: int = 42,
    stratify_by: str | None = None,
    print_stats: bool = False,
) -> dict:
    """
    Run the full Phase 1 data pipeline.

    Steps:
        1. Parse Dialogue.tsv + Annotation.csv → raw_samples.jsonl
        2. Enrich raw samples → enriched_samples.jsonl
        3. Build HF dataset → .data/proactive_sat/hf_dataset

    Args:
        limit: Optional limit on samples for parse and enrich steps
        neutralizer: Neutralization mode ("rule_based" or "openai")
        n: Number of samples for HF dataset
        seed: Random seed for deterministic sampling
        stratify_by: Field to stratify by for balanced sampling
        print_stats: Print statistics at each step

    Returns:
        dict: Pipeline statistics
    """
    from proactive_sat.data_pipeline.build_hf_dataset import build_hf_dataset
    from proactive_sat.data_pipeline.enrich_samples import enrich_samples
    from proactive_sat.data_pipeline.parse_data import parse_data

    stats = {}

    # Step 1: Parse data
    print("Step 1: Parsing source data...")
    raw_out = Path(".data/proactive_sat/raw_samples.jsonl")
    parse_rows, parse_stats = parse_data(
        out_path=raw_out,
        limit=limit,
        print_stats=print_stats,
    )
    stats["parse"] = parse_stats
    print(f"  Parsed {parse_stats['written_rows']} samples")

    # Step 2: Enrich samples
    print("Step 2: Enriching samples with neutralization and prosody instructions...")
    enriched_out = Path(".data/proactive_sat/enriched_samples.jsonl")
    enrich_rows, enrich_stats = enrich_samples(
        in_path=raw_out,
        out_path=enriched_out,
        limit=limit,
        neutralizer=neutralizer,
    )
    stats["enrich"] = enrich_stats
    print(f"  Enriched {enrich_stats['enriched_samples']} samples")

    # Step 3: Build HF dataset
    print("Step 3: Building HuggingFace dataset...")
    hf_out = Path(".data/proactive_sat/hf_dataset")
    ds_dict, ds_stats = build_hf_dataset(
        in_path=enriched_out,
        out_dir=hf_out,
        n=n,
        seed=seed,
        stratify_by=stratify_by,
    )
    stats["dataset"] = ds_stats
    print(f"  Created dataset with {ds_stats['selected_samples']} samples")
    print(f"  Saved to: {hf_out}")

    return stats


def main():
    """CLI entrypoint for the Phase 1 pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the full Phase 1 data pipeline: parse → enrich → build HF dataset"
    )

    # Shared options
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit samples processed in parse and enrich steps",
    )

    # Enrichment options
    parser.add_argument(
        "--neutralizer",
        choices=["rule_based", "openai"],
        default="rule_based",
        help="Neutralization mode (default: rule_based)",
    )

    # Dataset build options
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of samples in final HF dataset (default: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--stratify-by",
        default=None,
        help="Field to stratify by (e.g., 'prosody_style') for balanced sampling",
    )

    # Output options
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print statistics at each step",
    )

    args = parser.parse_args()

    try:
        stats = run_pipeline(
            limit=args.limit,
            neutralizer=args.neutralizer,
            n=args.n,
            seed=args.seed,
            stratify_by=args.stratify_by,
            print_stats=args.print_stats,
        )

        # Summary
        print("\n=== Pipeline Complete ===")
        print(f"Raw samples: {stats['parse']['written_rows']}")
        print(f"Enriched samples: {stats['enrich']['enriched_samples']}")
        print(f"HF dataset samples: {stats['dataset']['selected_samples']}")
        print(f"Output: {stats['dataset']['output_path']}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
