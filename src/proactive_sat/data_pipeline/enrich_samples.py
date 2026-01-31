#!/usr/bin/env python3
"""
Proactive-SAT Sample Enrichment CLI

Enriches raw samples with neutralized text and prosody instructions.
Produces enriched_samples.jsonl for downstream phases.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


def enrich_samples(
    in_path: Path,
    out_path: Path,
    *,
    limit: int | None = None,
    neutralizer: str = "rule_based",
    validate: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Enrich raw samples with neutral text and prosody instructions.

    Args:
        in_path: Path to raw_samples.jsonl
        out_path: Path to write enriched_samples.jsonl
        limit: Optional limit on number of samples to process
        neutralizer: Neutralization mode ("rule_based" or "openai")
        validate: Whether to validate output (exits on failure if True)

    Returns:
        tuple: (rows, stats) where rows is list of enriched samples
               and stats is a dict with processing counts

    Raises:
        FileNotFoundError: If in_path does not exist
        ValueError: If neutralizer mode is invalid
    """
    from proactive_sat.data_pipeline.neutralize import neutralize_text
    from proactive_sat.data_pipeline.prosody_instructions import (
        control_speaker_instruction,
        determine_prosody_style,
        trigger_speaker_instruction,
    )

    if neutralizer not in ("rule_based", "openai"):
        raise ValueError(
            f"Invalid neutralizer mode: {neutralizer!r}. "
            "Valid modes are: 'rule_based', 'openai'"
        )

    # Read input samples
    if not in_path.exists():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    samples = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # Apply limit
    if limit is not None:
        samples = samples[:limit]

    # Process each sample
    enriched_samples = []
    stats = {
        "input_samples": len(samples),
        "enriched_samples": 0,
        "skipped_samples": 0,
        "validation_failures": 0,
    }

    errors = []

    for sample in samples:
        # Get original text
        source_text = sample.get("source_text", "")

        # Skip samples with empty source_text (nothing to neutralize)
        if not source_text or not source_text.strip():
            stats["skipped_samples"] += 1
            continue

        emotion = sample.get("emotion")

        # Generate neutralized text
        neutral_text = neutralize_text(source_text, mode=neutralizer)

        # Determine prosody style
        prosody_style = determine_prosody_style(emotion)

        # Generate instructions
        control_instruction = control_speaker_instruction(neutral_text)
        trigger_instruction = trigger_speaker_instruction(neutral_text, prosody_style)

        # Create enriched sample (preserve all original fields)
        enriched = dict(sample)
        enriched["neutral_text"] = neutral_text
        enriched["prosody_style"] = prosody_style
        enriched["control_speaker_instruction"] = control_instruction
        enriched["trigger_speaker_instruction"] = trigger_instruction
        enriched["control_text"] = neutral_text
        enriched["trigger_text"] = neutral_text

        enriched_samples.append(enriched)
        stats["enriched_samples"] += 1

        # Validation (if requested)
        if validate:
            validation_errors = _validate_sample(enriched)
            if validation_errors:
                stats["validation_failures"] += 1
                errors.extend(validation_errors)

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for sample in enriched_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    # Report validation errors if any
    if validate and errors:
        for error in errors[:10]:  # Limit error output
            print(f"Validation error: {error}", file=sys.stderr)
        if len(errors) > 10:
            print(f"... and {len(errors) - 10} more errors", file=sys.stderr)
        sys.exit(1)

    return enriched_samples, stats


def _validate_sample(sample: dict[str, Any]) -> list[str]:
    """
    Validate an enriched sample has all required fields.

    Args:
        sample: The enriched sample to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    # Required original fields
    required_original = ["sample_id", "source_text"]
    for field in required_original:
        if field not in sample or not sample[field]:
            errors.append(f"Missing required field: {field}")

    # Required enriched fields
    required_enriched = [
        "neutral_text",
        "prosody_style",
        "control_text",
        "trigger_text",
    ]
    for field in required_enriched:
        if field not in sample:
            errors.append(f"Missing enriched field: {field}")

    # Validate neutral_text is non-empty
    if "neutral_text" in sample:
        if not sample["neutral_text"] or not sample["neutral_text"].strip():
            errors.append("neutral_text must be non-empty")

    # Validate prosody_style is one of allowed values
    valid_styles = {"sarcastic", "frustrated", "distressed"}
    if "prosody_style" in sample:
        if sample["prosody_style"] not in valid_styles:
            errors.append(
                f"prosody_style must be one of {valid_styles}, got: {sample['prosody_style']!r}"
            )

    # Validate control_text and trigger_text equal neutral_text
    neutral_text = sample.get("neutral_text", "")
    if "control_text" in sample and sample["control_text"] != neutral_text:
        errors.append("control_text must equal neutral_text")
    if "trigger_text" in sample and sample["trigger_text"] != neutral_text:
        errors.append("trigger_text must equal neutral_text")

    return errors


def main():
    """CLI entrypoint for sample enrichment."""
    parser = argparse.ArgumentParser(
        description="Enrich raw samples with neutralized text and prosody instructions."
    )
    parser.add_argument(
        "--in",
        default=".data/proactive_sat/raw_samples.jsonl",
        dest="input_path",
        help="Input JSONL file (default: .data/proactive_sat/raw_samples.jsonl)",
    )
    parser.add_argument(
        "--out",
        default=".data/proactive_sat/enriched_samples.jsonl",
        help="Output JSONL file (default: .data/proactive_sat/enriched_samples.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of samples to process",
    )
    parser.add_argument(
        "--neutralizer",
        choices=["rule_based", "openai"],
        default="rule_based",
        help="Neutralization mode (default: rule_based)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output has required fields and non-empty neutral_text",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print processing statistics",
    )

    args = parser.parse_args()

    try:
        in_path = Path(args.input_path)
        out_path = Path(args.out)
        limit = args.limit
        neutralizer = args.neutralizer
        validate = args.validate
        print_stats = args.print_stats

        rows, stats = enrich_samples(
            in_path=in_path,
            out_path=out_path,
            limit=limit,
            neutralizer=neutralizer,
            validate=validate,
        )

        if print_stats:
            print(f"Input samples: {stats['input_samples']}")
            print(f"Enriched samples: {stats['enriched_samples']}")
            if validate:
                print(f"Validation failures: {stats['validation_failures']}")
            print(f"Output: {out_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
