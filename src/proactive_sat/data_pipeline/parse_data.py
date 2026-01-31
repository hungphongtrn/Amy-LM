#!/usr/bin/env python3
"""
Proactive-SAT Data Pipeline Parser

Parses Dialogue.tsv + Annotation.csv into canonical JSONL format.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any


def find_source_file(directory: Path, pattern: str) -> Path | None:
    """Find a source file matching the given pattern (suffix)."""
    for file in directory.iterdir():
        if file.is_file() and file.name.endswith(pattern):
            return file
    return None


def detect_source_root() -> Path:
    """Detect the source root containing Dialogue.tsv and Annotation.csv files."""
    # Check .data/ first
    data_dir = Path(".data")
    if data_dir.exists():
        dialogue_file = find_source_file(data_dir, "Dialogue.tsv")
        annotation_file = find_source_file(data_dir, "Annotation.csv")
        if dialogue_file and annotation_file:
            return data_dir

    # Fall back to data/
    data_dir = Path("data")
    if data_dir.exists():
        dialogue_file = find_source_file(data_dir, "Dialogue.tsv")
        annotation_file = find_source_file(data_dir, "Annotation.csv")
        if dialogue_file and annotation_file:
            return data_dir

    # Neither directory has the required files
    print("Error: Could not find source files.", file=sys.stderr)
    print(
        "Please place Dialogue.tsv and Annotation.csv in either .data/ or data/",
        file=sys.stderr,
    )
    sys.exit(1)


def parse_tsv(file_path: Path) -> dict[str, dict]:
    """Parse a TSV file and return a dict mapping dialog_id to row data."""
    rows = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            dialog_id = row.get("dialog_id")
            if dialog_id:
                rows[dialog_id] = row
    return rows


def parse_csv(file_path: Path) -> dict[str, dict]:
    """Parse a CSV file and return a dict mapping dialog_id to row data."""
    rows = {}
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dialog_id = row.get("dialog_id")
            if dialog_id:
                rows[dialog_id] = row
    return rows


def create_sample(
    dialog_id: str,
    dialogue_row: dict[str, str],
    annotation_row: dict[str, str] | None,
    dialogue_path: Path,
    annotation_path: Path,
) -> dict[str, Any]:
    """Create a sample object from joined dialogue and annotation data."""
    sample = {
        "sample_id": dialog_id,
        "dialog_id": dialog_id,
        "source_text": dialogue_row.get("user1", ""),
        "source": {
            "dialogue_path": str(dialogue_path),
            "annotation_path": str(annotation_path),
        },
    }

    # Add annotation fields if available
    if annotation_row:
        sample["speech_act"] = annotation_row.get("speech_act")
        sample["intent"] = annotation_row.get("intent")
        sample["emotion"] = annotation_row.get("emotion")
        sample["implicature_text"] = annotation_row.get("implicature_text")
        sample["confidence"] = annotation_row.get("confidence")
    else:
        sample["speech_act"] = None
        sample["intent"] = None
        sample["emotion"] = None
        sample["implicature_text"] = None
        sample["confidence"] = None

    return sample


def parse_data(
    out_path: Path | None = None,
    limit: int | None = None,
    print_stats: bool = False,
) -> tuple[list[dict], dict[str, int]]:
    """
    Parse Dialogue.tsv + Annotation.csv into canonical JSONL format.

    Returns:
        tuple: (rows, stats) where rows is the list of sample dicts and stats
               is a dict with counts (dialogue_rows, annotation_rows, joined_rows, written_rows)
    """
    # Detect source root
    source_root = detect_source_root()

    # Find source files
    dialogue_path = find_source_file(source_root, "Dialogue.tsv")
    annotation_path = find_source_file(source_root, "Annotation.csv")

    if not dialogue_path:
        print(
            "Error: Could not find *Dialogue.tsv in source directory", file=sys.stderr
        )
        sys.exit(1)

    if not annotation_path:
        print(
            "Error: Could not find *Annotation.csv in source directory", file=sys.stderr
        )
        sys.exit(1)

    # Parse source files
    dialogue_rows = parse_tsv(dialogue_path)
    annotation_rows = parse_csv(annotation_path)

    stats = {
        "dialogue_rows": len(dialogue_rows),
        "annotation_rows": len(annotation_rows),
    }

    # Join on dialog_id (inner join)
    joined_rows = []
    for dialog_id in dialogue_rows:
        if dialog_id in annotation_rows:
            sample = create_sample(
                dialog_id,
                dialogue_rows[dialog_id],
                annotation_rows[dialog_id],
                dialogue_path,
                annotation_path,
            )
            joined_rows.append(sample)

    stats["joined_rows"] = len(joined_rows)

    # Sort by dialog_id for determinism
    joined_rows.sort(key=lambda x: x["dialog_id"])

    # Apply limit if specified
    if limit is not None:
        joined_rows = joined_rows[:limit]

    # Write to output file
    written_rows = 0
    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for sample in joined_rows:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                written_rows += 1

    stats["written_rows"] = written_rows

    if print_stats:
        print(f"Dialogue rows: {stats['dialogue_rows']}")
        print(f"Annotation rows: {stats['annotation_rows']}")
        print(f"Joined rows: {stats['joined_rows']}")
        print(f"Written rows: {stats['written_rows']}")

    return joined_rows, stats


def main():
    """CLI entrypoint for the parser."""
    parser = argparse.ArgumentParser(
        description="Parse Proactive-SAT source data into canonical JSONL format."
    )
    parser.add_argument(
        "--out",
        default=".data/proactive_sat/raw_samples.jsonl",
        help="Output file path (default: .data/proactive_sat/raw_samples.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of rows written (after sorting)",
    )
    parser.add_argument(
        "--print-stats",
        action="store_true",
        help="Print row counts (dialogue, annotation, joined, written)",
    )

    args = parser.parse_args()

    out_path = Path(args.out)
    limit = args.limit
    print_stats = args.print_stats

    parse_data(out_path=out_path, limit=limit, print_stats=print_stats)


if __name__ == "__main__":
    main()
