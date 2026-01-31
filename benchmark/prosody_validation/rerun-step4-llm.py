#!/usr/bin/env python3
"""
Rerun LLM stage for Step 4 with a new prompt.

Loads step4_asr_responses.jsonl and re-runs the LLM stage for ALL rows with
successful transcriptions (not just failed ones) using an updated prompt.

Writes a new JSONL with updated asr_response field while keeping other fields unchanged.

Usage:
  python rerun-step4-llm.py \
    --input outputs/responses/step4_asr_responses.jsonl \
    --output outputs/responses/step4_asr_responses.new.jsonl \
    --llm-model google/gemini-2.5-flash
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Add parent directory to path for imports (adjust if your script lives elsewhere)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

# NEW PROMPT - Updated ASR_PROMPT
ASR_PROMPT = """You heard someone say: "{transcription}"

Respond naturally as if you're talking to them directly. Show that you understand how they feel and what they're trying to communicate."""


def create_asr_prompt(transcription: str) -> str:
    """Create the ASR prompt with the transcription text."""
    return ASR_PROMPT.format(transcription=transcription)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read JSONL file and return list of dictionaries."""
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_no} in {path}: {e}"
                ) from e
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    """Write list of dictionaries to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def should_process(row: Dict[str, Any]) -> bool:
    """
    Process all rows with successful transcriptions.
    - transcription_success == True
    - transcription is non-empty
    """
    if not row.get("transcription_success", False):
        return False
    tr = (row.get("transcription") or "").strip()
    if not tr:
        return False
    return True


async def rerun_llm_for_all_rows(
    rows: List[Dict[str, Any]],
    llm_model: str,
    max_concurrency: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Rerun LLM stage for all rows with successful transcriptions.

    Args:
        rows: List of all rows from the input JSONL
        llm_model: LLM model to use
        max_concurrency: Maximum concurrent requests
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response

    Returns:
        Tuple of (updated rows, statistics)
    """
    # Get indices of rows to process
    idxs = [i for i, r in enumerate(rows) if should_process(r)]
    stats = {
        "total_rows": len(rows),
        "to_process": len(idxs),
        "process_success": 0,
        "process_failed": 0,
        "skipped": len(rows) - len(idxs),
    }

    if not idxs:
        logger.warning("No rows with successful transcriptions to process")
        return rows, stats

    logger.info(f"Processing {len(idxs)} rows with successful transcriptions")

    sem = asyncio.Semaphore(max_concurrency)
    client = OpenRouterClient()

    async def _process_one(i: int) -> Tuple[int, Dict[str, Any]]:
        """Process a single row through the LLM."""
        r = rows[i]
        transcription = (r.get("transcription") or "").strip()
        prompt = create_asr_prompt(transcription)

        try:
            async with sem:
                resp = await client.chat_text(
                    prompt=prompt,
                    model=llm_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            resp_text = (resp or "").strip()

            if not resp_text:
                # Empty response - mark as failed but keep transcription fields
                updated = {
                    **r,
                    "asr_response": None,
                    "asr_success": False,
                    "asr_error": "Empty LLM response on rerun",
                }
            else:
                # Successful response
                updated = {
                    **r,
                    "asr_response": resp_text,
                    "asr_success": True,
                    "asr_error": None,
                }
            return i, updated

        except Exception as e:
            updated = {
                **r,
                "asr_response": None,
                "asr_success": False,
                "asr_error": f"Rerun failed: {e}",
            }
            return i, updated

    try:
        # Create all tasks
        tasks = [asyncio.create_task(_process_one(i)) for i in idxs]

        # Process with progress bar
        for t in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Rerunning LLM",
            unit="rows",
        ):
            i, updated = await t
            rows[i] = updated
            if updated.get("asr_success"):
                stats["process_success"] += 1
            else:
                stats["process_failed"] += 1

    finally:
        await client.close()

    return rows, stats


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Rerun LLM stage for Step 4 with updated prompt"
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL (step4_asr_responses.jsonl)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL with updated asr_response field",
    )
    ap.add_argument(
        "--llm-model",
        type=str,
        default="google/gemini-2.5-flash",
        help="LLM model name",
    )
    ap.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Max in-flight LLM requests",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature",
    )
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max tokens for response",
    )
    ap.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    ap.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to write a JSON report",
    )
    args = ap.parse_args()

    setup_logging(args.log_level)

    # Load input data
    rows = read_jsonl(args.input)
    logger.info(f"Loaded {len(rows)} rows from {args.input}")

    # Rerun LLM for all applicable rows
    updated_rows, stats = asyncio.run(
        rerun_llm_for_all_rows(
            rows=rows,
            llm_model=args.llm_model,
            max_concurrency=args.max_concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    # Write output
    write_jsonl(updated_rows, args.output)
    logger.info(f"Wrote updated JSONL to {args.output}")

    # Log summary
    logger.info(
        f"Rerun summary: to_process={stats['to_process']} success={stats['process_success']} "
        f"failed={stats['process_failed']} skipped={stats['skipped']}"
    )

    # Write report if requested
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
