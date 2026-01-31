#!/usr/bin/env python3
"""
Retry LLM stage for Step 4 outputs.

Loads step4_asr_responses.jsonl and re-runs ONLY the LLM stage for rows where:
- transcription_success == True
- transcription is non-empty
- asr_success == False  (LLM failed / skipped / empty response)

Writes a new JSONL with updated rows (others unchanged).
By default also writes a small retry report.

Usage:
  python retry_step4_llm.py \
    --input path/to/step4_asr_responses.jsonl \
    --output path/to/step4_asr_responses.retry.jsonl \
    --llm-model openai/gpt-4o-mini \
    --max-concurrency 16 \
    --max-tokens 256 \
    --temperature 0.7
"""

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

# Add parent directory to path for imports (adjust if your script lives elsewhere)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)

ASR_PROMPT = """You heard someone say: "{transcription}"

How would you respond? What do you think they mean?"""


def create_asr_prompt(transcription: str) -> str:
    return ASR_PROMPT.format(transcription=transcription)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}: {e}") from e
    return rows


def write_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def should_retry(row: Dict[str, Any]) -> bool:
    """
    Retry only when ASR transcription succeeded but LLM response failed.
    Matches your request: "after asr succeeded but llm fail".
    """
    if not row.get("transcription_success", False):
        return False
    tr = (row.get("transcription") or "").strip()
    if not tr:
        return False
    # In your schema, "asr_success" is the LLM stage success flag.
    if row.get("asr_success", False):
        return False
    return True


async def retry_llm_rows(
    rows: List[Dict[str, Any]],
    llm_model: str,
    max_concurrency: int,
    temperature: float,
    max_tokens: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    idxs = [i for i, r in enumerate(rows) if should_retry(r)]
    stats = {
        "total_rows": len(rows),
        "to_retry": len(idxs),
        "retry_success": 0,
        "retry_failed": 0,
        "skipped": len(rows) - len(idxs),
    }

    if not idxs:
        return rows, stats

    sem = asyncio.Semaphore(max_concurrency)
    client = OpenRouterClient()

    async def _one(i: int) -> Tuple[int, Dict[str, Any]]:
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
                # Treat empty as failure, keep transcription fields intact
                updated = {
                    **r,
                    "asr_response": None,
                    "asr_success": False,
                    "asr_error": "Empty LLM response on retry",
                }
                return i, updated

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
                "asr_error": f"Retry failed: {e}",
            }
            return i, updated

    try:
        tasks = [asyncio.create_task(_one(i)) for i in idxs]
        for t in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Retrying LLM", unit="rows"):
            i, updated = await t
            rows[i] = updated
            if updated.get("asr_success"):
                stats["retry_success"] += 1
            else:
                stats["retry_failed"] += 1
    finally:
        await client.close()

    return rows, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Retry LLM stage for failed rows in step4_asr_responses.jsonl")
    ap.add_argument("--input", type=Path, required=True, help="Input JSONL (step4_asr_responses.jsonl)")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL with retried rows updated")
    ap.add_argument("--llm-model", type=str, default="openai/gpt-4o-mini", help="LLM model name")
    ap.add_argument("--max-concurrency", type=int, default=16, help="Max in-flight LLM requests")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max tokens for response")
    ap.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    ap.add_argument("--report", type=Path, default=None, help="Optional path to write a small JSON report")
    args = ap.parse_args()

    setup_logging(args.log_level)

    rows = read_jsonl(args.input)
    logger.info(f"Loaded {len(rows)} rows from {args.input}")

    updated_rows, stats = asyncio.run(
        retry_llm_rows(
            rows=rows,
            llm_model=args.llm_model,
            max_concurrency=args.max_concurrency,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    )

    write_jsonl(updated_rows, args.output)
    logger.info(f"Wrote updated JSONL to {args.output}")

    logger.info(
        f"Retry summary: to_retry={stats['to_retry']} success={stats['retry_success']} failed={stats['retry_failed']} skipped={stats['skipped']}"
    )

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open("w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info(f"Wrote report to {args.report}")


if __name__ == "__main__":
    main()
