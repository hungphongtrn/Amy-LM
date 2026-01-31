"""
Step 5: End-to-End Audio Pipeline - Direct audio analysis with multimodal LLM
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils import setup_logging, read_csv, write_jsonl
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


# E2E audio prompt template
E2E_PROMPT = """Listen to this audio recording.

How would you respond to this person? What are they trying to communicate?"""


def create_e2e_prompt() -> str:
    return E2E_PROMPT


async def process_single_e2e(
    client: OpenRouterClient,
    item: Dict[str, Any],
    model: str,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    audio_path = Path(item.get("audio_path", ""))

    async def _run() -> Dict[str, Any]:
        # Validate audio file exists
        if not audio_path.exists():
            return {
                "dialog_id": item.get("dialog_id", ""),
                "audio_path": str(audio_path),
                "original_utterance": item.get("text", ""),
                "rewritten_text": item.get("text", ""),
                "emotion": item.get("emotion", ""),
                "model": model,
                "e2e_response": None,
                "success": False,
                "error": "Audio file not found",
            }

        prompt = create_e2e_prompt()
        response = await client.chat_with_audio(
            audio_path=audio_path,
            prompt=prompt,
            model=model,
            temperature=0.7,
            max_tokens=256,
        )

        return {
            "dialog_id": item.get("dialog_id", ""),
            "audio_path": str(audio_path),
            "original_utterance": item.get("text", ""),
            "rewritten_text": item.get("text", ""),
            "emotion": item.get("emotion", ""),
            "model": model,
            "e2e_response": response.strip() if isinstance(response, str) else response,
            "success": True,
            "error": None,
        }

    try:
        if semaphore is None:
            return await _run()
        async with semaphore:
            return await _run()
    except Exception as e:
        logger.error(f"E2E error for {audio_path}: {e}")
        return {
            "dialog_id": item.get("dialog_id", ""),
            "audio_path": str(audio_path),
            "original_utterance": item.get("text", ""),
            "rewritten_text": item.get("text", ""),
            "emotion": item.get("emotion", ""),
            "model": model,
            "e2e_response": None,
            "success": False,
            "error": str(e),
        }


async def process_e2e_batch(
    client: OpenRouterClient,
    items: List[Dict[str, Any]],
    model: str,
    progress_desc: str = "Processing E2E audio",
    concurrency: int = 8,
) -> List[Dict[str, Any]]:
    from tqdm.asyncio import tqdm_asyncio

    sem = asyncio.Semaphore(concurrency)

    tasks = [
        process_single_e2e(client, item, model, semaphore=sem)
        for item in items
    ]

    # tqdm_asyncio.gather shows a progress bar and awaits all tasks
    results = await tqdm_asyncio.gather(*tasks, desc=progress_desc, unit="files")
    return list(results)


async def run_step5_async(
    manifest_path: Path,
    output_path: Path,
    model: str = "google/gemini-2.5-flash",
    concurrency: int = 8,
) -> bool:
    logger.info("=" * 60)
    logger.info("STEP 5: End-to-End Audio Pipeline")
    logger.info("=" * 60)

    try:
        logger.info(f"Loading manifest from {manifest_path}")
        manifest = read_csv(manifest_path)
        logger.info(f"Loaded {len(manifest)} items")

        successful_items = [
            item for item in manifest if str(item.get("success", "")).lower() == "true"
        ]
        logger.info(f"Processing {len(successful_items)} successful audio files")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if not successful_items:
            logger.warning("No successful audio files to process")
            write_jsonl([], output_path)
            return True

        # IMPORTANT: create and close client inside the SAME event loop
        client = OpenRouterClient()
        try:
            results = await process_e2e_batch(
                client=client,
                items=successful_items,
                model=model,
                progress_desc="Processing E2E audio with multimodal LLM",
                concurrency=concurrency,
            )
        finally:
            await client.close()

        results.sort(key=lambda x: x.get("dialog_id", ""))

        write_jsonl(results, output_path)
        logger.info(f"Saved {len(results)} E2E responses to {output_path}")

        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"E2E pipeline success: {success_count}/{len(results)}")

        logger.info("Sample E2E responses:")
        for result in results[:3]:
            if result.get("success"):
                logger.info(f"  [{result.get('dialog_id')}]")
                logger.info(f"    Original: {result.get('original_utterance', '')[:80]}...")
                logger.info(f"    E2E Response: {str(result.get('e2e_response', ''))[:80]}...")

        logger.info("=" * 60)
        logger.info("STEP 5 COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Step 5 failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_step5(
    manifest_path: Path,
    output_path: Path,
    model: str = "google/gemini-2.5-flash",
    concurrency: int = 8,
) -> bool:
    # Single asyncio.run for the whole step
    return asyncio.run(run_step5_async(manifest_path, output_path, model, concurrency))


def main():
    parser = argparse.ArgumentParser(description="Step 5: End-to-End Audio Pipeline")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step2_audio_manifest.csv",
        help="Path to audio manifest CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "responses" / "step5_e2e_responses.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Multimodal model for E2E processing",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="How many audio files to process concurrently",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()
    setup_logging(args.log_level)

    success = run_step5(
        manifest_path=args.manifest,
        output_path=args.output,
        model=args.model,
        concurrency=args.concurrency,
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
