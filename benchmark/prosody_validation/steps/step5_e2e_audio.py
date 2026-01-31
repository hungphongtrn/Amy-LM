"""
Step 5: End-to-End Audio Pipeline - Direct audio analysis with multimodal LLM
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import get_config
from src.utils import setup_logging, read_csv, write_jsonl
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


# E2E audio prompt template
E2E_PROMPT = """Listen to this audio recording.

How would you respond to this person? What are they trying to communicate?"""


def create_e2e_prompt() -> str:
    """Create the E2E audio prompt.

    Returns:
        Formatted prompt string
    """
    return E2E_PROMPT


async def process_single_e2e(
    client: OpenRouterClient, item: Dict[str, Any], model: str
) -> Dict[str, Any]:
    """Process a single audio file with multimodal LLM.

    Args:
        client: OpenRouter client
        item: Dictionary with audio_path and metadata
        model: Multimodal model to use

    Returns:
        Dictionary with processing results
    """
    audio_path = Path(item.get("audio_path", ""))

    try:
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

        # Get response from multimodal LLM
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
            "e2e_response": response.strip(),
            "success": True,
            "error": None,
        }

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
) -> List[Dict[str, Any]]:
    """Process a batch of audio files with multimodal LLM.

    Args:
        client: OpenRouter client
        items: List of dictionaries with audio data
        model: Multimodal model to use
        progress_desc: Description for progress bar

    Returns:
        List of processing result dictionaries
    """
    from tqdm.asyncio import tqdm_asyncio

    # Create tasks for all items
    tasks = [process_single_e2e(client, item, model) for item in items]

    # Process with progress bar
    progress_bar = tqdm_asyncio(total=len(tasks), desc=progress_desc, unit="files")

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        progress_bar.update(1)

    progress_bar.close()

    return results


def run_step5(
    manifest_path: Path, output_path: Path, model: str = "google/gemini-2.5-flash"
) -> bool:
    """Execute Step 5: End-to-End Audio Pipeline.

    Args:
        manifest_path: Path to step2_audio_manifest.csv
        output_path: Path to save step5_e2e_responses.jsonl
        model: Multimodal model to use

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 5: End-to-End Audio Pipeline")
    logger.info("=" * 60)

    try:
        # Load manifest
        logger.info(f"Loading manifest from {manifest_path}")
        manifest = read_csv(manifest_path)
        logger.info(f"Loaded {len(manifest)} items")

        # Filter to only successful audio files
        successful_items = [
            item for item in manifest if item.get("success", "").lower() == "true"
        ]
        logger.info(f"Processing {len(successful_items)} successful audio files")

        if not successful_items:
            logger.warning("No successful audio files to process")
            # Still write empty output
            write_jsonl([], output_path)
            return True

        # Create OpenRouter client
        client = OpenRouterClient()

        try:
            # Process all items
            results = asyncio.run(
                process_e2e_batch(
                    client,
                    successful_items,
                    model,
                    "Processing E2E audio with multimodal LLM",
                )
            )

        finally:
            asyncio.run(client.close())

        # Sort results by dialog_id
        results.sort(key=lambda x: x.get("dialog_id", ""))

        # Save output
        write_jsonl(results, output_path)
        logger.info(f"Saved {len(results)} E2E responses to {output_path}")

        # Log summary
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"E2E pipeline success: {success_count}/{len(results)}")

        # Log sample responses
        logger.info("Sample E2E responses:")
        for i, result in enumerate(results[:3]):
            if result.get("success"):
                logger.info(f"  [{result.get('dialog_id')}]")
                logger.info(
                    f"    Original: {result.get('original_utterance', '')[:80]}..."
                )
                logger.info(
                    f"    E2E Response: {result.get('e2e_response', '')[:80]}..."
                )

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


def main():
    """Main entry point for Step 5."""
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
        default=Path(__file__).parent.parent
        / "outputs"
        / "responses"
        / "step5_e2e_responses.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Multimodal model for E2E processing",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Run step
    success = run_step5(
        manifest_path=args.manifest, output_path=args.output, model=args.model
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
