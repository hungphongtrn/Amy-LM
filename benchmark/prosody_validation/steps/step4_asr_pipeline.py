"""
Step 4: ASR Pipeline - Transcribe audio, then analyze with LLM
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
from src.whisper_wrapper import WhisperWrapper
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


# ASR response prompt template
ASR_PROMPT = """You heard someone say: "{transcription}"

How would you respond? What do you think they mean?"""


def create_asr_prompt(transcription: str) -> str:
    """Create the ASR response prompt.

    Args:
        transcription: The transcribed audio text

    Returns:
        Formatted prompt string
    """
    return ASR_PROMPT.format(transcription=transcription)


async def process_single_asr(
    whisper_wrapper: WhisperWrapper,
    openrouter_client: OpenRouterClient,
    item: Dict[str, Any],
    asr_model: str,
    llm_model: str,
) -> Dict[str, Any]:
    """Process a single audio file through ASR pipeline.

    Args:
        whisper_wrapper: Whisper wrapper instance
        openrouter_client: OpenRouter client
        item: Dictionary with audio_path and metadata
        asr_model: Whisper model size
        llm_model: LLM model for response

    Returns:
        Dictionary with processing results
    """
    audio_path = Path(item.get("audio_path", ""))

    # Step 1: Transcribe audio
    try:
        transcription_result = whisper_wrapper.transcribe_audio(audio_path)

        if not transcription_result.get("success"):
            return {
                "dialog_id": item.get("dialog_id", ""),
                "audio_path": str(audio_path),
                "original_utterance": item.get("text", ""),
                "rewritten_text": item.get("text", ""),
                "emotion": item.get("emotion", ""),
                "transcription": None,
                "transcription_success": False,
                "transcription_error": transcription_result.get(
                    "error", "Unknown error"
                ),
                "asr_response": None,
                "asr_success": False,
                "asr_error": "Skipped due to transcription failure",
            }

        transcription = transcription_result.get("text", "")

    except Exception as e:
        logger.error(f"Transcription error for {audio_path}: {e}")
        return {
            "dialog_id": item.get("dialog_id", ""),
            "audio_path": str(audio_path),
            "original_utterance": item.get("text", ""),
            "rewritten_text": item.get("text", ""),
            "emotion": item.get("emotion", ""),
            "transcription": None,
            "transcription_success": False,
            "transcription_error": str(e),
            "asr_response": None,
            "asr_success": False,
            "asr_error": "Skipped due to transcription failure",
        }

    # Step 2: Get LLM response from transcription
    try:
        prompt = create_asr_prompt(transcription)
        response = await openrouter_client.chat_text(
            prompt=prompt, model=llm_model, temperature=0.7, max_tokens=256
        )

        return {
            "dialog_id": item.get("dialog_id", ""),
            "audio_path": str(audio_path),
            "original_utterance": item.get("text", ""),
            "rewritten_text": item.get("text", ""),
            "emotion": item.get("emotion", ""),
            "transcription": transcription,
            "transcription_success": True,
            "transcription_error": None,
            "asr_response": response.strip(),
            "asr_success": True,
            "asr_error": None,
        }

    except Exception as e:
        logger.error(f"ASR LLM error for {audio_path}: {e}")
        return {
            "dialog_id": item.get("dialog_id", ""),
            "audio_path": str(audio_path),
            "original_utterance": item.get("text", ""),
            "rewritten_text": item.get("text", ""),
            "emotion": item.get("emotion", ""),
            "transcription": transcription,
            "transcription_success": True,
            "transcription_error": None,
            "asr_response": None,
            "asr_success": False,
            "asr_error": str(e),
        }


async def process_asr_batch(
    whisper_wrapper: WhisperWrapper,
    openrouter_client: OpenRouterClient,
    items: List[Dict[str, Any]],
    asr_model: str,
    llm_model: str,
    progress_desc: str = "Processing ASR pipeline",
) -> List[Dict[str, Any]]:
    """Process a batch of audio files through ASR pipeline.

    Args:
        whisper_wrapper: Whisper wrapper instance
        openrouter_client: OpenRouter client
        items: List of dictionaries with audio data
        asr_model: Whisper model size
        llm_model: LLM model for response
        progress_desc: Description for progress bar

    Returns:
        List of processing result dictionaries
    """
    from tqdm.asyncio import tqdm_asyncio

    # Create tasks for all items
    tasks = [
        process_single_asr(
            whisper_wrapper, openrouter_client, item, asr_model, llm_model
        )
        for item in items
    ]

    # Process with progress bar
    progress_bar = tqdm_asyncio(total=len(tasks), desc=progress_desc, unit="files")

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        progress_bar.update(1)

    progress_bar.close()

    return results


def run_step4(
    manifest_path: Path,
    output_path: Path,
    asr_model: str = "base",
    llm_model: str = "openai/gpt-4o-mini",
) -> bool:
    """Execute Step 4: ASR Pipeline.

    Args:
        manifest_path: Path to step2_audio_manifest.csv
        output_path: Path to save step4_asr_responses.jsonl
        asr_model: Whisper model size
        llm_model: LLM model for response

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 4: ASR Pipeline")
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

        # Create Whisper wrapper
        whisper_wrapper = WhisperWrapper(
            model_size=asr_model, device=get_config().WHISPER_DEVICE
        )

        if not whisper_wrapper.load_model():
            logger.error("Failed to load Whisper model")
            return False

        # Create OpenRouter client
        openrouter_client = OpenRouterClient()

        try:
            # Process all items
            results = asyncio.run(
                process_asr_batch(
                    whisper_wrapper,
                    openrouter_client,
                    successful_items,
                    asr_model,
                    llm_model,
                    "Processing ASR pipeline",
                )
            )

        finally:
            whisper_wrapper.unload_model()
            asyncio.run(openrouter_client.close())

        # Sort results by dialog_id
        results.sort(key=lambda x: x.get("dialog_id", ""))

        # Save output
        write_jsonl(results, output_path)
        logger.info(f"Saved {len(results)} ASR responses to {output_path}")

        # Log summary
        transcription_success = sum(
            1 for r in results if r.get("transcription_success")
        )
        asr_success = sum(1 for r in results if r.get("asr_success"))
        logger.info(f"Transcription success: {transcription_success}/{len(results)}")
        logger.info(f"ASR pipeline success: {asr_success}/{len(results)}")

        # Log sample responses
        logger.info("Sample ASR responses:")
        for i, result in enumerate(results[:3]):
            if result.get("asr_success"):
                logger.info(f"  [{result.get('dialog_id')}]")
                logger.info(
                    f"    Original: {result.get('original_utterance', '')[:80]}..."
                )
                logger.info(
                    f"    Transcription: {result.get('transcription', '')[:80]}..."
                )
                logger.info(f"    Response: {result.get('asr_response', '')[:80]}...")

        logger.info("=" * 60)
        logger.info("STEP 4 COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Step 4 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for Step 4."""
    parser = argparse.ArgumentParser(description="Step 4: ASR Pipeline")
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
        / "step4_asr_responses.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="LLM model for response",
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
    success = run_step4(
        manifest_path=args.manifest,
        output_path=args.output,
        asr_model=args.asr_model,
        llm_model=args.llm_model,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
