"""
Step 3: Text-only baseline using rewritten text
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


# Response prompt template
RESPONSE_PROMPT = """User says: "{rewritten_text}"

How would you respond? What do you think they mean?"""


def create_response_prompt(rewritten_text: str) -> str:
    """Create the response prompt for a single utterance.

    Args:
        rewritten_text: The rewritten/neutral utterance

    Returns:
        Formatted prompt string
    """
    return RESPONSE_PROMPT.format(rewritten_text=rewritten_text)


async def get_response_single(
    client: OpenRouterClient, item: Dict[str, Any], model: str
) -> Dict[str, Any]:
    """Get a natural response for a single rewritten utterance.

    Args:
        client: OpenRouter client
        item: Dictionary with dialog_id and rewritten_text
        model: Model to use for response

    Returns:
        Dictionary with response data
    """
    prompt = create_response_prompt(item.get("rewritten_text", ""))

    try:
        response = await client.chat_text(
            prompt=prompt, model=model, temperature=0.7, max_tokens=256
        )

        return {
            "dialog_id": item.get("dialog_id", ""),
            "original_utterance": item.get("utterance", ""),
            "rewritten_text": item.get("rewritten_text", ""),
            "emotion": item.get("emotion", ""),
            "intent": item.get("intent", ""),
            "speech_act": item.get("speech_act", ""),
            "model": model,
            "response": response.strip(),
            "success": True,
            "error": None,
        }

    except Exception as e:
        logger.error(f"Error getting response for {item.get('dialog_id')}: {e}")
        return {
            "dialog_id": item.get("dialog_id", ""),
            "original_utterance": item.get("utterance", ""),
            "rewritten_text": item.get("rewritten_text", ""),
            "emotion": item.get("emotion", ""),
            "intent": item.get("intent", ""),
            "speech_act": item.get("speech_act", ""),
            "model": model,
            "response": None,
            "success": False,
            "error": str(e),
        }


async def get_responses_batch(
    client: OpenRouterClient,
    items: List[Dict[str, Any]],
    model: str,
    progress_desc: str = "Getting text responses",
) -> List[Dict[str, Any]]:
    """Get responses for a batch of utterances concurrently.

    Args:
        client: OpenRouter client
        items: List of dictionaries with utterance data
        model: Model to use
        progress_desc: Description for progress bar

    Returns:
        List of response dictionaries
    """
    from tqdm.asyncio import tqdm_asyncio

    # Create tasks for all items
    tasks = [get_response_single(client, item, model) for item in items]

    # Process with progress bar
    progress_bar = tqdm_asyncio(total=len(tasks), desc=progress_desc, unit="responses")

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        progress_bar.update(1)

    progress_bar.close()

    return results


def run_step3(
    input_path: Path, output_path: Path, model: str = "openai/gpt-4o-mini"
) -> bool:
    """Execute Step 3: Text-only baseline.

    Args:
        input_path: Path to step1_rewritten.csv
        output_path: Path to save step3_text_responses.jsonl
        model: Model to use for responses

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 3: Text-Only Baseline")
    logger.info("=" * 60)

    try:
        # Load input data
        logger.info(f"Loading input data from {input_path}")
        items = read_csv(input_path)
        logger.info(f"Loaded {len(items)} items")

        # Create client
        client = OpenRouterClient()

        try:
            # Get responses for all items
            results = asyncio.run(
                get_responses_batch(client, items, model, "Getting text responses")
            )

        finally:
            asyncio.run(client.close())

        # Sort results by dialog_id
        results.sort(key=lambda x: x.get("dialog_id", ""))

        # Save output
        write_jsonl(results, output_path)
        logger.info(f"Saved {len(results)} responses to {output_path}")

        # Log summary
        success_count = sum(1 for r in results if r.get("success"))
        logger.info(f"Successful responses: {success_count}/{len(results)}")

        # Log sample responses
        logger.info("Sample responses:")
        for i, result in enumerate(results[:3]):
            if result.get("success"):
                logger.info(
                    f"  [{result.get('dialog_id')}] {result.get('response', '')[:100]}..."
                )

        logger.info("=" * 60)
        logger.info("STEP 3 COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Step 3 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for Step 3."""
    parser = argparse.ArgumentParser(description="Step 3: Text-Only Baseline")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step1_rewritten.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "responses"
        / "step3_text_responses.jsonl",
        help="Path to output JSONL file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use for responses",
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
    success = run_step3(
        input_path=args.input, output_path=args.output, model=args.model
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
