"""
Step 1: Rewrite text to be neutral/positive-leaning using LLM
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, read_csv, write_csv
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


# Rewrite prompt template
REWRITE_PROMPT = """Rewrite this utterance to be semantically NEUTRAL or POSITIVE-LEANING.
Remove emotional indicators (angry, frustrated, annoyed, excited, etc.).
Preserve factual content. Text alone should NOT convey the emotion.

Original: {original}
Emotion: {emotion}
Intent: {intent}

RULES:
- Output ONLY the rewritten sentence
- NO prefixes like "Rewritten:" or "Here is:"
- NO explanations or multiple options
- NO quotation marks around the output
- Just the plain rewritten text"""


def create_rewrite_prompt(original: str, emotion: str, intent: str) -> str:
    """Create the rewrite prompt for a single utterance.

    Args:
        original: Original utterance text
        emotion: Emotion label
        intent: Intent label

    Returns:
        Formatted prompt string
    """
    return REWRITE_PROMPT.format(
        original=original, emotion=emotion or "neutral", intent=intent or "inform"
    )


async def rewrite_single(
    client: OpenRouterClient, item: Dict[str, Any], model: str
) -> Dict[str, Any]:
    """Rewrite a single utterance.

    Args:
        client: OpenRouter client
        item: Dictionary with dialog_id, utterance, emotion, intent
        model: Model to use for rewriting

    Returns:
        Updated dictionary with rewritten_text
    """
    prompt = create_rewrite_prompt(
        item.get("utterance", ""),
        item.get("emotion", "neutral"),
        item.get("intent", "inform"),
    )

    try:
        rewritten = await client.chat_text(
            prompt=prompt, model=model, temperature=0.3, max_tokens=256
        )

        # Clean up the response - remove prefixes, quotes, and take first line only
        rewritten = rewritten.strip()

        # Remove common prefixes
        prefixes = [
            "rewritten:",
            "rewritten text:",
            "here is:",
            "here is the rewritten text:",
            "result:",
            "output:",
            "neutral version:",
            "positive version:",
            "option 1:",
            "1.",
            "1)",
            "- ",
            "* ",
        ]
        lower = rewritten.lower()
        for prefix in prefixes:
            if lower.startswith(prefix):
                rewritten = rewritten[len(prefix) :].strip()
                lower = rewritten.lower()

        # Remove quotes and extra whitespace
        rewritten = rewritten.strip('"').strip("'").strip()

        # If multiple lines, take only the first one (ignore explanations/options)
        if "\n" in rewritten:
            rewritten = rewritten.split("\n")[0].strip()

        return {**item, "rewritten_text": rewritten}

    except Exception as e:
        logger.error(f"Error rewriting {item.get('dialog_id')}: {e}")
        return {**item, "rewritten_text": f"ERROR: {str(e)}"}


async def rewrite_batch(
    client: OpenRouterClient,
    items: List[Dict[str, Any]],
    model: str,
    progress_desc: str = "Rewriting text",
) -> List[Dict[str, Any]]:
    """Rewrite a batch of utterances concurrently.

    Args:
        client: OpenRouter client
        items: List of dictionaries with utterance data
        model: Model to use
        progress_desc: Description for progress bar

    Returns:
        List of dictionaries with rewritten_text added
    """
    from tqdm.asyncio import tqdm_asyncio

    # Create tasks for all items
    tasks = [rewrite_single(client, item, model) for item in items]

    # Process with progress bar
    progress_bar = tqdm_asyncio(total=len(tasks), desc=progress_desc, unit="utterances")

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        progress_bar.update(1)

    progress_bar.close()

    return results


def run_step1(
    input_path: Path,
    output_path: Path,
    model: str = "google/gemini-2.5-flash",
    max_concurrent: int = 50,
) -> bool:
    """Execute Step 1: Rewrite text to be neutral.

    Args:
        input_path: Path to step0_sampled_200.csv
        output_path: Path to save step1_rewritten.csv
        model: Model to use for rewriting
        max_concurrent: Maximum concurrent API calls

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 1: Rewrite Text to Neutral/Positive")
    logger.info("=" * 60)

    try:
        # Load input data
        logger.info(f"Loading input data from {input_path}")
        items = read_csv(input_path)
        logger.info(f"Loaded {len(items)} items")

        # Update concurrency limit
        config = get_config()
        original_max = config.MAX_CONCURRENT_LLM
        config.MAX_CONCURRENT_LLM = max_concurrent

        # Create client
        client = OpenRouterClient()

        async def run_rewrite():
            """Run rewrite batch and cleanup in same event loop."""
            try:
                return await rewrite_batch(
                    client, items, model, "Rewriting utterances to neutral form"
                )
            finally:
                await client.close()

        results = asyncio.run(run_rewrite())
        config.MAX_CONCURRENT_LLM = original_max

        # Sort results by dialog_id to maintain order
        results.sort(key=lambda x: x.get("dialog_id", ""))

        # Save output
        if results:
            fieldnames = list(results[0].keys())
            write_csv(results, output_path, fieldnames)
            logger.info(f"Saved {len(results)} rewritten items to {output_path}")
        else:
            logger.warning("No results to save")
            return False

        # Log summary
        success_count = sum(
            1 for r in results if not r.get("rewritten_text", "").startswith("ERROR")
        )
        logger.info(f"Successfully rewritten: {success_count}/{len(results)}")

        logger.info("=" * 60)
        logger.info("STEP 1 COMPLETED SUCCESSFULLY")
        logger.info(f"Output: {output_path}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Step 1 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for Step 1."""
    parser = argparse.ArgumentParser(description="Step 1: Rewrite Text to Neutral")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step0_sampled_200.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step1_rewritten.csv",
        help="Path to output CSV file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Model to use for rewriting",
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=50, help="Maximum concurrent API calls"
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
    success = run_step1(
        input_path=args.input,
        output_path=args.output,
        model=args.model,
        max_concurrent=args.max_concurrent,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
