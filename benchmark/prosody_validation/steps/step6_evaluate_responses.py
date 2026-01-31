"""
Step 6: Evaluate responses using LLM as judge

Evaluates text-only, ASR, and E2E responses for appropriateness
given the emotion, intent, and speech-act context.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, read_jsonl, write_jsonl, save_json
from src.openrouter_client import OpenRouterClient

logger = logging.getLogger(__name__)


class Appropriate(BaseModel):
    """Structured output for response appropriateness evaluation."""

    approved: bool = Field(
        description="Whether the response appropriately addresses the user's emotion, intent, and speech act"
    )
    evaluation_rationale: str = Field(
        description="Brief explanation of why the response is approved or rejected, referencing emotion, intent, and speech act"
    )


# Evaluation prompt template
EVALUATION_PROMPT = """You are evaluating whether an AI assistant's response appropriately addresses a user's message.

Context:
- User's original message: "{original_utterance}"
- Emotion detected: {emotion}
- Intent: {intent}
- Speech act: {speech_act}

AI Assistant's response: "{response}"

Evaluate if the response appropriately acknowledges and addresses:
1. The user's emotional state ({emotion})
2. The user's underlying intent ({intent})
3. The speech act type ({speech_act})

Does this response appropriately address the user's communication? Answer with only true or false."""


def create_evaluation_prompt(
    original_utterance: str,
    response: str,
    emotion: str,
    intent: str,
    speech_act: str,
) -> str:
    """Create the evaluation prompt for a single response.

    Args:
        original_utterance: The original user utterance
        response: The AI assistant's response
        emotion: Detected emotion
        intent: Detected intent
        speech_act: Detected speech act

    Returns:
        Formatted evaluation prompt
    """
    return EVALUATION_PROMPT.format(
        original_utterance=original_utterance,
        response=response,
        emotion=emotion,
        intent=intent,
        speech_act=speech_act,
    )


async def evaluate_single(
    client: OpenRouterClient,
    item: Dict[str, Any],
    model: str,
    response_key: str,
) -> Dict[str, Any]:
    """Evaluate a single response for appropriateness.

    Args:
        client: OpenRouter client
        item: Dictionary with response data
        model: Model to use for evaluation
        response_key: Key to extract the response (e.g., 'response', 'asr_response', 'e2e_response')

    Returns:
        Dictionary with evaluation results
    """
    response = item.get(response_key, "")

    # Skip if no response
    if not response:
        return {
            **item,
            "evaluation_approved": None,
            "evaluation_error": "No response to evaluate",
        }

    # Get metadata (handle different field names across steps)
    original_utterance = item.get("original_utterance", "")
    emotion = item.get("emotion", "")
    intent = item.get("intent", "")
    speech_act = item.get("speech_act", "")

    try:
        prompt = create_evaluation_prompt(
            original_utterance=original_utterance,
            response=response,
            emotion=emotion,
            intent=intent,
            speech_act=speech_act,
        )

        # Use structured output for evaluation
        evaluation_result = await client.chat_structured(
            prompt=prompt,
            model=model,
            response_format=Appropriate,
            temperature=0.0,  # Deterministic for evaluation
            max_tokens=100,
        )

        return {
            **item,
            "evaluation_approved": evaluation_result.approved,
            "evaluation_error": None,
        }

    except Exception as e:
        logger.error(f"Error evaluating response for {item.get('dialog_id')}: {e}")
        return {
            **item,
            "evaluation_approved": None,
            "evaluation_error": str(e),
        }


async def evaluate_batch(
    client: OpenRouterClient,
    items: List[Dict[str, Any]],
    model: str,
    response_key: str,
    progress_desc: str = "Evaluating responses",
) -> List[Dict[str, Any]]:
    """Evaluate a batch of responses concurrently.

    Args:
        client: OpenRouter client
        items: List of response dictionaries
        model: Model to use for evaluation
        response_key: Key to extract the response
        progress_desc: Description for progress bar

    Returns:
        List of evaluation results
    """
    from tqdm.asyncio import tqdm_asyncio

    # Create tasks for all items
    tasks = [evaluate_single(client, item, model, response_key) for item in items]

    # Process with progress bar
    progress_bar = tqdm_asyncio(total=len(tasks), desc=progress_desc, unit="evals")

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
        progress_bar.update(1)

    progress_bar.close()

    return results


def calculate_accuracy(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate accuracy metrics from evaluation results.

    Args:
        results: List of evaluation results

    Returns:
        Dictionary with accuracy metrics
    """
    total = len(results)

    # Count successful evaluations
    evaluated = [r for r in results if r.get("evaluation_approved") is not None]
    approved = sum(1 for r in evaluated if r.get("evaluation_approved") is True)
    rejected = sum(1 for r in evaluated if r.get("evaluation_approved") is False)
    errors = total - len(evaluated)

    if len(evaluated) > 0:
        accuracy = approved / len(evaluated)
    else:
        accuracy = 0.0

    return {
        "total": total,
        "evaluated": len(evaluated),
        "approved": approved,
        "rejected": rejected,
        "errors": errors,
        "accuracy": accuracy,
        "approval_rate": approved / len(evaluated) if len(evaluated) > 0 else 0.0,
    }


async def run_evaluation(
    client: OpenRouterClient,
    input_path: Path,
    output_path: Path,
    model: str,
    response_key: str,
    pipeline_name: str,
) -> bool:
    """Execute evaluation for a single pipeline.

    Args:
        client: OpenRouter client
        input_path: Path to input JSONL file
        output_path: Path to save evaluation results
        model: Model to use for evaluation
        response_key: Key to extract the response
        pipeline_name: Name of the pipeline for logging

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Evaluating {pipeline_name} pipeline...")

    try:
        # Load input data
        logger.info(f"Loading responses from {input_path}")
        items = read_jsonl(input_path)
        logger.info(f"Loaded {len(items)} items")

        if not items:
            logger.warning(f"No items to evaluate for {pipeline_name}")
            return True

        # Filter to successful responses only
        success_key = "success" if response_key != "asr_response" else "asr_success"
        successful_items = [item for item in items if item.get(success_key, False)]
        logger.info(f"Evaluating {len(successful_items)} successful responses")

        if not successful_items:
            logger.warning(f"No successful responses to evaluate for {pipeline_name}")
            # Write empty results
            write_jsonl([], output_path)
            return True

        try:
            # Evaluate all items
            results = await evaluate_batch(
                client,
                successful_items,
                model,
                response_key,
                f"Evaluating {pipeline_name}",
            )

        except Exception as e:
            logger.error(f"Error during batch evaluation: {e}")
            raise

        # Sort results by dialog_id
        results.sort(key=lambda x: x.get("dialog_id", ""))

        # Save output
        write_jsonl(results, output_path)
        logger.info(f"Saved {len(results)} evaluations to {output_path}")

        # Calculate and log accuracy
        metrics = calculate_accuracy(results)
        logger.info(f"{pipeline_name} Evaluation Metrics:")
        logger.info(f"  Total: {metrics['total']}")
        logger.info(f"  Evaluated: {metrics['evaluated']}")
        logger.info(f"  Approved: {metrics['approved']}")
        logger.info(f"  Rejected: {metrics['rejected']}")
        logger.info(f"  Errors: {metrics['errors']}")
        logger.info(f"  Accuracy: {metrics['accuracy']:.2%}")

        return True

    except Exception as e:
        logger.error(f"Evaluation failed for {pipeline_name}: {e}")
        import traceback

        traceback.print_exc()
        return False


async def run_all_evaluations(
    text_responses_path: Path,
    asr_responses_path: Path,
    e2e_responses_path: Path,
    output_dir: Path,
    model: str,
) -> bool:
    """Execute evaluation for all three pipelines.

    Args:
        text_responses_path: Path to step3_text_responses.jsonl
        asr_responses_path: Path to step4_asr_responses.jsonl
        e2e_responses_path: Path to step5_e2e_responses.jsonl
        output_dir: Directory to save evaluation results
        model: Model to use for evaluation

    Returns:
        True if all evaluations successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 6: Response Evaluation")
    logger.info("=" * 60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define pipelines to evaluate
    pipelines = [
        {
            "name": "Text-Only",
            "input": text_responses_path,
            "output": output_dir / "step6_text_evaluations.jsonl",
            "response_key": "response",
        },
        {
            "name": "ASR",
            "input": asr_responses_path,
            "output": output_dir / "step6_asr_evaluations.jsonl",
            "response_key": "asr_response",
        },
        {
            "name": "E2E",
            "input": e2e_responses_path,
            "output": output_dir / "step6_e2e_evaluations.jsonl",
            "response_key": "e2e_response",
        },
    ]

    all_success = True
    all_metrics = {}

    # Create client once for all evaluations
    client = OpenRouterClient()

    try:
        for pipeline in pipelines:
            if not pipeline["input"].exists():
                logger.warning(f"Input file not found: {pipeline['input']}")
                all_success = False
                continue

            success = await run_evaluation(
                client=client,
                input_path=pipeline["input"],
                output_path=pipeline["output"],
                model=model,
                response_key=pipeline["response_key"],
                pipeline_name=pipeline["name"],
            )

            if success:
                # Load results and calculate metrics
                results = read_jsonl(pipeline["output"])
                metrics = calculate_accuracy(results)
                all_metrics[pipeline["name"]] = metrics
            else:
                all_success = False

    finally:
        await client.close()

    # Save summary report
    summary_path = output_dir / "evaluation_summary.json"
    save_json(all_metrics, summary_path)
    logger.info(f"Saved evaluation summary to {summary_path}")

    # Print final comparison
    logger.info("=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    for name, metrics in all_metrics.items():
        logger.info(f"{name}:")
        logger.info(
            f"  Accuracy: {metrics['accuracy']:.2%} ({metrics['approved']}/{metrics['evaluated']})"
        )
    logger.info("=" * 60)

    return all_success


def main():
    """Main entry point for Step 6."""
    parser = argparse.ArgumentParser(description="Step 6: Response Evaluation")
    parser.add_argument(
        "--text-responses",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "responses"
        / "step3_text_responses.jsonl",
        help="Path to text responses JSONL",
    )
    parser.add_argument(
        "--asr-responses",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "responses"
        / "step4_asr_responses.jsonl",
        help="Path to ASR responses JSONL",
    )
    parser.add_argument(
        "--e2e-responses",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "responses"
        / "step5_e2e_responses.jsonl",
        help="Path to E2E responses JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "evaluations",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model to use for evaluation",
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

    # Run all evaluations with a single event loop
    success = asyncio.run(
        run_all_evaluations(
            text_responses_path=args.text_responses,
            asr_responses_path=args.asr_responses,
            e2e_responses_path=args.e2e_responses,
            output_dir=args.output_dir,
            model=args.model,
        )
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
