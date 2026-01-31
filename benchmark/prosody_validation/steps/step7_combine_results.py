"""
Step 7: Combine all evaluation results into a single CSV

Merges text-only, ASR, and E2E responses with their evaluations
into a comprehensive comparison CSV.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_config
from src.utils import setup_logging, read_jsonl, write_csv

logger = logging.getLogger(__name__)


def load_and_index(
    file_path: Path, key_field: str = "dialog_id"
) -> Dict[str, Dict[str, Any]]:
    """Load JSONL file and index by key field.

    Args:
        file_path: Path to JSONL file
        key_field: Field to use as index key

    Returns:
        Dictionary indexed by key field
    """
    if not file_path.exists():
        logger.warning(f"File not found: {file_path}")
        return {}

    items = read_jsonl(file_path)
    return {item.get(key_field, ""): item for item in items if item.get(key_field)}


def combine_results(
    text_responses_path: Path,
    asr_responses_path: Path,
    e2e_responses_path: Path,
    text_evaluations_path: Path,
    asr_evaluations_path: Path,
    e2e_evaluations_path: Path,
    output_path: Path,
) -> bool:
    """Combine all results into a single CSV.

    Args:
        text_responses_path: Path to text responses JSONL
        asr_responses_path: Path to ASR responses JSONL
        e2e_responses_path: Path to E2E responses JSONL
        text_evaluations_path: Path to text evaluations JSONL
        asr_evaluations_path: Path to ASR evaluations JSONL
        e2e_evaluations_path: Path to E2E evaluations JSONL
        output_path: Path to save combined CSV

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 7: Combining Results")
    logger.info("=" * 60)

    try:
        # Load all data
        logger.info("Loading response data...")
        text_responses = load_and_index(text_responses_path)
        asr_responses = load_and_index(asr_responses_path)
        e2e_responses = load_and_index(e2e_responses_path)

        logger.info("Loading evaluation data...")
        text_evaluations = load_and_index(text_evaluations_path)
        asr_evaluations = load_and_index(asr_evaluations_path)
        e2e_evaluations = load_and_index(e2e_evaluations_path)

        # Get all unique dialog_ids
        all_ids = set()
        all_ids.update(text_responses.keys())
        all_ids.update(asr_responses.keys())
        all_ids.update(e2e_responses.keys())
        all_ids.update(text_evaluations.keys())
        all_ids.update(asr_evaluations.keys())
        all_ids.update(e2e_evaluations.keys())

        logger.info(f"Found {len(all_ids)} unique dialog IDs")

        if not all_ids:
            logger.warning("No data to combine")
            return False

        # Build combined records
        combined_records = []

        for dialog_id in sorted(all_ids):
            # Get data from each source
            text_resp = text_responses.get(dialog_id, {})
            asr_resp = asr_responses.get(dialog_id, {})
            e2e_resp = e2e_responses.get(dialog_id, {})
            text_eval = text_evaluations.get(dialog_id, {})
            asr_eval = asr_evaluations.get(dialog_id, {})
            e2e_eval = e2e_evaluations.get(dialog_id, {})

            # Build record starting with original columns
            record = {}

            # Add original metadata (from any available source)
            source = (
                text_resp or asr_resp or e2e_resp or text_eval or asr_eval or e2e_eval
            )
            if source:
                record["dialog_id"] = dialog_id
                record["original_utterance"] = source.get("original_utterance", "")
                record["rewritten_text"] = source.get("rewritten_text", "")
                record["emotion"] = source.get("emotion", "")
                record["intent"] = source.get("intent", "")
                record["speech_act"] = source.get("speech_act", "")

                # Add any other original columns that exist
                for key in source.keys():
                    if key not in record and not key.startswith(
                        ("response", "asr_", "e2e_", "evaluation_")
                    ):
                        record[key] = source.get(key, "")

            # Add responses
            record["text_only_response"] = text_resp.get("response", "")
            record["whisper_asr_response"] = asr_resp.get("asr_response", "")
            record["e2e_response"] = e2e_resp.get("e2e_response", "")

            # Add evaluations (rationale)
            record["text_only_evaluation"] = text_eval.get("evaluation_rationale", "")
            record["whisper_asr_evaluation"] = asr_eval.get("evaluation_rationale", "")
            record["e2e_evaluation"] = e2e_eval.get("evaluation_rationale", "")

            # Add approval flags as 0/1
            text_approved = text_eval.get("evaluation_approved")
            asr_approved = asr_eval.get("evaluation_approved")
            e2e_approved = e2e_eval.get("evaluation_approved")

            record["text_only_approved"] = (
                1 if text_approved is True else 0 if text_approved is False else ""
            )
            record["whisper_asr_approved"] = (
                1 if asr_approved is True else 0 if asr_approved is False else ""
            )
            record["e2e_approved"] = (
                1 if e2e_approved is True else 0 if e2e_approved is False else ""
            )

            combined_records.append(record)

        # Write CSV
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(combined_records, output_path)

        logger.info(f"Saved {len(combined_records)} records to {output_path}")

        # Log summary statistics
        text_approved_count = sum(
            1 for r in combined_records if r.get("text_only_approved") == 1
        )
        asr_approved_count = sum(
            1 for r in combined_records if r.get("whisper_asr_approved") == 1
        )
        e2e_approved_count = sum(
            1 for r in combined_records if r.get("e2e_approved") == 1
        )

        total_evaluated = len(
            [r for r in combined_records if r.get("text_only_approved") != ""]
        )

        logger.info("=" * 60)
        logger.info("COMBINATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total records: {len(combined_records)}")
        logger.info(f"Text-Only approved: {text_approved_count}/{total_evaluated}")
        logger.info(f"ASR approved: {asr_approved_count}/{total_evaluated}")
        logger.info(f"E2E approved: {e2e_approved_count}/{total_evaluated}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Combination failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for Step 7."""
    parser = argparse.ArgumentParser(description="Step 7: Combine Results into CSV")
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
        "--text-evaluations",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "evaluations"
        / "step6_text_evaluations.jsonl",
        help="Path to text evaluations JSONL",
    )
    parser.add_argument(
        "--asr-evaluations",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "evaluations"
        / "step6_asr_evaluations.jsonl",
        help="Path to ASR evaluations JSONL",
    )
    parser.add_argument(
        "--e2e-evaluations",
        type=Path,
        default=Path(__file__).parent.parent
        / "outputs"
        / "evaluations"
        / "step6_e2e_evaluations.jsonl",
        help="Path to E2E evaluations JSONL",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "combined_results.csv",
        help="Path to save combined CSV",
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

    # Run combination
    success = combine_results(
        text_responses_path=args.text_responses,
        asr_responses_path=args.asr_responses,
        e2e_responses_path=args.e2e_responses,
        text_evaluations_path=args.text_evaluations,
        asr_evaluations_path=args.asr_evaluations,
        e2e_evaluations_path=args.e2e_evaluations,
        output_path=args.output,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
