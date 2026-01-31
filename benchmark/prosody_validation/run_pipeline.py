#!/usr/bin/env python3
"""
Prosody Validation Benchmark Pipeline Runner

A comprehensive benchmark for testing SLM prosody understanding.
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_level: str, log_file: Path = None):
    """Setup logging configuration."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(level=numeric_level, format=log_format, handlers=handlers)


def run_steps_sequentially(steps: list, config: dict = None):
    """Run multiple pipeline steps sequentially.

    Args:
        steps: List of step functions to run
        config: Optional configuration dictionary
    """
    logger = logging.getLogger(__name__)

    for i, step_func in enumerate(steps, 1):
        step_name = step_func.__name__.replace("_", " ").title()

        logger.info("=" * 60)
        logger.info(f"Running Step {i}: {step_name}")
        logger.info("=" * 60)

        try:
            if config:
                step_func(**config)
            else:
                step_func()

            logger.info(f"Step {i} completed successfully")

        except Exception as e:
            logger.error(f"Step {i} failed: {e}")
            raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Prosody Validation Benchmark Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run specific steps
  python run_pipeline.py --step 0 --step 1

  # Run all steps
  python run_pipeline.py --all

  # Run with custom settings
  python run_pipeline.py --all --device cuda --max-concurrent 50

  # Dry run (show what would run)
  python run_pipeline.py --all --dry-run
        """,
    )

    parser.add_argument(
        "--step",
        action="append",
        type=int,
        help="Run specific step (can be used multiple times)",
    )
    parser.add_argument("--all", action="store_true", help="Run all steps")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would run without executing"
    )

    # Configuration options
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for TTS/Whisper",
    )
    parser.add_argument(
        "--rewrite-model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Model for text rewriting",
    )
    parser.add_argument(
        "--text-model",
        type=str,
        default="openai/gpt-4o-mini",
        help="Model for text baseline",
    )
    parser.add_argument(
        "--e2e-model",
        type=str,
        default="google/gemini-2.5-flash",
        help="Model for E2E audio",
    )
    parser.add_argument(
        "--asr-model", type=str, default="base", help="Whisper model size"
    )
    parser.add_argument(
        "--max-concurrent", type=int, default=50, help="Max concurrent LLM calls"
    )
    parser.add_argument("--batch-size", type=int, default=10, help="Batch size for TTS")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )
    parser.add_argument("--log-file", type=Path, help="Log file path")

    args = parser.parse_args()

    # Validate arguments
    if not args.step and not args.all:
        parser.error("Must specify --step or --all")

    # Setup paths
    base_dir = Path(__file__).parent

    # Setup logging
    log_file = args.log_file or (
        base_dir / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    setup_logging(args.log_level, log_file)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("PROSODY VALIDATION BENCHMARK PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Log file: {log_file}")

    # Import steps
    sys.path.insert(0, str(base_dir))

    from steps.step0_merge_sample import run_step0
    from steps.step1_rewrite_text import run_step1
    from steps.step2_generate_speech import run_step2
    from steps.step3_text_baseline import run_step3
    from steps.step4_asr_pipeline import run_step4
    from steps.step5_e2e_audio import run_step5

    # Define step configurations
    step_configs = {
        0: {
            "annotation_path": base_dir
            / "data"
            / "(2000 samples) merged_output.xlsx - Annotation.csv",
            "dialogue_path": base_dir
            / "data"
            / "(2000 samples) merged_output.xlsx - Dialogue.tsv",
            "output_path": base_dir / "data" / "step0_sampled_200.csv",
            "n_samples": 200,
            "random_state": 42,
        },
        1: {
            "input_path": base_dir / "data" / "step0_sampled_200.csv",
            "output_path": base_dir / "data" / "step1_rewritten.csv",
            "model": args.rewrite_model,
            "max_concurrent": args.max_concurrent,
        },
        2: {
            "input_path": base_dir / "data" / "step1_rewritten.csv",
            "audio_dir": base_dir / "outputs" / "audio",
            "manifest_path": base_dir / "data" / "step2_audio_manifest.csv",
            "batch_size": args.batch_size,
            "device": args.device,
        },
        3: {
            "input_path": base_dir / "data" / "step1_rewritten.csv",
            "output_path": base_dir
            / "outputs"
            / "responses"
            / "step3_text_responses.jsonl",
            "model": args.text_model,
        },
        4: {
            "manifest_path": base_dir / "data" / "step2_audio_manifest.csv",
            "output_path": base_dir
            / "outputs"
            / "responses"
            / "step4_asr_responses.jsonl",
            "asr_model": args.asr_model,
            "llm_model": args.text_model,
        },
        5: {
            "manifest_path": base_dir / "data" / "step2_audio_manifest.csv",
            "output_path": base_dir
            / "outputs"
            / "responses"
            / "step5_e2e_responses.jsonl",
            "model": args.e2e_model,
        },
    }

    # Define step functions
    step_funcs = {
        0: run_step0,
        1: run_step1,
        2: run_step2,
        3: run_step3,
        4: run_step4,
        5: run_step5,
    }

    # Determine which steps to run
    if args.all:
        steps_to_run = list(range(6))
    else:
        steps_to_run = sorted(set(args.step))

    # Validate step numbers
    invalid_steps = [s for s in steps_to_run if s not in step_funcs]
    if invalid_steps:
        logger.error(f"Invalid step numbers: {invalid_steps}")
        sys.exit(1)

    logger.info(f"Steps to run: {steps_to_run}")

    if args.dry_run:
        logger.info("DRY RUN - No steps will be executed")
        for step_num in steps_to_run:
            step_name = step_funcs[step_num].__name__
            logger.info(f"  Step {step_num}: {step_name}")
            config = step_configs.get(step_num, {})
            for key, value in config.items():
                logger.info(f"    {key}: {value}")
        sys.exit(0)

    # Run steps
    try:
        for step_num in steps_to_run:
            step_func = step_funcs[step_num]
            config = step_configs.get(step_num, {})
            step_func(**config)

        logger.info("=" * 60)
        logger.info("ALL STEPS COMPLETED SUCCESSFULLY")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
