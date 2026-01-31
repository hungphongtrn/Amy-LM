"""
Step 2: Generate speech with prosody using Qwen3-TTS
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
from src.utils import setup_logging, read_csv, write_csv
from src.qwen3_tts_wrapper import Qwen3TTSWrapper

logger = logging.getLogger(__name__)


def run_step2(
    input_path: Path,
    audio_dir: Path,
    manifest_path: Path,
    model_path: str = None,
    batch_size: int = 10,
    device: str = "cuda",
) -> bool:
    """Execute Step 2: Generate speech with prosody.

    Args:
        input_path: Path to step1_rewritten.csv
        audio_dir: Directory to save generated audio files
        manifest_path: Path to save the audio manifest CSV
        model_path: Path to Qwen3-TTS model repository
        batch_size: Batch size for processing
        device: Device to run TTS on

    Returns:
        True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("STEP 2: Generate Prosody-Guided Speech")
    logger.info("=" * 60)

    try:
        # Load input data
        logger.info(f"Loading input data from {input_path}")
        items = read_csv(input_path)
        logger.info(f"Loaded {len(items)} items")

        # Create TTS wrapper
        wrapper = Qwen3TTSWrapper(
            repo_path=model_path, device=device, batch_size=batch_size
        )

        # Load model
        if not asyncio.run(wrapper.load_model()):
            logger.error("Failed to load Qwen3-TTS model")
            return False

        # Prepare items for TTS
        tts_items = []
        for item in items:
            tts_items.append(
                {
                    "dialog_id": item["dialog_id"],
                    "text": item.get("rewritten_text", item.get("utterance", "")),
                    "emotion": item.get("emotion", "neutral"),
                    "speech_act": item.get("speech_act", "statement"),
                    "intent": item.get("intent", "inform"),
                }
            )

        # Generate speech
        logger.info(f"Generating speech for {len(tts_items)} items...")
        results = asyncio.run(
            wrapper.generate_batch(
                tts_items, audio_dir, "Generating prosody-guided speech"
            )
        )

        # Unload model
        wrapper.unload_model()

        # Create manifest
        manifest_items = []
        for item in items:
            # Find corresponding result
            dialog_id = item["dialog_id"]
            result = next((r for r in results if r["dialog_id"] == dialog_id), None)

            if result and result.get("success"):
                manifest_items.append(
                    {
                        "dialog_id": dialog_id,
                        "audio_path": result["audio_path"],
                        "text": item.get("rewritten_text", item.get("utterance", "")),
                        "emotion": item.get("emotion", "neutral"),
                        "speech_act": item.get("speech_act", "statement"),
                        "intent": item.get("intent", "inform"),
                        "success": True,
                    }
                )
            else:
                manifest_items.append(
                    {
                        "dialog_id": dialog_id,
                        "audio_path": None,
                        "text": item.get("rewritten_text", item.get("utterance", "")),
                        "emotion": item.get("emotion", "neutral"),
                        "speech_act": item.get("speech_act", "statement"),
                        "intent": item.get("intent", "inform"),
                        "success": False,
                    }
                )

        # Save manifest
        if manifest_items:
            fieldnames = list(manifest_items[0].keys())
            write_csv(manifest_items, manifest_path, fieldnames)
            logger.info(f"Saved manifest to {manifest_path}")

        # Log summary
        success_count = sum(1 for m in manifest_items if m.get("success"))
        logger.info(
            f"Successfully generated: {success_count}/{len(manifest_items)} audio files"
        )

        logger.info("=" * 60)
        logger.info("STEP 2 COMPLETED SUCCESSFULLY")
        logger.info(f"Audio directory: {audio_dir}")
        logger.info(f"Manifest: {manifest_path}")
        logger.info("=" * 60)

        return True

    except Exception as e:
        logger.error(f"Step 2 failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point for Step 2."""
    parser = argparse.ArgumentParser(description="Step 2: Generate Speech with Prosody")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step1_rewritten.csv",
        help="Path to input CSV file",
    )
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path(__file__).parent.parent / "outputs" / "audio",
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "step2_audio_manifest.csv",
        help="Path to save audio manifest",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Qwen3-TTS model repository",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10, help="Batch size for processing"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run TTS on",
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
    success = run_step2(
        input_path=args.input,
        audio_dir=args.audio_dir,
        manifest_path=args.manifest,
        model_path=args.model_path,
        batch_size=args.batch_size,
        device=args.device,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
