#!/usr/bin/env python3
"""CLI entry point for Amy-LM preprocessing pipeline.

This script provides a command-line interface for preprocessing audio datasets
through FACodec encoder, extracting content, prosody, and timbre codebook indices.

Usage:
    python scripts/preprocess.py \
        --dataset huggingface/dataset-name \
        --split train \
        --output-repo org/processed-dataset \
        [--max-samples 1000] \
        [--device cuda] \
        [--output-dir data/processed] \
        [--no-push]

Example:
    # Process MUSTARD++ dataset
    python scripts/preprocess.py \
        --dataset hungphongtrn/mustard_plus_plus \
        --split train \
        --output-repo hungphongtrn/mustard_facodec
"""

import argparse
import sys
import signal
from pathlib import Path
from typing import Optional

import numpy as np

# Add src to path for imports
def _setup_path():
    """Add src directory to Python path."""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

_setup_path()

from preprocessing import FACodecEncoder, ProcessingSummary, generate_report
from preprocessing.dataset_processor import DatasetProcessor


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="preprocess",
        description="Preprocess audio datasets through FACodec encoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full dataset and push to hub
  python scripts/preprocess.py --dataset user/dataset --split train --output-repo org/output
  
  # Process only 100 samples locally for testing
  python scripts/preprocess.py --dataset user/dataset --split train --output-repo org/output \\
      --max-samples 100 --no-push
  
  # Use GPU and custom output directory
  python scripts/preprocess.py --dataset user/dataset --split train --output-repo org/output \\
      --device cuda --output-dir /mnt/data/processed
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Hugging Face dataset name (e.g., 'huggingface/dataset-name')"
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Dataset split to process (e.g., 'train', 'validation', 'test')"
    )
    parser.add_argument(
        "--output-repo",
        type=str,
        required=True,
        help="Hugging Face repository to push processed dataset to (e.g., 'org/dataset-name')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process (for testing). Default: process all samples"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on. Default: cpu"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Local directory to save processed dataset. Default: data/processed"
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Skip pushing to Hugging Face Hub (save locally only)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Dataset config/subset name (e.g., 'emns' for CAMEO). Only needed for multi-config datasets."
    )
    
    return parser


def print_summary(summary: ProcessingSummary, report_path: Path) -> None:
    """Print a formatted summary of preprocessing results.
    
    Args:
        summary: ProcessingSummary containing statistics
        report_path: Path where JSON report was saved
    """
    stats = summary.to_dict()
    
    print("\n" + "=" * 60)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 60)
    
    # Sample counts
    total_attempted = stats["total_processed"] + stats["total_failed"]
    success_rate = (stats["total_processed"] / total_attempted * 100) if total_attempted > 0 else 0
    
    print(f"\n✓ Successfully processed: {stats['total_processed']} samples")
    if stats["total_failed"] > 0:
        print(f"✗ Failed: {stats['total_failed']} samples")
    print(f"📈 Success rate: {success_rate:.1f}%")
    
    # Frame statistics
    print(f"\n🎵 Frame Statistics:")
    print(f"   Average content frames: {stats['avg_content_frames']:.1f}")
    print(f"   Average prosody frames: {stats['avg_prosody_frames']:.1f}")
    print(f"   Average acoustic frames: {stats['avg_acoustic_frames']:.1f}")
    
    # Duration statistics
    print(f"\n⏱️  Duration Statistics:")
    print(f"   Average duration: {stats['avg_duration_sec']:.2f}s")
    print(f"\n   Duration Distribution:")
    for bin_name, count in stats["duration_histogram"].items():
        if count > 0:
            percentage = (count / stats["total_processed"] * 100) if stats["total_processed"] > 0 else 0
            bar = "█" * int(percentage / 5)
            print(f"      {bin_name:6}: {count:4} ({percentage:5.1f}%) {bar}")
    
    # Failed samples summary
    if stats["failed_samples"]:
        print(f"\n⚠️  Failed Samples (showing first 5):")
        for failure in stats["failed_samples"][:5]:
            print(f"   - {failure['id']}: {failure['error'][:60]}{'...' if len(failure['error']) > 60 else ''}")
        if len(stats["failed_samples"]) > 5:
            print(f"   ... and {len(stats['failed_samples']) - 5} more")
    
    print(f"\n📝 Report saved to: {report_path}")
    print("=" * 60)


def setup_signal_handlers() -> None:
    """Set up signal handlers for graceful interruption."""
    def signal_handler(signum, frame):
        print("\n\n⚠️  Interrupted by user. Exiting gracefully...")
        sys.exit(130)  # Standard exit code for Ctrl+C
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def main(
    dataset: str,
    split: str,
    output_repo: str,
    max_samples: Optional[int],
    device: str,
    output_dir: str,
    no_push: bool,
    config: Optional[str] = None
) -> int:
    """Main preprocessing pipeline.
    
    Args:
        dataset: Hugging Face dataset name
        split: Dataset split to process
        output_repo: HF repository to push to
        max_samples: Maximum samples to process (None for all)
        device: Device to run on
        output_dir: Local output directory
        no_push: If True, skip pushing to HF Hub
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    print("🚀 Amy-LM Preprocessing Pipeline")
    print(f"   Dataset: {dataset}/{split}")
    print(f"   Device: {device}")
    if max_samples:
        print(f"   Max samples: {max_samples}")
    print(f"   Output: {output_dir}")
    if not no_push:
        print(f"   Target repo: {output_repo}")
    else:
        print("   Push to hub: disabled (--no-push)")
    print()
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize FACodecEncoder
    print("🎛️  Initializing FACodec encoder...")
    try:
        encoder = FACodecEncoder(device=device)
        if encoder._mock:
            print("   ⚠️  Using mock encoder (Amphion not available)")
        else:
            print("   ✓ Using real FACodec encoder")
    except Exception as e:
        print(f"   ✗ Failed to initialize encoder: {e}", file=sys.stderr)
        return 1
    
    # Create DatasetProcessor
    print("📂 Setting up dataset processor...")
    try:
        processor = DatasetProcessor(encoder, output_path, device)
        print(f"   ✓ Output directory: {output_path.absolute()}")
    except ImportError as e:
        print(f"   ✗ {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"   ✗ Failed to initialize processor: {e}", file=sys.stderr)
        return 1
    
    # Process dataset
    print(f"\n🔧 Processing dataset: {dataset}/{split}")
    print("-" * 60)
    
    try:
        processed_dataset = processor.process_dataset(
            dataset_name=dataset,
            split=split,
            max_samples=max_samples,
            config=config
        )
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Exiting gracefully...")
        return 130
    except Exception as e:
        print(f"\n✗ Processing failed: {e}", file=sys.stderr)
        return 1
    
    print("-" * 60)
    print(f"✓ Processed {len(processed_dataset)} samples")
    
    # Create processing summary from results
    print("\n📊 Generating processing summary...")
    summary = ProcessingSummary()
    
    # Extract statistics from processed dataset
    for sample in processed_dataset:
        audio_field = sample.get("audio", {})
        if isinstance(audio_field, dict):
            audio_array = audio_field.get("array", [])
            sampling_rate = audio_field.get("sampling_rate", 16000)
        else:
            try:
                audio_array = audio_field["array"]
                sampling_rate = audio_field["sampling_rate"]
            except Exception:
                audio_array = np.array([])
                sampling_rate = 16000
        duration_sec = len(audio_array) / sampling_rate if sampling_rate > 0 and hasattr(audio_array, '__len__') else 0.0
        
        # Get frame counts from codebook indices
        # Note: content and acoustic are now nested [codebooks, frames]
        content_indices = sample.get("content_codebooks_idx", [])
        prosody_indices = sample.get("prosody_codebooks_idx", [])
        acoustic_indices = sample.get("acoustic_codebooks_idx", [])

        # For nested lists, get temporal length from first codebook
        content_frames = len(content_indices[0]) if content_indices else 0
        prosody_frames = len(prosody_indices)
        acoustic_frames = len(acoustic_indices[0]) if acoustic_indices else 0

        summary.add_processed(
            content_frames=content_frames,
            prosody_frames=prosody_frames,
            acoustic_frames=acoustic_frames,
            duration_sec=duration_sec
        )
    
    # Note: failures are already logged during processing
    # We don't track them separately here since they're printed by DatasetProcessor
    
    # Save dataset locally
    print(f"\n💾 Saving dataset locally...")
    try:
        saved_path = processor.save(processed_dataset, output_repo, split=split)
        print(f"   ✓ Saved to: {saved_path}")
    except Exception as e:
        print(f"   ✗ Failed to save dataset: {e}", file=sys.stderr)
        return 1
    
    # Generate report
    report_dir = output_path / output_repo / "reports"
    report_path = report_dir / f"{split}_report.json"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        generate_report(summary, report_path)
        print(f"   ✓ Report saved to: {report_path}")
    except Exception as e:
        print(f"   ⚠️  Failed to save report: {e}")
    
    # Push to HF Hub (unless --no-push)
    if not no_push:
        print(f"\n☁️  Pushing to Hugging Face Hub: {output_repo}")
        try:
            processor.push_to_hub(processed_dataset, output_repo)
        except Exception as e:
            print(f"   ✗ Failed to push to hub: {e}", file=sys.stderr)
            print("   ⚠️  Dataset was saved locally. You can push manually later.")
            # Don't return error - local save succeeded
    else:
        print("\n☁️  Skipping push to HF Hub (--no-push specified)")
    
    # Print summary
    print_summary(summary, report_path)
    
    return 0


def cli_entry_point() -> int:
    """CLI entry point - parses arguments and calls main().
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    setup_signal_handlers()
    
    parser = create_parser()
    args = parser.parse_args()
    
    return main(
        dataset=args.dataset,
        split=args.split,
        output_repo=args.output_repo,
        max_samples=args.max_samples,
        device=args.device,
        output_dir=args.output_dir,
        no_push=args.no_push,
        config=args.config
    )


if __name__ == "__main__":
    sys.exit(cli_entry_point())
