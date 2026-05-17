#!/usr/bin/env python3
"""Prepare MUStARD dataset: download videos, extract audio, match annotations, build HF dataset.

Usage:
    uv run python scripts/prepare_mustard_dataset.py [--skip-download] [--skip-audio-extract]

Output: data/mustard_dataset/ — a HuggingFace dataset with columns:
    audio, sarcasm
"""

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def download_videos(raw_dir: Path):
    """Download and extract MUStARD raw video zip."""
    zip_path = raw_dir / "mmsd_raw_data.zip"
    url = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"

    if zip_path.exists():
        print(f"  Zip already exists: {zip_path}")
    else:
        print(f"  Downloading {url} ...")
        subprocess.run(
            ["wget", "-O", str(zip_path), url],
            check=True,
        )

    # The zip extracts context_final/ and utterances_final/ directly into raw_dir
    context_dir = raw_dir / "context_final"
    if context_dir.exists():
        print(f"  Already extracted to {context_dir}")
    else:
        print(f"  Extracting to {raw_dir} ...")
        subprocess.run(["unzip", "-q", str(zip_path), "-d", str(raw_dir)], check=True)

    # Use context_final (includes surrounding context, not just utterance)
    return raw_dir / "context_final"


def _extract_utterance_number(uid: str) -> str:
    """Extract the utterance number from a MUStARD++ key.

    Keys can be:
      - "1_S01E01_128_u" (show_seasonEp_utterance_suffix) -> "128"
      - "2_60_u" (show_utterance_suffix) -> "60"
      - "2_60" (show_utterance) -> "60"
    """
    parts = uid.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return uid


def find_video_for_utterance(video_dir: Path, show: str, utterance_id: str) -> Path | None:
    """Find video file matching a MUStARD utterance.

    Video files follow the pattern {show}_{utterance}_c.mp4 (context)
    or {show}_{utterance}_t.mp4 (target).

    'show' can be either a numeric ID (extracted from KEY) or text name
    (from SHOW column). We use it directly and also try the numeric prefix
    of the utterance_id.
    """
    utt_num = _extract_utterance_number(utterance_id)

    # Try with the provided show, and also with the numeric prefix of KEY
    show_candidates = [show]
    key_prefix = utterance_id.split("_")[0]
    if key_prefix != show:
        show_candidates.append(key_prefix)

    candidates = []
    for sc in show_candidates:
        candidates.extend([
            video_dir / f"{sc}_{utt_num}_c.mp4",
            video_dir / f"{sc}_{utt_num}_t.mp4",
            video_dir / f"{sc}_{utt_num}.mp4",
        ])

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # Fallback: glob search
    for sc in show_candidates:
        glob_pattern = f"{sc}_{utt_num}.*"
        matches = list(video_dir.glob(glob_pattern))
        if matches:
            return matches[0]

    return None


def extract_audio(video_path: Path, output_wav: Path) -> bool:
    """Extract mono 16kHz WAV from video using ffmpeg."""
    if output_wav.exists():
        return True

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-i", str(video_path),
                "-ac", "1",
                "-ar", "16000",
                "-sample_fmt", "s16",
                "-y",
                "-loglevel", "error",
                str(output_wav),
            ],
            check=True,
            timeout=30,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"    WARNING: ffmpeg failed for {video_path}: {e}", file=sys.stderr)
        return False


def load_annotations(csv_path: str) -> list[dict]:
    """Load MUStARD CSV, filter to target utterances (rows with sarcasm label)."""
    annotations = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sarcasm = row.get("Sarcasm", "").strip()
            if sarcasm == "":
                continue
            annotations.append({
                "id": row["KEY"].strip(),
                "sentence": row["SENTENCE"].strip(),
                "show": row["SHOW"].strip(),
                "speaker": row["SPEAKER"].strip(),
                "sarcasm": int(sarcasm),
            })
    return annotations


def build_dataset(
    video_dir: Path,
    audio_dir: Path,
    annotations: list[dict],
    skip_audio: bool = False,
) -> list[dict]:
    """Match videos to annotations, extract audio, build dataset rows."""
    rows = []
    missing = 0
    failed_audio = 0

    for i, ann in enumerate(annotations):
        uid = ann["id"]
        print(f"  [{i+1}/{len(annotations)}] {uid}: {ann['sentence'][:50]}...", end="")

        video_path = find_video_for_utterance(video_dir, ann["show"], uid)
        if video_path is None:
            print(" NO VIDEO")
            missing += 1
            continue

        audio_path = audio_dir / f"{uid}.wav"
        if not skip_audio:
            if not extract_audio(video_path, audio_path):
                print(" AUDIO FAIL")
                failed_audio += 1
                continue

        try:
            audio_data, sr = sf.read(str(audio_path))
        except Exception as e:
            print(f" READ FAIL: {e}")
            failed_audio += 1
            continue

        rows.append({
            "id": uid,
            "audio": {
                "array": audio_data.astype(np.float32),
                "sampling_rate": sr,
            },
            "sarcasm": ann["sarcasm"],
        })
        print(" OK")

    if missing:
        print(f"\n  WARNING: {missing} annotations had no matching video")
    if failed_audio:
        print(f"\n  WARNING: {failed_audio} audio extractions failed")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Prepare MUStARD dataset")
    parser.add_argument("--skip-download", action="store_true", help="Skip video download")
    parser.add_argument("--skip-audio", action="store_true", help="Skip audio extraction (use existing)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "mustard_raw"
    audio_dir = project_root / "data" / "mustard_audio"
    csv_path = project_root / "data" / "mustard++_text.csv"
    output_dir = project_root / "data" / "mustard_dataset"

    raw_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_download:
        print("=== Step 1: Download videos ===")
        video_dir = download_videos(raw_dir)
    else:
        video_dir = raw_dir / "context_final"

    if not csv_path.exists():
        print("=== Step 2: Download annotations ===")
        subprocess.run(
            [
                "wget",
                "-O", str(csv_path),
                "https://raw.githubusercontent.com/cfiltnlp/MUStARD_Plus_Plus/main/mustard++_text.csv",
            ],
            check=True,
        )
    else:
        print("=== Step 2: Annotations CSV already present ===")

    print("=== Step 3: Load annotations ===")
    annotations = load_annotations(str(csv_path))
    print(f"  Found {len(annotations)} target utterances with sarcasm labels")

    print("=== Step 4: Match videos and extract audio ===")
    rows = build_dataset(video_dir, audio_dir, annotations, skip_audio=args.skip_audio)
    print(f"\n  Total dataset rows: {len(rows)}")

    print("\n=== Step 5: Save HF dataset ===")
    from datasets import Dataset, Audio

    ds = Dataset.from_list(rows)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    ds.save_to_disk(str(output_dir))
    print(f"  Saved to {output_dir}")

    print(f"\n  Samples: {len(ds)}")
    print(f"  Columns: {ds.column_names}")
    print(f"  Sarcasm breakdown: {sum(r['sarcasm'] for r in rows)} positive / {sum(1 - r['sarcasm'] for r in rows)} negative")


if __name__ == "__main__":
    main()