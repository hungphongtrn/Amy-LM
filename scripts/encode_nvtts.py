from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torchaudio
from datasets import Audio, Dataset, Features, Sequence, Value, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.facodec_encoder import FACodecEncoder


SPLITS = ("train", "dev", "test")


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encode NVTTS with FACodec streams")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output-repo", default="hungphongtrn/nvtts_facodec")
    parser.add_argument("--output-dir", default="data/nvtts_facodec")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument(
        "--pairs-path",
        default="data/nvtts_preference_pairs/pairs_scored.parquet",
    )
    return parser


def _decode_and_resample(audio_obj: Any, target_sr: int = 16000) -> np.ndarray:
    if isinstance(audio_obj, dict):
        waveform = torch.tensor(audio_obj["array"], dtype=torch.float32).unsqueeze(0)
        source_sr = int(audio_obj["sampling_rate"])
    else:
        samples = audio_obj.get_all_samples()
        waveform = samples.data.float()
        source_sr = int(samples.sample_rate)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
    if source_sr != target_sr:
        waveform = torchaudio.transforms.Resample(orig_freq=source_sr, new_freq=target_sr)(waveform)
    return waveform.squeeze(0).cpu().numpy().astype(np.float32)


def _dataset_features() -> Features:
    return Features(
        {
            "index": Value("string"),
            "audio": Audio(sampling_rate=16000),
            "prosody_codebooks_idx": Sequence(Value("int64")),
            "content_codebooks_idx": Sequence(Sequence(Value("int64"))),
            "acoustic_codebooks_idx": Sequence(Sequence(Value("int64"))),
            "timbre_vector": Sequence(Value("float32")),
            "chosen": Value("string"),
            "rejected": Value("string"),
            "cosine_similarity": Value("float32"),
        }
    )


def encode_split(
    split_ds: Dataset,
    pairs_df: pd.DataFrame,
    encoder: FACodecEncoder,
    batch_size: int,
) -> list[dict[str, Any]]:
    pairs_by_id = pairs_df.set_index("id")
    rows: list[dict[str, Any]] = []
    batch_audio: list[torch.Tensor] = []
    pending: list[tuple[str, np.ndarray]] = []

    for sample in split_ds:
        idx = str(sample["index"])
        if idx not in pairs_by_id.index:
            raise KeyError(f"Missing preference pair for id={idx}")
        audio_np = _decode_and_resample(sample["audio"], target_sr=16000)
        batch_audio.append(torch.from_numpy(audio_np).float())
        pending.append((idx, audio_np))

        if len(batch_audio) >= batch_size:
            streams = encoder.encode_batch(batch_audio)
            rows.extend(_build_rows(streams, pending, pairs_by_id))
            batch_audio = []
            pending = []

    if batch_audio:
        streams = encoder.encode_batch(batch_audio)
        rows.extend(_build_rows(streams, pending, pairs_by_id))

    return rows


def _build_rows(streams_batch: list[Any], pending: list[tuple[str, np.ndarray]], pairs_by_id: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for streams, (idx, audio_np) in zip(streams_batch, pending):
        pair = pairs_by_id.loc[idx]
        out.append(
            {
                "index": idx,
                "audio": {"array": audio_np, "sampling_rate": 16000},
                "prosody_codebooks_idx": streams.prosody_codebooks_idx.squeeze(0).cpu().numpy().astype(np.int64).tolist(),
                "content_codebooks_idx": streams.content_codebooks_idx.cpu().numpy().astype(np.int64).tolist(),
                "acoustic_codebooks_idx": streams.acoustic_codebooks_idx.cpu().numpy().astype(np.int64).tolist(),
                "timbre_vector": streams.timbre_vector.cpu().numpy().astype(np.float32).tolist(),
                "chosen": str(pair["chosen"]),
                "rejected": str(pair["rejected"]),
                "cosine_similarity": np.float32(pair["cosine_similarity"]).item(),
            }
        )
    return out


def main() -> int:
    args = create_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs_df = pd.read_parquet(args.pairs_path)
    encoder = FACodecEncoder(device=args.device, force_mock=args.mock)

    for split in SPLITS:
        split_ds = load_dataset("deepvk/NonverbalTTS", split=split)
        rows = encode_split(split_ds=split_ds, pairs_df=pairs_df, encoder=encoder, batch_size=args.batch_size)
        dataset = Dataset.from_list(rows, features=_dataset_features())
        split_path = output_dir / f"{split}.parquet"
        dataset.to_parquet(str(split_path))
        if not args.no_push:
            dataset.push_to_hub(args.output_repo, split=split)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
