"""Preference dataset processor for FACodec preference-pair preprocessing."""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Audio as HFAudio
from datasets import Dataset, Features, Sequence, Value

from src.preprocessing.facodec_encoder import FACodecEncoder


PREFERENCE_FEATURES = Features(
    {
        "dataset": Value("string"),
        "id": Value("string"),
        "audio": HFAudio(sampling_rate=16000),
        "prosody_codebooks_idx": Sequence(Value("int64")),
        "timbre_vector": Sequence(Value("float32")),
        "chosen": Value("string"),
        "rejected": Value("string"),
        "rationale_chosen": Value("string"),
        "rationale_rejected": Value("string"),
        "emotion_label": Value("string"),
        "speaker_name": Value("string"),
        "speaker_gender": Value("string"),
        "speaker_age_context": Value("string"),
        "speaker_nationality": Value("string"),
        "transcript_with_tags": Value("string"),
        "bare_transcript": Value("string"),
        "cosine_similarity": Value("float32"),
        "label": Value("int64"),
    }
)


class PreferenceDatasetProcessor:
    """Process preference pair data through FACodec (prosody+timbre only)."""

    def __init__(self, encoder: FACodecEncoder, batch_size: int = 8) -> None:
        self.encoder = encoder
        self.batch_size = batch_size

    def process_dataset(self, dataset: Dataset, dataset_tag: str = "nvtts-preference") -> Dataset:
        processed_rows: List[Dict[str, Any]] = []

        for start in range(0, len(dataset), self.batch_size):
            batch_rows = [dataset[i] for i in range(start, min(start + self.batch_size, len(dataset)))]
            audio_batch = [self._extract_audio_tensor(row) for row in batch_rows]
            streams_batch = self.encoder.encode_batch(audio_batch)

            for row, streams in zip(batch_rows, streams_batch):
                processed_rows.append(self._build_processed_entry(row, streams, dataset_tag))

        return Dataset.from_list(processed_rows, features=PREFERENCE_FEATURES)

    def _extract_audio_tensor(self, row: Dict[str, Any]) -> torch.Tensor:
        audio = row["audio"]
        array = audio["array"] if isinstance(audio, dict) else audio

        if isinstance(array, (bytes, bytearray)):
            array = np.frombuffer(array, dtype=np.float32)
        elif isinstance(array, list):
            array = np.array(array, dtype=np.float32)
        elif isinstance(array, np.ndarray):
            array = array.astype(np.float32)

        return torch.from_numpy(array).float()

    def _build_processed_entry(self, row: Dict[str, Any], streams: Any, dataset_tag: str) -> Dict[str, Any]:
        prosody = streams.prosody_codebooks_idx.squeeze(0).tolist()
        timbre = streams.timbre_vector.tolist()
        audio = row["audio"]
        audio_array = audio.get("array") if isinstance(audio, dict) else audio
        if isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)
        elif isinstance(audio_array, (bytes, bytearray)):
            audio_array = np.frombuffer(audio_array, dtype=np.float32)

        return {
            "dataset": row.get("dataset", dataset_tag),
            "id": row.get("id", ""),
            "audio": {
                "path": audio.get("path") if isinstance(audio, dict) else None,
                "array": audio_array,
                "sampling_rate": audio.get("sampling_rate", 16000) if isinstance(audio, dict) else 16000,
            },
            "prosody_codebooks_idx": prosody,
            "timbre_vector": timbre,
            "chosen": row.get("chosen", ""),
            "rejected": row.get("rejected", ""),
            "rationale_chosen": row.get("rationale_chosen", ""),
            "rationale_rejected": row.get("rationale_rejected", ""),
            "emotion_label": row.get("emotion_label", ""),
            "speaker_name": row.get("speaker_name", ""),
            "speaker_gender": row.get("speaker_gender", ""),
            "speaker_age_context": row.get("speaker_age_context", ""),
            "speaker_nationality": row.get("speaker_nationality", ""),
            "transcript_with_tags": row.get("transcript_with_tags", ""),
            "bare_transcript": row.get("bare_transcript", ""),
            "cosine_similarity": np.float32(row.get("cosine_similarity", 0.0)).item(),
            "label": -1,
        }
