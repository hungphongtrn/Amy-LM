"""Preference dataset processor for FACodec preference-pair preprocessing.

Encodes only prosody and timbre streams; content + acoustic are skipped.
FACodecEncoder.encode_batch() returns List[FACodecStreams] — one per sample.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import Audio as HFAudio
from datasets import Dataset, Features, Sequence, Value

from src.preprocessing.facodec_encoder import FACodecEncoder

DEFAULT_DATASET_TAG = "nvtts-preference"
DEFAULT_ADVERSARIAL_TAG = "nvtts-adversarial"
TIMBRE_DIM = 256

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

ADVERSARIAL_FEATURES = Features(
    {
        **PREFERENCE_FEATURES,
        "inverse_emotion": Value("string"),
        "strategy": Value("string"),
        "judge_fidelity_chosen": Value("int32"),
        "judge_fidelity_rejected": Value("int32"),
        "judge_ambiguity_chosen": Value("int32"),
        "judge_ambiguity_rejected": Value("int32"),
        "generation_attempts": Value("int32"),
    }
)


class PreferenceDatasetProcessor:
    """Process preference pair data through FACodec (prosody+timbre only)."""

    def __init__(self, encoder: FACodecEncoder, batch_size: int = 8) -> None:
        self.encoder = encoder
        self.batch_size = batch_size

    def process_dataset(self, dataset: Dataset, dataset_tag: str = DEFAULT_DATASET_TAG) -> Dataset:
        processed_rows: List[Dict[str, Any]] = []

        for start in range(0, len(dataset), self.batch_size):
            batch_rows = [dataset[i] for i in range(start, min(start + self.batch_size, len(dataset)))]
            audio_batch = [self._extract_audio_tensor(row) for row in batch_rows]
            streams_batch = self.encoder.encode_batch(audio_batch)

            if len(streams_batch) != len(batch_rows):
                raise ValueError(
                    f"encode_batch returned {len(streams_batch)} streams "
                    f"for {len(batch_rows)} inputs"
                )

            for row, streams in zip(batch_rows, streams_batch):
                processed_rows.append(self._build_processed_entry(row, streams, dataset_tag))

        return Dataset.from_list(processed_rows, features=PREFERENCE_FEATURES)

    def process_adversarial_dataset(
        self,
        jsonl_path: str,
        nvtts_dataset: Dataset,
        dataset_tag: str = DEFAULT_ADVERSARIAL_TAG,
    ) -> Dataset:
        pairs_by_id: Dict[str, Dict[str, Any]] = {}
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                p_id = p.get("id")
                if p_id:
                    pairs_by_id[p_id] = p

        matched: List[tuple[Dict[str, Any], Dict[str, Any]]] = []
        for i in range(len(nvtts_dataset)):
            row = nvtts_dataset[i]
            sample_id = row.get("id") or row.get("index", "")
            if sample_id in pairs_by_id:
                matched.append((row, pairs_by_id[sample_id]))

        if not matched:
            raise ValueError(
                f"No matching adversarial pairs found for any NVTTS sample. "
                f"Check that the JSONL ids match the NVTTS dataset ids."
            )

        processed_rows: List[Dict[str, Any]] = []
        for start in range(0, len(matched), self.batch_size):
            batch = matched[start : start + self.batch_size]
            audio_batch = [self._extract_audio_tensor(row) for row, _ in batch]
            streams_batch = self.encoder.encode_batch(audio_batch)

            if len(streams_batch) != len(batch):
                raise ValueError(
                    f"encode_batch returned {len(streams_batch)} streams "
                    f"for {len(batch)} inputs"
                )

            for (row, pair), streams in zip(batch, streams_batch):
                processed_rows.append(
                    self._build_adversarial_entry(row, pair, streams, dataset_tag)
                )

        return Dataset.from_list(processed_rows, features=ADVERSARIAL_FEATURES)

    def _extract_audio_tensor(self, row: Dict[str, Any]) -> torch.Tensor:
        audio = row["audio"]

        if isinstance(audio, dict):
            array = audio["array"]
        elif hasattr(audio, "get_all_samples"):
            tensor = audio.get_all_samples().data
            return tensor.float().squeeze(0)
        else:
            array = audio

        if isinstance(array, torch.Tensor):
            return array.float().squeeze(0)
        if isinstance(array, (bytes, bytearray)):
            array = np.frombuffer(array, dtype=np.float32)
        elif isinstance(array, list):
            array = np.array(array, dtype=np.float32)
        elif isinstance(array, np.ndarray):
            array = array.astype(np.float32)

        return torch.from_numpy(array).float()

    def _build_processed_entry(self, row: Dict[str, Any], streams: Any, dataset_tag: str) -> Dict[str, Any]:
        sample_id = row.get("id") or row.get("index", "?")
        prosody_tensor = streams.prosody_codebooks_idx
        if prosody_tensor is None or prosody_tensor.numel() == 0:
            raise ValueError(f"Empty prosody stream for sample {sample_id}")
        prosody = prosody_tensor.squeeze(0).tolist()

        timbre_tensor = streams.timbre_vector
        if timbre_tensor is None or timbre_tensor.shape[-1] != TIMBRE_DIM:
            raise ValueError(
                f"Timbre vector has {timbre_tensor.shape[-1] if timbre_tensor is not None else 0} dims "
                f"(expected {TIMBRE_DIM}) for sample {sample_id}"
            )
        timbre = timbre_tensor.tolist()

        audio = row["audio"]
        if isinstance(audio, dict):
            audio_array = audio.get("array")
            audio_sr = audio.get("sampling_rate", 16000)
            audio_path = audio.get("path")
        elif hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            audio_array = samples.data.squeeze(0).numpy().astype(np.float32)
            audio_sr = audio.metadata.sample_rate
            audio_path = None
        else:
            audio_array = audio
            audio_sr = 16000
            audio_path = None

        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.numpy().astype(np.float32)
        elif isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)
        elif isinstance(audio_array, (bytes, bytearray)):
            audio_array = np.frombuffer(audio_array, dtype=np.float32)

        return {
            "dataset": row.get("dataset", dataset_tag),
            "id": row.get("id") or row.get("index", ""),
            "audio": {
                "path": audio_path,
                "array": audio_array,
                "sampling_rate": audio_sr,
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

    def _build_adversarial_entry(
        self, row: Dict[str, Any], pair: Dict[str, Any], streams: Any, dataset_tag: str
    ) -> Dict[str, Any]:
        sample_id = row.get("id") or row.get("index", "?")
        prosody_tensor = streams.prosody_codebooks_idx
        if prosody_tensor is None or prosody_tensor.numel() == 0:
            raise ValueError(f"Empty prosody stream for sample {sample_id}")
        prosody = prosody_tensor.squeeze(0).tolist()

        timbre_tensor = streams.timbre_vector
        if timbre_tensor is None or timbre_tensor.shape[-1] != TIMBRE_DIM:
            raise ValueError(
                f"Timbre vector has {timbre_tensor.shape[-1] if timbre_tensor is not None else 0} dims "
                f"(expected {TIMBRE_DIM}) for sample {sample_id}"
            )
        timbre = timbre_tensor.tolist()

        audio = row["audio"]
        if isinstance(audio, dict):
            audio_array = audio.get("array")
            audio_sr = audio.get("sampling_rate", 16000)
            audio_path = audio.get("path")
        elif hasattr(audio, "get_all_samples"):
            samples = audio.get_all_samples()
            audio_array = samples.data.squeeze(0).numpy().astype(np.float32)
            audio_sr = audio.metadata.sample_rate
            audio_path = None
        else:
            audio_array = audio
            audio_sr = 16000
            audio_path = None

        if isinstance(audio_array, torch.Tensor):
            audio_array = audio_array.numpy().astype(np.float32)
        elif isinstance(audio_array, list):
            audio_array = np.array(audio_array, dtype=np.float32)
        elif isinstance(audio_array, (bytes, bytearray)):
            audio_array = np.frombuffer(audio_array, dtype=np.float32)

        return {
            "dataset": row.get("dataset", dataset_tag),
            "id": row.get("id") or row.get("index", ""),
            "audio": {
                "path": audio_path,
                "array": audio_array,
                "sampling_rate": audio_sr,
            },
            "prosody_codebooks_idx": prosody,
            "timbre_vector": timbre,
            "chosen": pair.get("chosen", ""),
            "rejected": pair.get("rejected", ""),
            "rationale_chosen": "",
            "rationale_rejected": "",
            "emotion_label": row.get("emotion_label", pair.get("emotion_label", "")),
            "speaker_name": row.get("speaker_name", ""),
            "speaker_gender": row.get("speaker_gender", ""),
            "speaker_age_context": row.get("speaker_age_context", ""),
            "speaker_nationality": row.get("speaker_nationality", ""),
            "transcript_with_tags": row.get("transcript_with_tags", ""),
            "bare_transcript": row.get("bare_transcript", ""),
            "cosine_similarity": 0.0,
            "label": -1,
            "inverse_emotion": pair.get("inverse_emotion", ""),
            "strategy": pair.get("strategy", ""),
            "judge_fidelity_chosen": int(pair.get("judge_fidelity_chosen", 0)),
            "judge_fidelity_rejected": int(pair.get("judge_fidelity_rejected", 0)),
            "judge_ambiguity_chosen": int(pair.get("judge_ambiguity_chosen", 0)),
            "judge_ambiguity_rejected": int(pair.get("judge_ambiguity_rejected", 0)),
            "generation_attempts": int(pair.get("generation_attempts", 0)),
        }
