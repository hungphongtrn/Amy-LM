"""DPO data collator for Amy LM - builds concatenated batches from NVTTS-FACodec rows."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch

_VENDOR_MOSS_AUDIO_SRC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src")
)
if _VENDOR_MOSS_AUDIO_SRC_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_MOSS_AUDIO_SRC_PATH)
from processing_moss_audio import MossAudioProcessor


class DPOCollator:
    """Collates NVTTS-FACodec rows into DPO batches for AmyLM."""

    AUDIO_TOKEN_ID: int = 151654
    AUDIO_BOS_ID: int = 151669
    AUDIO_EOS_ID: int = 151670
    SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "Listen carefully to the speaker's tone and respond appropriately "
        "to the following speech: <|audio_bos|><|AUDIO|><|audio_eos|>"
    )

    def __init__(
        self,
        processor: MossAudioProcessor,
        pad_token_id: int,
        max_length: int = 1024,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor._base_tokenizer
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token_id = pad_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def _extract_audio(self, audio_input: Any) -> tuple[torch.Tensor, int]:
        if isinstance(audio_input, dict):
            waveform = audio_input["array"]
            sample_rate = int(audio_input.get("sampling_rate", 16000))
        elif isinstance(audio_input, np.ndarray):
            waveform = audio_input
            sample_rate = 16000
        elif torch.is_tensor(audio_input):
            waveform = audio_input
            sample_rate = 16000
        else:
            raise TypeError(f"Unsupported audio format: {type(audio_input)!r}")

        waveform_tensor = torch.as_tensor(waveform, dtype=torch.float32).flatten()
        return waveform_tensor, sample_rate

    def _extract_mel_batch(self, waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        mels = [self.processor._extract_mel(waveform) for waveform in waveforms]
        seqlens = torch.tensor([mel.shape[-1] for mel in mels], dtype=torch.long)
        max_len = int(seqlens.max().item()) if len(mels) > 0 else 0
        audio_data = torch.zeros((len(mels), 128, max_len), dtype=torch.float32)
        for i, mel in enumerate(mels):
            audio_data[i, :, : mel.shape[-1]] = mel.to(torch.float32)
        return audio_data, seqlens

    def _tokenize_sample(self, audio: torch.Tensor, response_text: str) -> dict[str, Any]:
        mel = self.processor._extract_mel(audio)
        num_audio_frames = self.processor._conv3_downsample_len(mel.shape[-1])
        audio_placeholder_ids = self.processor._build_audio_placeholder_ids(num_audio_frames)

        span = self.processor._AUDIO_SPAN_RE.search(self.SYSTEM_PROMPT)
        if span is None:
            raise ValueError("SYSTEM_PROMPT does not contain a valid audio span")
        prefix = self.SYSTEM_PROMPT[: span.start()]
        suffix = self.SYSTEM_PROMPT[span.end() :]

        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        prompt_ids = prefix_ids + [self.AUDIO_BOS_ID] + audio_placeholder_ids + [self.AUDIO_EOS_ID] + suffix_ids

        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        response_ids = response_ids + [self.tokenizer.eos_token_id]
        full_ids = prompt_ids + response_ids
        audio_input_mask_positions = [token_id == self.AUDIO_TOKEN_ID for token_id in full_ids]
        return {
            "input_ids": full_ids,
            "audio_input_mask_positions": audio_input_mask_positions,
            "prompt_len": len(prompt_ids),
        }

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        waveforms = [self._extract_audio(example["audio"])[0] for example in examples]
        audio_data, audio_data_seqlens = self._extract_mel_batch(waveforms)

        prosody_sequences = [
            torch.as_tensor(example["prosody_codebooks_idx"], dtype=torch.long).flatten()
            for example in examples
        ]
        max_prosody_len = max(seq.shape[0] for seq in prosody_sequences)
        prosody_indices = torch.zeros((len(examples), 1, max_prosody_len), dtype=torch.long)
        for i, seq in enumerate(prosody_sequences):
            prosody_indices[i, 0, : seq.shape[0]] = seq

        timbre_vector = torch.stack(
            [torch.as_tensor(example["timbre_vector"], dtype=torch.float32) for example in examples],
            dim=0,
        )

        chosen_data = [self._tokenize_sample(waveforms[i], examples[i]["chosen"]) for i in range(len(examples))]
        rejected_data = [
            self._tokenize_sample(waveforms[i], examples[i]["rejected"]) for i in range(len(examples))
        ]

        chosen_lengths = [min(len(item["input_ids"]), self.max_length) for item in chosen_data]
        rejected_lengths = [min(len(item["input_ids"]), self.max_length) for item in rejected_data]
        max_len = max(chosen_lengths + rejected_lengths)
        if self.pad_to_multiple_of is not None and max_len % self.pad_to_multiple_of != 0:
            max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

        def _pad_side(batch_items: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

            input_ids = torch.full((len(batch_items), max_len), self.pad_token_id, dtype=torch.long)
            attention_mask = torch.zeros((len(batch_items), max_len), dtype=torch.long)
            completion_mask = torch.zeros((len(batch_items), max_len), dtype=torch.long)
            audio_input_mask = torch.zeros((len(batch_items), max_len), dtype=torch.bool)

            for i, item in enumerate(batch_items):
                ids = item["input_ids"][: self.max_length]
                audio_mask = item["audio_input_mask_positions"][: self.max_length]
                prompt_len = min(item["prompt_len"], len(ids))
                seq_len = len(ids)

                input_ids[i, :seq_len] = torch.tensor(ids, dtype=torch.long)
                attention_mask[i, :seq_len] = 1
                completion_mask[i, prompt_len:seq_len] = 1
                audio_input_mask[i, :seq_len] = torch.tensor(audio_mask, dtype=torch.bool)

            return input_ids, attention_mask, completion_mask, audio_input_mask

        chosen_ids, chosen_attn, chosen_completion, chosen_audio_mask = _pad_side(chosen_data)
        rejected_ids, rejected_attn, rejected_completion, rejected_audio_mask = _pad_side(rejected_data)

        input_ids = torch.cat([chosen_ids, rejected_ids], dim=0)
        attention_mask = torch.cat([chosen_attn, rejected_attn], dim=0)
        completion_mask = torch.cat([chosen_completion, rejected_completion], dim=0)
        audio_input_mask = torch.cat([chosen_audio_mask, rejected_audio_mask], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "audio_data": torch.cat([audio_data, audio_data], dim=0),
            "audio_data_seqlens": torch.cat([audio_data_seqlens, audio_data_seqlens], dim=0),
            "audio_input_mask": audio_input_mask,
            "prosody_indices": torch.cat([prosody_indices, prosody_indices], dim=0),
            "timbre_vector": torch.cat([timbre_vector, timbre_vector], dim=0),
        }
