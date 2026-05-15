"""MOSS-Audio backbone wrapper for Amy LM."""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn


_VENDOR_MOSS_AUDIO_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio")
)
_VENDOR_MOSS_AUDIO_SRC_PATH = os.path.join(_VENDOR_MOSS_AUDIO_PATH, "src")
if os.path.isdir(_VENDOR_MOSS_AUDIO_SRC_PATH) and _VENDOR_MOSS_AUDIO_SRC_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_MOSS_AUDIO_SRC_PATH)

from modeling_moss_audio import MossAudioModel
from processing_moss_audio import MossAudioProcessor


class MossAudioWrapper(nn.Module):
    """Frozen MOSS-Audio backbone exposing sub-modules for Amy LM.

    Loads MOSS-Audio-4B-Thinking via transformers, then extracts:
      - audio_encoder: Whisper-style audio encoder
      - audio_adapter: GatedMLP adapter (audio features -> LLM embedding space)
      - language_model: Qwen3 language model

    All sub-modules are frozen (requires_grad=False). The wrapper provides
    encode_semantic() which runs audio through encoder + adapter to produce
    the Semantic Stream S_t [B, T_frames, 2560].
    """

    def __init__(
        self,
        model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model_id = model_id
        self.device = torch.device(device)

        self.model = MossAudioModel.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype="auto",
            device_map=str(self.device),
        )
        self.model.eval()
        self.processor = MossAudioProcessor.from_pretrained(
            model_id,
            trust_remote_code=True,
            enable_time_marker=True,
        )

        self.audio_encoder = self.model.audio_encoder
        self.audio_adapter = self.model.audio_adapter
        self.language_model = self.model.language_model

        for param in self.parameters():
            param.requires_grad = False

        self.to(self.device)
        self.eval()

    def encode_semantic(self, audio: torch.Tensor) -> torch.Tensor:
        """Produce Semantic Stream S_t from raw audio waveform.

        Args:
            audio: Raw audio waveform [B, T_audio] at 16 kHz.

        Returns:
            Semantic Stream [B, T_frames, 2560] at ~12.5 Hz frame rate.
        """
        audio = audio.to(self.device)
        with torch.no_grad():
            if audio.dim() != 2:
                raise ValueError(f"Expected audio shape [B, T_audio], got {tuple(audio.shape)}")

            mels = [self.processor._extract_mel(audio[i].detach().cpu()) for i in range(audio.shape[0])]
            seqlens = torch.tensor([mel.shape[-1] for mel in mels], dtype=torch.long)
            max_len = int(seqlens.max().item())
            audio_data = torch.zeros(
                (len(mels), mels[0].shape[0], max_len),
                dtype=self.model.dtype,
            )
            for idx, mel in enumerate(mels):
                audio_data[idx, :, : mel.shape[-1]] = mel.to(dtype=self.model.dtype)
            audio_data = audio_data.to(self.device)
            audio_data_seqlens = seqlens.to(self.device)

            semantic_list = []
            max_frames = 0
            for idx in range(audio_data.shape[0]):
                features = self.audio_encoder(
                    input_features=audio_data[idx],
                    feature_lens=audio_data_seqlens[idx : idx + 1],
                    output_deepstack_hidden_states=False,
                ).last_hidden_state
                semantic = self.audio_adapter(features)
                semantic = semantic.squeeze(0)
                semantic_list.append(semantic)
                max_frames = max(max_frames, semantic.shape[0])

            semantic_batch = torch.zeros(
                (len(semantic_list), max_frames, semantic_list[0].shape[-1]),
                device=self.device,
                dtype=semantic_list[0].dtype,
            )
            for idx, semantic in enumerate(semantic_list):
                semantic_batch[idx, : semantic.shape[0], :] = semantic
        return semantic_batch
