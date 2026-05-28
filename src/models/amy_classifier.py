"""AmyForProsodyClassification — end-to-end model for binary sarcasm classification."""

from __future__ import annotations

import torch
import torch.nn as nn

from .amy_lm import AmyMossLM, AmyMossLMConfig
from .moss_audio_model import MossAudioModel


class AmyForProsodyClassification(nn.Module):
    """Amy LM for binary sarcasm classification with Prosody + Timbre residual fusion.

    Composes (HAS-A) an AmyMossLM for audio encoding and FACodec enrichment.
    Freezes MOSS-Audio backbone (audio_encoder, audio_adapter, language_model).
    Trainable: FACodec modules (on self.amy_moss) and a 2-class classifier head.

    Args:
        moss_model_id: HuggingFace model ID for MOSS-Audio backbone.
        device: Device for model (default: "cuda" if available).
        torch_dtype: Precision for MOSS-Audio backbone (default: bfloat16).
        load_in_4bit: Whether to load MOSS-Audio in 4-bit quantization.
        prosody_warm_start_vectors_path: Path to FACodec decoder checkpoint
            for warm-starting ProsodyEmbedding.
        num_classes: Number of output classes (default: 2 for binary sarcasm).
        gradient_checkpointing: Enable gradient checkpointing on Qwen3 LM.
    """

    def __init__(
        self,
        moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        prosody_warm_start_vectors_path: str | None = None,
        warm_start_vectors: torch.Tensor | None = None,
        num_classes: int = 2,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._device = torch.device(device)

        from_pretrained_kwargs: dict = {
            "trust_remote_code": True,
        }
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            from_pretrained_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            from_pretrained_kwargs["torch_dtype"] = torch_dtype
        if self._device.type == "cuda":
            from_pretrained_kwargs["device_map"] = str(self._device)

        moss = MossAudioModel.from_pretrained(moss_model_id, **from_pretrained_kwargs)
        moss.eval()

        amy_config = AmyMossLMConfig(
            moss_config=moss.config,
            prosody_warm_start_vectors_path=prosody_warm_start_vectors_path,
        )
        self.amy_moss = AmyMossLM(amy_config, moss=moss)

        if warm_start_vectors is not None and prosody_warm_start_vectors_path is None:
            self.amy_moss._warm_start_prosody(warm_start_vectors)

        backbone_device = next(self.amy_moss.parameters()).device
        self.classifier = nn.Linear(amy_config.hidden_dim, num_classes, device=backbone_device)

        if gradient_checkpointing:
            self.amy_moss.moss.language_model.gradient_checkpointing_enable()

    def get_language_model(self) -> nn.Module:
        return self.amy_moss.moss.language_model

    def forward(
        self,
        audio: torch.Tensor,
        prosody_indices: torch.Tensor,
        timbre_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass: audio + FACodec features → 2-class logits.

        Args:
            audio: Raw waveform [B, T_audio] at 16kHz.
            prosody_indices: FACodec prosody VQ IDs [B, 1, T80].
            timbre_vector: FACodec speaker embedding [B, 256].

        Returns:
            Logits [B, 2] for binary sarcasm classification.
        """
        H = self.amy_moss.encode_enriched_audio_embeds(
            audio, prosody_indices=prosody_indices, timbre_vector=timbre_vector
        )

        language_model = self.get_language_model()
        lm_dtype = next(language_model.parameters()).dtype
        lm_out = language_model(inputs_embeds=H.to(dtype=lm_dtype)).last_hidden_state

        pooled = lm_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
