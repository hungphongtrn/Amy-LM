"""Baseline classifier: MOSS-Audio frozen backbone + Linear(2560->2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .amy_lm import AmyMossLM, AmyMossLMConfig
from .moss_audio_model import MossAudioModel


class BaselineClassifier(nn.Module):
    """MOSS-Audio semantic encoder + Qwen3 language model + mean-pool + Linear classifier.

    Composes (HAS-A) an AmyMossLM (with no FACodec enrichment).
    Frozen MOSS-Audio backbone. Trainable: LayerNorm + Linear(2560->2) classifier head.
    Used as the baseline for measuring prosody/timbre contribution.
    """

    def __init__(
        self,
        moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str | None = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        load_in_4bit: bool = False,
        num_classes: int = 2,
        hidden_dim: int = 2560,
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

        amy_config = AmyMossLMConfig(moss_config=moss.config)
        self.amy_moss = AmyMossLM(amy_config, moss=moss)

        backbone_device = next(self.amy_moss.parameters()).device
        self.norm = nn.LayerNorm(hidden_dim, device=backbone_device)
        self.classifier = nn.Linear(hidden_dim, num_classes, device=backbone_device)

        if gradient_checkpointing:
            self.amy_moss.moss.language_model.gradient_checkpointing_enable()

    def get_language_model(self) -> nn.Module:
        return self.amy_moss.moss.language_model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        language_model = self.get_language_model()
        lm_dtype = next(language_model.parameters()).dtype

        logits_list = []
        for i in range(audio.shape[0]):
            H = self.amy_moss.encode_enriched_audio_embeds(audio[i : i + 1])
            H = self.norm(H)
            lm_out = language_model(inputs_embeds=H.to(dtype=lm_dtype)).last_hidden_state
            pooled = lm_out.mean(dim=1)
            logits_list.append(self.classifier(pooled))
        return torch.cat(logits_list, dim=0)
