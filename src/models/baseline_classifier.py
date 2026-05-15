"""Baseline classifier: MOSS-Audio frozen backbone + Linear(2560->2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .moss_audio import MossAudioWrapper


class BaselineClassifier(nn.Module):
    """MOSS-Audio semantic encoder + Qwen3 language model + mean-pool + Linear classifier.

    Frozen MOSS-Audio backbone. Trainable: Linear(2560->2) classifier head.
    No FACodec streams. Used as the baseline for measuring prosody/timbre contribution.
    """

    def __init__(
        self,
        moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str = "cpu",
        num_classes: int = 2,
        hidden_dim: int = 2560,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.wrapper = MossAudioWrapper(model_id=moss_model_id, device=self.device)
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._freeze_backbone()
        self._ensure_head_trainable()

    def _freeze_backbone(self) -> None:
        for param in self.wrapper.parameters():
            param.requires_grad = False

    def _ensure_head_trainable(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = True

    def get_language_model(self) -> nn.Module:
        return self.wrapper.language_model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            semantic = self.wrapper.encode_semantic(audio)
        semantic = semantic.float()
        semantic = self.norm(semantic)

        language_model = self.get_language_model()
        lm_dtype = next(language_model.parameters()).dtype
        h_lm = semantic.to(dtype=lm_dtype)
        lm_out = language_model(inputs_embeds=h_lm).last_hidden_state
        lm_out = lm_out.float()

        pooled = lm_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
