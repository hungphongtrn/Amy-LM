"""AmyForProsodyClassification — end-to-end model for binary sarcasm classification."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embedding import ProsodyEmbedding, TimbreProjection
from .fusion import ResidualFusion
from .moss_audio import MossAudioWrapper
from .pooling import TemporalPool


class AmyForProsodyClassification(nn.Module):
    """Amy LM for binary sarcasm classification with Prosody + Timbre residual fusion.

    Freezes MOSS-Audio backbone (audio_encoder, audio_adapter, language_model).
    Trainable modules: ProsodyEmbedding (warm-started), TimbreProjection,
    ResidualFusion lambdas, and a 2-class classifier head.

    Args:
        warm_start_vectors: FACodec prosody codebook vectors [1024, 8] for
            warm-starting ProsodyEmbedding.
        moss_model_id: HuggingFace model ID for MOSS-Audio backbone.
        device: Device for model (default: "cpu").
        stream_config: Dict controlling which FACodec streams are active.
            Default: prosody=True, timbre=True, content=False, acoustic=False.
        hidden_dim: Embedding dimension (must match MOSS-Audio hidden_size=2560).
        num_classes: Number of output classes (default: 2 for binary sarcasm).
        input_rate: FACodec frame rate in Hz (default: 80.0).
        output_rate: MOSS-Audio semantic frame rate in Hz (default: 12.5).
        timbre_dim: Dimensionality of input timbre vector (default: 256).
        vocab_size: FACodec codebook vocabulary size (default: 1024).
    """

    def __init__(
        self,
        warm_start_vectors: torch.Tensor,
        moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str = "cpu",
        stream_config: dict | None = None,
        hidden_dim: int = 2560,
        num_classes: int = 2,
        input_rate: float = 80.0,
        output_rate: float = 12.5,
        timbre_dim: int = 256,
        vocab_size: int = 1024,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.output_rate = output_rate

        if stream_config is None:
            stream_config = {
                "prosody": True,
                "content": False,
                "acoustic": False,
                "timbre": True,
            }
        self.stream_config = stream_config

        self.wrapper = MossAudioWrapper(model_id=moss_model_id, device=self.device)

        if stream_config.get("prosody", False):
            self.prosody_embedding = ProsodyEmbedding(
                vocab_size=vocab_size,
                embed_dim=hidden_dim,
                init_strategy="warm_start",
                warm_start_vectors=warm_start_vectors,
            )
        if stream_config.get("timbre", False):
            self.timbre_projection = TimbreProjection(
                timbre_dim=timbre_dim,
                output_dim=hidden_dim,
            )

        self.temporal_pool = TemporalPool(
            input_rate=input_rate,
            output_rate=output_rate,
        )

        self.fusion = ResidualFusion(hidden_dim=hidden_dim)

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._freeze_backbone()
        self._ensure_facodec_trainable()

    def _freeze_backbone(self) -> None:
        for param in self.wrapper.parameters():
            param.requires_grad = False

    def _ensure_facodec_trainable(self) -> None:
        for name, module in self.named_children():
            if name in ("wrapper",):
                continue
            for param in module.parameters():
                param.requires_grad = True

    def get_language_model(self) -> nn.Module:
        return self.wrapper.language_model

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
        with torch.no_grad():
            semantic = self.wrapper.encode_semantic(audio)
        semantic = semantic.float()
        T_moss = semantic.shape[1]

        p_emb = self.prosody_embedding(prosody_indices)
        P = self.temporal_pool(p_emb)
        if P.shape[1] != T_moss:
            P = P.transpose(1, 2)
            P = F.adaptive_avg_pool1d(P, T_moss)
            P = P.transpose(1, 2)

        t_proj = self.timbre_projection(timbre_vector)
        T = t_proj.unsqueeze(1).expand(-1, T_moss, -1)

        H = self.fusion(semantic, prosody=P, timbre=T, content=None, acoustic=None)

        language_model = self.get_language_model()
        lm_dtype = next(language_model.parameters()).dtype
        H_lm = H.to(dtype=lm_dtype)
        lm_out = language_model(inputs_embeds=H_lm).last_hidden_state
        lm_out = lm_out.float()

        pooled = lm_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
