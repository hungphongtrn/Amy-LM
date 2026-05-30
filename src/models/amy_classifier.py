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
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass: audio + FACodec features → 2-class logits.

        Args:
            audio: Raw waveform [B, T_audio] at 16kHz.
            prosody_indices: FACodec prosody VQ IDs [B, 1, T80].
            timbre_vector: FACodec speaker embedding [B, 256].
            **kwargs: Accepts HF-style PeftModel passthrough (input_ids, etc.) — ignored.

        Returns:
            Logits [B, 2] for binary sarcasm classification.
        """
        language_model = self.get_language_model()
        lm_dtype = next(language_model.parameters()).dtype

        H, audio_out_lens = self.amy_moss.encode_enriched_audio_embeds(
            audio,
            prosody_indices=prosody_indices,
            timbre_vector=timbre_vector,
        )
        B, T_max, D = H.shape

        pos = torch.arange(T_max, device=H.device).unsqueeze(0).expand(B, -1)
        audio_mask = pos < audio_out_lens.unsqueeze(-1)
        attention_mask = audio_mask.to(lm_dtype)

        lm_out = language_model(
            inputs_embeds=H.to(dtype=lm_dtype),
            attention_mask=attention_mask,
        ).last_hidden_state

        mask_float = audio_mask.to(dtype=lm_out.dtype)
        pooled = (lm_out * mask_float.unsqueeze(-1)).sum(dim=1) / mask_float.sum(dim=1, keepdim=True).clamp(min=1)
        return self.classifier(pooled)


def wrap_classifier_with_lora(
    model: AmyForProsodyClassification,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """Wrap AmyForProsodyClassification with LoRA for H2 gradient-flow test.

    LoRA targets (via regex):
      - moss.audio_adapter.* (gate_proj, up_proj, down_proj)  [GatedMLP]
      - moss.language_model.* (q_proj, k_proj, v_proj, o_proj,   [Qwen3]
                               up_proj, down_proj, gate_proj)

    modules_to_save (fully trainable, fp32):
      - amy_moss.prosody_embedding
      - amy_moss.timbre_projection
      - amy_moss.temporal_pool
      - amy_moss.residual_fusion
      - classifier

    Audio encoder stays frozen (no LoRA, no modules_to_save).

    Args:
        model: AmyForProsodyClassification instance (frozen backbone, trainable FACodec+head).
        r: LoRA rank (default: 8, matches DPO).
        lora_alpha: LoRA alpha (default: 16, matches DPO).
        lora_dropout: LoRA dropout (default: 0.05, matches DPO).

    Returns:
        PeftModel wrapping the classifier.
    """
    from peft import LoraConfig, TaskType, get_peft_model

    for module in (
        model.amy_moss.prosody_embedding,
        model.amy_moss.timbre_projection,
        model.amy_moss.temporal_pool,
        model.amy_moss.residual_fusion,
    ):
        module.to(dtype=torch.float32)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Pre-LoRA trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=(
            r"^(?:amy_moss\.moss\.audio_adapter|amy_moss\.moss\.language_model)"
            r".*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$"
        ),
        modules_to_save=[
            "amy_moss.prosody_embedding",
            "amy_moss.timbre_projection",
            "amy_moss.temporal_pool",
            "amy_moss.residual_fusion",
            "classifier",
        ],
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model
