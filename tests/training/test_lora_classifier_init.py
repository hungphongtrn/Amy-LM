"""Tests for wrap_classifier_with_lora() -- LoRA classifier wrapping."""

from __future__ import annotations

import pytest
import torch


def _tiny_moss_config():
    from src.models.moss_audio_model import MossAudioConfig

    return MossAudioConfig(
        language_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
        audio_config={
            "d_model": 32,
            "output_dim": 32,
            "num_mel_bins": 128,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 8,
            "deepstack_encoder_layer_indexes": [],
        },
        adapter_hidden_size=32,
    )


class TestLoraClassifierStaticInit:
    """Verify LoRA wrapper produces correct parameter partitioning (no forward)."""

    def test_lora_adapters_on_audio_adapter_and_language_model(self, device, monkeypatch):
        from src.models.moss_audio_model import MossAudioModel
        from src.models.amy_lm import AmyMossLMConfig
        from src.models.amy_classifier import (
            AmyForProsodyClassification,
            wrap_classifier_with_lora,
        )

        tiny = MossAudioModel(_tiny_moss_config()).to(device)
        hidden_dim = tiny.config.language_config.hidden_size

        monkeypatch.setattr(MossAudioModel, "from_pretrained", lambda *a, **kw: tiny)

        original_init = AmyMossLMConfig.__init__

        def patched_init(self, moss_config=None, hidden_dim=None, **kwargs):
            if moss_config is not None and hidden_dim is None:
                hidden_dim = moss_config.language_config.hidden_size
            return original_init(self, moss_config=moss_config, hidden_dim=hidden_dim, **kwargs)

        monkeypatch.setattr(AmyMossLMConfig, "__init__", patched_init)

        vectors = torch.randn(1024, hidden_dim, device=device)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
        peft_model = wrap_classifier_with_lora(model, r=2, lora_alpha=4, lora_dropout=0.0)

        trainable = [name for name, param in peft_model.named_parameters() if param.requires_grad]

        assert any("moss.audio_adapter" in name and "lora_" in name for name in trainable), (
            "Missing LoRA on audio_adapter"
        )
        assert any("moss.language_model" in name and "lora_" in name for name in trainable), (
            "Missing LoRA on language_model"
        )
        assert any("prosody_embedding" in name for name in trainable), "prosody_embedding not trainable"
        assert any("timbre_projection" in name for name in trainable), "timbre_projection not trainable"
        assert any("residual_fusion" in name for name in trainable), "residual_fusion not trainable"
        assert any("classifier" in name for name in trainable), "classifier not trainable"
        # temporal_pool has no parameters (pure functional layer) - not expected in trainable list

        assert not any("audio_encoder" in name and "lora_" in name for name in trainable), (
            "LoRA on audio_encoder (should be frozen)"
        )
        for name, param in peft_model.named_parameters():
            if "moss.language_model" in name and "lora_" not in name:
                assert not param.requires_grad, f"LM base param {name} should be frozen"
            if "moss.audio_encoder" in name:
                assert not param.requires_grad, f"Audio encoder param {name} should be frozen"


class TestLoraClassifierGradientFlow:
    """GPU tests: forward+backward produces non-zero LoRA/FACodec gradients."""

    def test_lora_and_facodec_gradients_nonzero(self, device, require_gpu):
        from src.models.amy_classifier import (
            AmyForProsodyClassification,
            wrap_classifier_with_lora,
        )
        import torch.nn as nn

        vectors = torch.randn(1024, 2560, device=device)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
        peft_model = wrap_classifier_with_lora(model, r=2, lora_alpha=4, lora_dropout=0.0)
        peft_model.train()

        B = 1
        audio = torch.randn(B, 24000, device=device)
        prosody = torch.randint(0, 1024, (B, 1, 120), device=device)
        timbre = torch.randn(B, 256, device=device)
        labels = torch.tensor([0], device=device)

        logits = peft_model(audio=audio, prosody_indices=prosody, timbre_vector=timbre)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()

        lora_params = [
            (n, p) for n, p in peft_model.named_parameters() if "lora_" in n and p.requires_grad
        ]
        assert len(lora_params) > 0, "No LoRA parameters found"
        for name, param in lora_params:
            assert param.grad is not None, f"LoRA {name} has None grad"
            assert param.grad.abs().sum().item() > 0, f"LoRA {name} has zero grad"

        save_params = [
            (n, p)
            for n, p in peft_model.named_parameters()
            if p.requires_grad
            and "lora_" not in n
            and "audio_encoder" not in n
        ]
        assert len(save_params) > 0, "No modules_to_save params found"
        for name, param in save_params:
            assert param.grad is not None, f"{name} has None grad"
            assert param.grad.abs().sum().item() > 0, f"{name} has zero grad"

        for name, param in peft_model.named_parameters():
            if "audio_encoder" in name:
                assert param.grad is None, f"Frozen encoder {name} has grad"
