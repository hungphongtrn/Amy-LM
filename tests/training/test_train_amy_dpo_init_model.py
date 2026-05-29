"""Regression tests for scripts/train_amy_dpo.py model initialization."""

from __future__ import annotations

import torch

from scripts import train_amy_dpo
from src.models.moss_audio_model import MossAudioConfig, MossAudioModel
from src.training.config import DPOTrainingConfig


def _tiny_moss() -> MossAudioModel:
    config = MossAudioConfig(
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
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "deepstack_encoder_layer_indexes": [],
        },
    )
    return MossAudioModel(config)


def _unwrap_amy(base) -> torch.nn.Module:
    """Navigate PEFT wrapper to find the underlying AmyMossLM."""
    module = base
    while hasattr(module, "base_model"):
        module = module.base_model
    return module


def test_dpo_init_model_keeps_facodec_trainable_and_scopes_lora_to_qwen(monkeypatch):
    """Fast regression for script train_27_dpo.py's QLoRA setup.

    DPO should train Qwen3 LoRA adapters plus FACodec modules. Audio encoder and
    audio adapter must remain frozen, including no accidental LoRA adapters.
    """
    monkeypatch.setattr(
        train_amy_dpo.MossAudioModel,
        "from_pretrained",
        lambda *args, **kwargs: _tiny_moss(),
    )

    model = train_amy_dpo.init_model(
        DPOTrainingConfig(lora_r=2, lora_alpha=4, lora_dropout=0.0)
    )
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]

    assert any("moss.language_model" in name and "lora_" in name for name in trainable)
    assert any("prosody_embedding" in name for name in trainable)
    assert any("timbre_projection" in name for name in trainable)
    assert any("residual_fusion" in name for name in trainable)

    assert not any("moss.audio_encoder" in name and "lora_" in name for name in trainable)
    assert not any("moss.audio_adapter" in name and "lora_" in name for name in trainable)


def test_dpo_forward_backward_gradient_flow(monkeypatch):
    """Fast CPU test: DPO model forward-backward produces non-zero grads on
    LoRA adapters and FACodec gate params, while frozen audio encoder/adapter
    and LM base weights get None grads.

    Lambda gates start at zero (by design), so prosody_embedding and
    timbre_projection gradients are mathematically zero at init. We set
    lambda_p/t = 1.0 to prove the full gradient path exists.

    Uses the same _tiny_moss() mock as the static init test above.
    """
    monkeypatch.setattr(
        train_amy_dpo.MossAudioModel,
        "from_pretrained",
        lambda *args, **kwargs: _tiny_moss(),
    )

    model = train_amy_dpo.init_model(
        DPOTrainingConfig(lora_r=2, lora_alpha=4, lora_dropout=0.0)
    )
    model.train()
    torch.manual_seed(0)

    # Un-prompt lambda gates so prosody/timbre gradients flow through
    amy = _unwrap_amy(model)
    amy.residual_fusion.lambda_p.data.fill_(1.0)
    amy.residual_fusion.lambda_t.data.fill_(1.0)

    B, S = 1, 8
    mel_len = 20

    input_ids = torch.randint(0, 63, (B, S))
    attention_mask = torch.ones(B, S, dtype=torch.long)
    labels = input_ids.clone()

    audio_data = torch.randn(B, 128, mel_len)
    audio_data_seqlens = torch.full((B,), mel_len, dtype=torch.long)

    # 3 audio tokens after conv3 downsampling of mel_len=20
    audio_input_mask = torch.zeros(B, S, dtype=torch.bool)
    audio_input_mask[0, 2:5] = True

    prosody_indices = torch.randint(0, 1024, (B, 1, 20))
    timbre_vector = torch.randn(B, 256)

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_data=audio_data,
        audio_data_seqlens=audio_data_seqlens,
        audio_input_mask=audio_input_mask,
        labels=labels,
        prosody_indices=prosody_indices,
        timbre_vector=timbre_vector,
    )
    output.loss.backward()

    # ── LoRA adapters (Qwen3) ─────────────────────────────────────────────
    lora_grads = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            lora_grads.append((name, param.grad))
    assert len(lora_grads) > 0, "No LoRA parameters found"
    for name, grad in lora_grads:
        assert grad is not None, f"LoRA param {name!r} has None gradient"
        assert grad.abs().sum().item() > 0, f"LoRA param {name!r} has zero gradient"

    # ── FACodec gate lambdas (gradient control points) ────────────────────
    for gate_name in ("lambda_p", "lambda_t", "lambda_c", "lambda_a"):
        param = getattr(amy.residual_fusion, gate_name)
        g = param.grad
        assert g is not None, (
            f"residual_fusion.{gate_name} has None gradient "
            f"(graph cut before fusion?)"
        )
        assert g.abs().sum().item() > 0, (
            f"residual_fusion.{gate_name} has zero gradient"
        )

    # ── Prosody/timbre embeddings (lambda set to 1 → path is open) ────────
    facodec_embedding_params = []
    for name, param in amy.named_parameters():
        if "prosody_embedding" in name or "timbre_projection" in name:
            facodec_embedding_params.append((name, param.grad))
    assert len(facodec_embedding_params) > 0, "No prosody/timbre params found"
    for name, grad in facodec_embedding_params:
        assert grad is not None, (
            f"{name!r} has None gradient (graph cut after enrichment?)"
        )
        assert grad.abs().sum().item() > 0, (
            f"{name!r} has zero gradient (lambda may still be zero?)"
        )

    # ── Frozen modules: audio_encoder, audio_adapter, LM base ─────────────
    for name, param in model.named_parameters():
        if "audio_encoder" in name or "audio_adapter" in name:
            assert param.grad is None, (
                f"Frozen param {name!r} has non-None gradient "
                f"(sum={param.grad.abs().sum().item():.6f})"
            )
        if "moss.language_model" in name and "lora_" not in name:
            assert param.grad is None, (
                f"Frozen LM base param {name!r} has non-None gradient "
                f"(sum={param.grad.abs().sum().item():.6f})"
            )
