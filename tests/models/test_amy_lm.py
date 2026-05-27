"""Tests for AmyMossLM — composition-based Speech LM (Issue #26)."""
import os
import pytest
import torch

from src.models.moss_audio_model import MossAudioConfig, MossAudioModel
from src.models.amy_lm import AmyMossLMConfig, AmyMossLM


def _tiny_config(**overrides) -> AmyMossLMConfig:
    """Minimal config for fast CPU-friendly shape checks.

    hidden_dim, prosody_vocab_size, and timbre_dim should be set via
    **overrides to match the tiny language_model dimensions.
    """
    lang = {
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    audio = {
        "d_model": 64,
        "output_dim": 64,
        "encoder_layers": 2,
        "encoder_attention_heads": 2,
        "deepstack_encoder_layer_indexes": [],
    }
    lang.update(overrides.pop("language_config", {}))
    audio.update(overrides.pop("audio_config", {}))
    return AmyMossLMConfig(
        moss_config={"language_config": lang, "audio_config": audio},
        **overrides,
    )


class TestAmyMossLMConfig:
    def test_model_type(self):
        config = _tiny_config()
        assert config.model_type == "amy_moss_lm"

    def test_nested_moss_config(self):
        config = _tiny_config()
        assert hasattr(config, "moss_config")
        assert isinstance(config.moss_config, MossAudioConfig)

    def test_facodec_fields(self):
        config = _tiny_config()
        assert config.prosody_vocab_size == 1024
        assert config.prosody_init_strategy == "random"
        assert config.timbre_dim == 256
        assert config.hidden_dim == 2560

    def test_freeze_defaults(self):
        config = _tiny_config()
        assert config.freeze_audio_encoder is True
        assert config.freeze_audio_adapter is True
        assert config.freeze_llm is True

    def test_custom_moss_config_dict(self):
        config = AmyMossLMConfig(moss_config={"language_config": {"vocab_size": 500}})
        assert config.moss_config.language_config.vocab_size == 500
        assert config.vocab_size == 500

    def test_custom_moss_config_object(self):
        mc = MossAudioConfig(language_config={"vocab_size": 777})
        config = AmyMossLMConfig(moss_config=mc)
        assert config.moss_config is mc
        assert config.vocab_size == 777

    def test_custom_values(self):
        config = _tiny_config(prosody_vocab_size=512, timbre_dim=128, freeze_audio_encoder=False)
        assert config.prosody_vocab_size == 512
        assert config.timbre_dim == 128
        assert config.freeze_audio_encoder is False

    def test_to_dict_roundtrip(self):
        config = _tiny_config(prosody_vocab_size=512, timbre_dim=128)
        d = config.to_dict()
        assert d["prosody_vocab_size"] == 512
        assert d["timbre_dim"] == 128
        assert "moss_config" in d
        assert d["model_type"] == "amy_moss_lm"
        restored = AmyMossLMConfig.from_dict(d)
        assert restored.prosody_vocab_size == 512
        assert restored.timbre_dim == 128
        assert restored.moss_config.language_config.vocab_size == config.moss_config.language_config.vocab_size

    def test_vocab_size_propagated(self):
        config = AmyMossLMConfig()
        assert config.vocab_size == config.moss_config.language_config.vocab_size

    def test_flash_attention_configured(self):
        config = AmyMossLMConfig()
        assert config.moss_config.language_config._attn_implementation == "flash_attention_2"


class TestAmyMossLMModelStructure:
    @pytest.fixture
    def config(self):
        return _tiny_config()

    @pytest.fixture
    def model(self, config):
        return AmyMossLM(config)

    def test_composition_not_inheritance(self, model):
        assert not isinstance(model, MossAudioModel)
        assert isinstance(model.moss, MossAudioModel)

    def test_has_moss_attribute(self, model):
        assert hasattr(model, "moss")

    def test_has_facodec_modules_on_self(self, model):
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "timbre_projection")
        assert hasattr(model, "temporal_pool")
        assert hasattr(model, "residual_fusion")

    def test_audio_modules_on_moss_not_self(self, model):
        assert hasattr(model.moss, "audio_encoder")
        assert hasattr(model.moss, "audio_adapter")
        assert hasattr(model.moss, "language_model")
        assert hasattr(model.moss, "lm_head")
        assert not hasattr(model, "audio_encoder")

    def test_freeze_applies_to_moss_modules(self, model):
        for p in model.moss.audio_encoder.parameters():
            assert not p.requires_grad
        for p in model.moss.audio_adapter.parameters():
            assert not p.requires_grad
        for p in model.moss.language_model.parameters():
            assert not p.requires_grad

    def test_facodec_modules_are_trainable(self, model):
        if model.config.prosody_init_strategy == "random":
            for p in model.prosody_embedding.parameters():
                assert p.requires_grad
        for p in model.timbre_projection.parameters():
            assert p.requires_grad
        for name, p in model.residual_fusion.named_parameters():
            assert p.requires_grad, f"fusion.{name} should be trainable"

    def test_get_input_embeddings_delegates(self, model):
        emb = model.get_input_embeddings()
        moss_emb = model.moss.get_input_embeddings()
        assert emb is moss_emb

    def test_get_output_embeddings_delegates(self, model):
        out_emb = model.get_output_embeddings()
        moss_out_emb = model.moss.get_output_embeddings()
        assert out_emb is moss_out_emb

    def test_model_type_on_config(self, model):
        assert model.config.model_type == "amy_moss_lm"

    def test_base_model_prefix(self, model):
        assert model.base_model_prefix == "moss"

    def test_moss_audio_encoder_frozen_separately(self, model):
        for p in model.moss.audio_encoder.parameters():
            assert not p.requires_grad


class TestAmyMossLMConstructorInjection:
    def test_preloaded_moss_used(self):
        mc = MossAudioConfig(language_config={"vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 1})
        preloaded = MossAudioModel(mc)
        config = AmyMossLMConfig(moss_config=mc)
        model = AmyMossLM(config, moss=preloaded)
        assert model.moss is preloaded

    def test_facodec_modules_added_with_preloaded_moss(self):
        mc = MossAudioConfig(language_config={"vocab_size": 100, "hidden_size": 64, "num_hidden_layers": 1})
        preloaded = MossAudioModel(mc)
        config = AmyMossLMConfig(moss_config=mc)
        model = AmyMossLM(config, moss=preloaded)
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "residual_fusion")
        assert model.moss is preloaded


class TestAmyMossLMSaveLoad:
    @pytest.fixture
    def config(self):
        return _tiny_config()

    def test_save_pretrained_config(self, config, tmp_path):
        model = AmyMossLM(config)
        save_dir = tmp_path / "test_model"
        model.save_pretrained(str(save_dir))

        config_path = save_dir / "config.json"
        assert config_path.exists()

        loaded_config = AmyMossLMConfig.from_pretrained(str(save_dir))
        assert loaded_config.model_type == "amy_moss_lm"
        assert loaded_config.prosody_vocab_size == config.prosody_vocab_size
        assert loaded_config.timbre_dim == config.timbre_dim

    def test_save_pretrained_moss_key_prefix(self, config, tmp_path):
        model = AmyMossLM(config)
        save_dir = tmp_path / "test_model"
        model.save_pretrained(str(save_dir))

        safetensors_path = save_dir / "model.safetensors"
        assert safetensors_path.exists()

        from safetensors import safe_open
        with safe_open(str(safetensors_path), framework="pt") as f:
            keys = f.keys()
            moss_keys = [k for k in keys if k.startswith("moss.")]
            facodec_keys = [k for k in keys if "prosody_embedding" in k or "timbre_projection" in k]
            assert len(moss_keys) > 0, "MossAudio params should have moss. prefix"
            assert len(facodec_keys) > 0, "FACodec params should be saved"


class TestAmyMossLMForward:
    @pytest.fixture
    def config(self):
        return _tiny_config(
            hidden_dim=64,
            prosody_input_rate=80.0,
            prosody_output_rate=12.5,
            prosody_vocab_size=64,
            timbre_dim=32,
        )

    @pytest.fixture
    def model(self, config, require_gpu):
        config.moss_config.language_config._attn_implementation = "eager"
        return AmyMossLM(config).to("cuda")

    def test_forward_text_only(self, model):
        batch, seq = 1, 5
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert output.logits.shape == (batch, seq, 100)

    def test_forward_with_audio(self, model):
        batch, seq = 1, 20
        n_mels = 160
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        audio_data = torch.randn(batch, 128, n_mels, device="cuda")
        audio_data_seqlens = torch.tensor([n_mels], dtype=torch.long, device="cuda")
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        audio_input_mask[:, :] = True  # ~20 audio tokens from 160 mel frames

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_data=audio_data,
                audio_data_seqlens=audio_data_seqlens,
                audio_input_mask=audio_input_mask,
            )
        assert output.logits.shape == (batch, seq, 100)

    def test_forward_with_prosody(self, model):
        batch, seq = 1, 20
        n_mels = 160
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        audio_data = torch.randn(batch, 128, n_mels, device="cuda")
        audio_data_seqlens = torch.tensor([n_mels], dtype=torch.long, device="cuda")
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        audio_input_mask[:, :] = True  # ~20 audio tokens
        prosody_indices = torch.randint(0, 64, (batch, 1, 80), device="cuda")

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_data=audio_data,
                audio_data_seqlens=audio_data_seqlens,
                audio_input_mask=audio_input_mask,
                prosody_indices=prosody_indices,
            )
        assert output.logits.shape == (batch, seq, 100)

    def test_forward_with_timbre(self, model):
        batch, seq = 1, 20
        n_mels = 160
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        audio_data = torch.randn(batch, 128, n_mels, device="cuda")
        audio_data_seqlens = torch.tensor([n_mels], dtype=torch.long, device="cuda")
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        audio_input_mask[:, :] = True  # ~20 audio tokens
        timbre_vector = torch.randn(batch, 32, device="cuda")

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_data=audio_data,
                audio_data_seqlens=audio_data_seqlens,
                audio_input_mask=audio_input_mask,
                timbre_vector=timbre_vector,
            )
        assert output.logits.shape == (batch, seq, 100)

    def test_forward_with_both_prosody_and_timbre(self, model):
        batch, seq = 1, 20
        n_mels = 160
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        audio_data = torch.randn(batch, 128, n_mels, device="cuda")
        audio_data_seqlens = torch.tensor([n_mels], dtype=torch.long, device="cuda")
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        audio_input_mask[:, :] = True  # ~20 audio tokens
        prosody_indices = torch.randint(0, 64, (batch, 1, 80), device="cuda")
        timbre_vector = torch.randn(batch, 32, device="cuda")

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_data=audio_data,
                audio_data_seqlens=audio_data_seqlens,
                audio_input_mask=audio_input_mask,
                prosody_indices=prosody_indices,
                timbre_vector=timbre_vector,
            )
        assert output.logits.shape == (batch, seq, 100)

    def test_backward_compatible_no_facodec(self, model):
        batch, seq = 1, 15
        n_mels = 80
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        audio_data = torch.randn(batch, 128, n_mels, device="cuda")
        audio_data_seqlens = torch.tensor([n_mels], dtype=torch.long, device="cuda")
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        audio_input_mask[:, :10] = True  # ~10 audio tokens from 80 mel frames

        with torch.no_grad():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                audio_data=audio_data,
                audio_data_seqlens=audio_data_seqlens,
                audio_input_mask=audio_input_mask,
            )
        assert output.logits.shape == (batch, seq, 100)

    def test_gradient_flows_through_facodec(self, model):
        batch, seq = 1, 20
        n_mels = 160
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        audio_data = torch.randn(batch, 128, n_mels, device="cuda")
        audio_data_seqlens = torch.tensor([n_mels], dtype=torch.long, device="cuda")
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        audio_input_mask[:, :] = True  # ~20 audio tokens
        prosody_indices = torch.randint(0, 64, (batch, 1, 80), device="cuda")
        timbre_vector = torch.randn(batch, 32, device="cuda")

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            prosody_indices=prosody_indices,
            timbre_vector=timbre_vector,
        )
        loss = output.logits.sum()
        loss.backward()

        if model.config.prosody_init_strategy == "random":
            assert model.prosody_embedding.embedding.weight.grad is not None
        assert model.timbre_projection.linear.weight.grad is not None
        assert model.residual_fusion.lambda_p.grad is not None
        assert model.residual_fusion.lambda_t.grad is not None

    def test_frozen_moss_backbone_no_gradients(self, model):
        batch, seq = 1, 20
        n_mels = 160
        input_ids = torch.randint(0, 100, (batch, seq), device="cuda")
        attention_mask = torch.ones(batch, seq, dtype=torch.long, device="cuda")
        audio_data = torch.randn(batch, 128, n_mels, device="cuda")
        audio_data_seqlens = torch.tensor([n_mels], dtype=torch.long, device="cuda")
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool, device="cuda")
        audio_input_mask[:, :] = True  # ~20 audio tokens
        prosody_indices = torch.randint(0, 64, (batch, 1, 80), device="cuda")
        timbre_vector = torch.randn(batch, 32, device="cuda")

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            prosody_indices=prosody_indices,
            timbre_vector=timbre_vector,
        )
        loss = output.logits.sum()
        loss.backward()

        for name, p in model.moss.audio_encoder.named_parameters():
            assert p.grad is None, f"moss.audio_encoder.{name} should have no grad"
        for name, p in model.moss.audio_adapter.named_parameters():
            assert p.grad is None, f"moss.audio_adapter.{name} should have no grad"
