"""Tests for AmyLM — Issue #19."""
import os
import sys
import pytest
import torch

# Ensure MOSS-Audio path is set up for imports
_vendor_src = os.path.join(
    os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src"
)
if os.path.isdir(_vendor_src) and _vendor_src not in sys.path:
    sys.path.insert(0, _vendor_src)

from configuration_moss_audio import MossAudioConfig
from modeling_moss_audio import MossAudioModel
from src.models.amy_lm import AmyLMConfig, AmyLM


class TestAmyLMConfig:
    def test_config_extends_moss_audio_config(self):
        """AmyLMConfig is a subclass of MossAudioConfig."""
        config = AmyLMConfig()
        assert isinstance(config, MossAudioConfig)

    def test_config_model_type(self):
        """model_type is 'amy_lm'."""
        config = AmyLMConfig()
        assert config.model_type == "amy_lm"

    def test_config_facodec_fields(self):
        """Config includes FACodec-specific fields with defaults."""
        config = AmyLMConfig()
        assert config.prosody_vocab_size == 1024
        assert config.prosody_init_strategy == "random"
        assert config.timbre_dim == 256
        assert config.hidden_dim == 2560

    def test_config_freeze_defaults(self):
        """Default freeze: encoder, adapter, LLM frozen."""
        config = AmyLMConfig()
        assert config.freeze_audio_encoder is True
        assert config.freeze_audio_adapter is True
        assert config.freeze_llm is True

    def test_config_custom_values(self):
        """Custom values are preserved."""
        config = AmyLMConfig(
            prosody_vocab_size=512,
            timbre_dim=128,
            hidden_dim=1024,
            freeze_audio_encoder=False,
        )
        assert config.prosody_vocab_size == 512
        assert config.timbre_dim == 128
        assert config.hidden_dim == 1024
        assert config.freeze_audio_encoder is False

    def test_config_to_dict_roundtrip(self):
        """Config serializes and deserializes correctly for FACodec fields."""
        config = AmyLMConfig(prosody_vocab_size=512, timbre_dim=128)
        d = config.to_dict()
        assert d["prosody_vocab_size"] == 512
        assert d["timbre_dim"] == 128

        restored = AmyLMConfig.from_dict(d)
        assert restored.prosody_vocab_size == 512
        assert restored.timbre_dim == 128


class TestAmyLMModelStructure:
    @pytest.fixture
    def config(self):
        return AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )

    @pytest.fixture
    def model(self, config):
        return AmyLM(config)

    def test_model_inherits_moss_audio(self, model):
        """AmyLM is a MossAudioModel."""
        assert isinstance(model, MossAudioModel)

    def test_model_has_facodec_modules(self, model):
        """Model has prosody, timbre, pooling, and fusion modules."""
        assert hasattr(model, "prosody_embedding")
        assert hasattr(model, "timbre_projection")
        assert hasattr(model, "temporal_pool")
        assert hasattr(model, "residual_fusion")

    def test_model_has_audio_modules(self, model):
        """Model has audio encoder, adapter, and language model."""
        assert hasattr(model, "audio_encoder")
        assert hasattr(model, "audio_adapter")
        assert hasattr(model, "language_model")

    def test_model_has_lm_head(self, model):
        """Model has lm_head for text generation."""
        assert hasattr(model, "lm_head")

    def test_freeze_applies_correctly(self, model):
        """Audio encoder, adapter, LLM are frozen by default."""
        for p in model.audio_encoder.parameters():
            assert not p.requires_grad
        for p in model.audio_adapter.parameters():
            assert not p.requires_grad
        for p in model.language_model.parameters():
            assert not p.requires_grad

    def test_facodec_modules_are_trainable(self, model):
        """FACodec modules are trainable by default."""
        # Prosody embedding (random init mode)
        trainable = sum(1 for p in model.prosody_embedding.parameters() if p.requires_grad)
        total = sum(1 for p in model.prosody_embedding.parameters())
        assert trainable == total, f"Only {trainable}/{total} prosody params trainable"

        # Timbre projection
        trainable = sum(1 for p in model.timbre_projection.parameters() if p.requires_grad)
        total = sum(1 for p in model.timbre_projection.parameters())
        assert trainable == total, f"Only {trainable}/{total} timbre params trainable"

        # Lambda gates in fusion
        for name, p in model.residual_fusion.named_parameters():
            assert p.requires_grad, f"fusion.{name} should be trainable"

    def test_get_input_embeddings(self, model):
        """get_input_embeddings returns token embeddings."""
        emb = model.get_input_embeddings()
        assert emb is not None


class TestAmyLMForward:
    @pytest.fixture
    def config(self):
        return AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )

    @pytest.fixture
    def model(self, config):
        return AmyLM(config)

    def test_forward_text_only(self, model):
        """Forward with just text tokens (no audio, no FACodec)."""
        batch, seq = 2, 10
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)

        output = model(input_ids=input_ids, attention_mask=attention_mask)
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)

    def test_forward_with_audio(self, model):
        """Forward with audio_data works (no FACodec enrichment)."""
        batch, seq = 2, 20
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :15] = True

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
        )
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)

    def test_forward_with_prosody_indices(self, model):
        """Forward with prosody_indices enriches audio embeddings."""
        batch, seq = 2, 20
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :15] = True

        prosody_indices = torch.randint(0, 1024, (batch, 1, 80))

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            prosody_indices=prosody_indices,
        )
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)

    def test_forward_with_timbre_vector(self, model):
        """Forward with timbre_vector enriches audio embeddings."""
        batch, seq = 2, 20
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :15] = True

        timbre_vector = torch.randn(batch, 256)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            timbre_vector=timbre_vector,
        )
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)

    def test_forward_with_both_prosody_and_timbre(self, model):
        """Forward with both prosody and timbre."""
        batch, seq = 2, 20
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :15] = True

        prosody_indices = torch.randint(0, 1024, (batch, 1, 80))
        timbre_vector = torch.randn(batch, 256)

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
            prosody_indices=prosody_indices,
            timbre_vector=timbre_vector,
        )
        assert hasattr(output, "logits")
        assert output.logits.shape[:2] == (batch, seq)

    def test_backward_compatible_no_facodec_inputs(self, model):
        """Without prosody/timbre, forward works (backward compatible)."""
        batch, seq = 2, 15
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :10] = True

        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            audio_data=audio_data,
            audio_data_seqlens=audio_data_seqlens,
            audio_input_mask=audio_input_mask,
        )
        assert output.logits.shape[:2] == (batch, seq)

    def test_gradient_flows_through_facodec_modules(self, model):
        """Gradient flows through prosody embedding and timbre projection."""
        batch, seq = 2, 20
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :15] = True

        prosody_indices = torch.randint(0, 1024, (batch, 1, 80))
        timbre_vector = torch.randn(batch, 256)

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

        # FACodec modules should have gradients
        assert model.prosody_embedding.embedding.weight.grad is not None
        assert model.timbre_projection.linear.weight.grad is not None

        # Lambda gates should have gradients
        assert model.residual_fusion.lambda_p.grad is not None
        assert model.residual_fusion.lambda_t.grad is not None

    def test_frozen_backbone_has_no_gradients(self, model):
        """Frozen encoder, adapter, LLM have no gradients after backward."""
        batch, seq = 2, 20
        input_ids = torch.randint(0, 1000, (batch, seq))
        attention_mask = torch.ones(batch, seq, dtype=torch.long)
        audio_data = torch.randn(batch, 128, 3000)
        audio_data_seqlens = torch.tensor([3000, 3000], dtype=torch.long)
        audio_input_mask = torch.zeros(batch, seq, dtype=torch.bool)
        audio_input_mask[:, :15] = True

        prosody_indices = torch.randint(0, 1024, (batch, 1, 80))
        timbre_vector = torch.randn(batch, 256)

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

        # Audio encoder should have no gradients
        for name, p in model.audio_encoder.named_parameters():
            assert p.grad is None, f"audio_encoder.{name} should have no grad"

        # Audio adapter should have no gradients
        for name, p in model.audio_adapter.named_parameters():
            assert p.grad is None, f"audio_adapter.{name} should have no grad"


class TestAmyLMSaveLoad:
    @pytest.fixture
    def config(self):
        return AmyLMConfig(
            language_config={"vocab_size": 1000, "hidden_size": 2560},
        )

    def test_save_pretrained_config(self, config, tmp_path):
        """save_pretrained saves config.json with AmyLMConfig fields."""
        model = AmyLM(config)
        save_dir = tmp_path / "test_model"
        model.save_pretrained(str(save_dir))

        config_path = save_dir / "config.json"
        assert config_path.exists()

        loaded_config = AmyLMConfig.from_pretrained(str(save_dir))
        assert loaded_config.model_type == "amy_lm"
        assert loaded_config.prosody_vocab_size == config.prosody_vocab_size
        assert loaded_config.timbre_dim == config.timbre_dim
