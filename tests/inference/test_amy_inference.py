import sys
import types

import pytest
import torch

from src.inference.amy_inference import AmyInference, parse_response, resample_audio
from src.models.fusion import ResidualFusion


def test_parse_response_yes_no_empty():
    assert parse_response("Yes") == 1
    assert parse_response("No") == 0
    assert parse_response("") == 0


def test_resample_audio_16k_passthrough():
    audio = torch.randn(16000)
    out = resample_audio(audio, 16000)
    assert out.shape == audio.shape
    assert torch.equal(out, audio)


def test_resample_audio_8k_resampled(monkeypatch):
    class _FakeResample:
        def __init__(self, orig_freq, new_freq):
            self.orig_freq = orig_freq
            self.new_freq = new_freq

        def __call__(self, x):
            target_len = int(x.shape[-1] * self.new_freq / self.orig_freq)
            return torch.nn.functional.interpolate(x.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0)

    fake_torchaudio = types.SimpleNamespace(
        transforms=types.SimpleNamespace(Resample=_FakeResample)
    )
    monkeypatch.setitem(sys.modules, "torchaudio", fake_torchaudio)
    audio = torch.randn(8000)
    out = resample_audio(audio, 8000)
    assert out.shape[0] != audio.shape[0]


def test_resample_audio_without_torchaudio(monkeypatch):
    monkeypatch.setitem(sys.modules, "torchaudio", None)
    audio = torch.randn(8000)
    out = resample_audio(audio, 8000)
    assert torch.equal(out, audio)


class _DummyAmyMoss:
    def __init__(self):
        self.residual_fusion = ResidualFusion(hidden_dim=2560)

    def encode_enriched_audio_embeds(self, audio, prosody_indices=None, timbre_vector=None):
        torch.manual_seed(7)
        return torch.randn(audio.shape[0], 12, 2560), torch.full((audio.shape[0],), 12, dtype=torch.long)


class _DummyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(128, 2560)
        self.dummy = torch.nn.Parameter(torch.tensor(0.0))

    def get_input_embeddings(self):
        return self.embed

    def generate(self, **kwargs):
        return torch.tensor([[1, 2, 3]])


class _DummyModel:
    def __init__(self):
        self.amy_moss = _DummyAmyMoss()
        self._lm = _DummyLM()

    def get_language_model(self):
        return self._lm


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, chat, tokenize=False, add_generation_prompt=True):
        return chat[0]["content"]

    def encode(self, prompt, return_tensors="pt"):
        return torch.tensor([[10, 11, 12]])

    def decode(self, tokens, skip_special_tokens=True):
        return "Yes"


def test_compute_fused_h_shape_and_zero_lambda_baseline():
    inference = AmyInference(_DummyModel(), encoder=None, tokenizer=_DummyTokenizer())
    audio = torch.randn(1, 16000)
    prosody_indices = torch.randint(0, 1024, (1, 1, 80))
    timbre_vector = torch.randn(1, 256)

    fused = inference.compute_fused_h(audio, prosody_indices, timbre_vector)
    assert fused.shape == (1, 12, 2560)


def test_assemble_inputs_shape_and_mask():
    inference = AmyInference(_DummyModel(), encoder=None, tokenizer=_DummyTokenizer())
    fused_h = torch.randn(1, 12, 2560)
    inputs_embeds, attention_mask = inference._assemble_inputs(fused_h, "Is this sarcastic?")
    assert inputs_embeds.shape == (1, 15, 2560)
    assert attention_mask.shape == (1, 15)
    assert torch.all(attention_mask == 1)


def test_predict_returns_label_and_response():
    inference = AmyInference(_DummyModel(), encoder=None, tokenizer=_DummyTokenizer())
    out = inference.predict(
        audio=torch.randn(1, 16000),
        instruction="Is this sarcastic?",
        prosody_indices=torch.randint(0, 1024, (1, 1, 80)),
        timbre_vector=torch.randn(1, 256),
    )
    assert out["prediction"] == 1
    assert out["response"] == "Yes"


def test_evaluate_accuracy_and_results():
    inference = AmyInference(_DummyModel(), encoder=None, tokenizer=_DummyTokenizer())
    samples = [
        {"prediction": 1, "label": 1},
        {"prediction": 0, "label": 0},
        {"prediction": 1, "label": 0},
    ]
    out = inference.evaluate(samples)
    assert out["accuracy"] == pytest.approx(2 / 3)
    assert out["correct"] == 2
    assert out["total"] == 3
    assert len(out["results"]) == 3
