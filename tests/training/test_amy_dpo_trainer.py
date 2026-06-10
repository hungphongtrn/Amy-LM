"""Tests for AmyDPOTrainer."""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
import torch
import torch.nn as nn
from trl import DPOConfig, DPOTrainer

from src.training.amy_dpo_trainer import AmyDPOTrainer


class MockResidualFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_p = nn.Parameter(torch.tensor(0.0))
        self.lambda_t = nn.Parameter(torch.tensor(0.0))


class MockAmyMossLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_fusion = MockResidualFusion()

    def forward(self, **kwargs):
        input_ids = kwargs.get("input_ids", torch.zeros(1, 1, dtype=torch.long))
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        vocab_size = 152064
        logits = torch.randn(batch_size, seq_len, vocab_size)
        return type("MockOutput", (), {"logits": logits, "loss": torch.tensor(0.0)})()


def _stub_dpo_init(self, model, ref_model=None, args=None, **kwargs):
    self.model = model
    self.ref_model = ref_model
    self.args = args
    self.accelerator = MagicMock()
    self.accelerator.unwrap_model = MagicMock(return_value=model)


def test_precompute_ref_log_probs_required(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)

    config = DPOConfig(output_dir="./tmp", precompute_ref_log_probs=False)
    with pytest.raises(ValueError, match="precompute_ref_log_probs=True"):
        AmyDPOTrainer(model=MockAmyMossLM(), args=config)


def test_default_config_sets_precompute(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)

    trainer = AmyDPOTrainer(model=MockAmyMossLM())
    assert trainer.args.precompute_ref_log_probs is True


def test_lambda_logging(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)
    monkeypatch.setattr(DPOTrainer, "log", lambda self, logs, *args, **kwargs: None)

    model = MockAmyMossLM()
    model.residual_fusion.lambda_p.data = torch.tensor(0.5)
    model.residual_fusion.lambda_t.data = torch.tensor(0.3)

    trainer = AmyDPOTrainer(model=model)

    logs = {"loss": 0.5, "rewards/margins": 0.1}
    trainer.log(logs)

    assert "lambda_p" in logs
    assert "lambda_t" in logs
    assert abs(logs["lambda_p"] - 0.5) < 1e-6
    assert abs(logs["lambda_t"] - 0.3) < 1e-6


def test_saved_lambda_grads_initialized(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)

    trainer = AmyDPOTrainer(model=MockAmyMossLM())
    assert hasattr(trainer, "_saved_lambda_grads")
    assert trainer._saved_lambda_grads == {}


def test_lambda_grad_hooks_registered(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)

    model = MockAmyMossLM()
    trainer = AmyDPOTrainer(model=model)

    lambda_p = model.residual_fusion.lambda_p
    lambda_t = model.residual_fusion.lambda_t

    assert len(lambda_p._backward_hooks) == 1
    assert len(lambda_t._backward_hooks) == 1


def test_lambda_grad_hook_fires(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)

    model = MockAmyMossLM()
    trainer = AmyDPOTrainer(model=model)

    lambda_p = model.residual_fusion.lambda_p
    lambda_p.grad = torch.tensor(0.42)

    for handle in lambda_p._backward_hooks.values():
        handle(lambda_p.grad)

    assert "lambda_p" in trainer._saved_lambda_grads
    assert abs(trainer._saved_lambda_grads["lambda_p"] - 0.42) < 1e-6


def test_log_injects_lambda_grads_when_populated(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)
    monkeypatch.setattr(DPOTrainer, "log", lambda self, logs, *args, **kwargs: None)

    model = MockAmyMossLM()
    model.residual_fusion.lambda_p.data = torch.tensor(0.5)
    model.residual_fusion.lambda_t.data = torch.tensor(0.3)

    trainer = AmyDPOTrainer(model=model)
    trainer._saved_lambda_grads = {"lambda_p": 0.1, "lambda_t": 0.2}

    logs = {"loss": 0.5}
    trainer.log(logs)

    assert abs(logs["lambda_p"] - 0.5) < 1e-6
    assert abs(logs["lambda_t"] - 0.3) < 1e-6
    assert logs["lambda_p_grad"] == 0.1
    assert logs["lambda_t_grad"] == 0.2


def test_log_skips_grads_when_not_populated(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)
    monkeypatch.setattr(DPOTrainer, "log", lambda self, logs, *args, **kwargs: None)

    model = MockAmyMossLM()
    trainer = AmyDPOTrainer(model=model)
    assert trainer._saved_lambda_grads == {}

    logs = {"loss": 0.5}
    trainer.log(logs)

    assert "lambda_p_grad" not in logs
    assert "lambda_t_grad" not in logs


def test_log_handles_missing_fusion_gracefully(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)
    monkeypatch.setattr(DPOTrainer, "log", lambda self, logs, *args, **kwargs: None)

    class ModelWithoutFusion(nn.Module):
        pass

    trainer = AmyDPOTrainer(model=ModelWithoutFusion())
    logs = {"loss": 0.5}
    trainer.log(logs)

    assert "lambda_p" not in logs


def test_log_handles_peft_like_model(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)
    monkeypatch.setattr(DPOTrainer, "log", lambda self, logs, *args, **kwargs: None)

    base = MockAmyMossLM()
    base.residual_fusion.lambda_p.data = torch.tensor(0.7)

    class PEFTLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base

    trainer = AmyDPOTrainer(model=PEFTLikeModel())
    logs = {"loss": 0.5}
    trainer.log(logs)

    assert abs(logs["lambda_p"] - 0.7) < 1e-6


def test_hooks_registered_for_peft_like_model(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)

    base = MockAmyMossLM()

    class PEFTLikeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base_model = base

    trainer = AmyDPOTrainer(model=PEFTLikeModel())

    assert len(base.residual_fusion.lambda_p._backward_hooks) == 1
    assert len(base.residual_fusion.lambda_t._backward_hooks) == 1
