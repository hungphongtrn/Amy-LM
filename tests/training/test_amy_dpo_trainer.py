"""Tests for AmyDPOTrainer."""

from __future__ import annotations

from unittest.mock import MagicMock

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


class MockAmyLM(nn.Module):
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
        AmyDPOTrainer(model=MockAmyLM(), args=config)


def test_default_config_sets_precompute(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)

    trainer = AmyDPOTrainer(model=MockAmyLM())
    assert trainer.args.precompute_ref_log_probs is True


def test_lambda_logging(monkeypatch):
    monkeypatch.setattr(DPOTrainer, "__init__", _stub_dpo_init)
    monkeypatch.setattr(DPOTrainer, "log", lambda self, logs, *args, **kwargs: None)

    model = MockAmyLM()
    model.residual_fusion.lambda_p.data = torch.tensor(0.5)
    model.residual_fusion.lambda_t.data = torch.tensor(0.3)

    trainer = AmyDPOTrainer(model=model)

    logs = {"loss": 0.5, "rewards/margins": 0.1}
    trainer.log(logs)

    assert "lambda_p" in logs
    assert "lambda_t" in logs
    assert abs(logs["lambda_p"] - 0.5) < 1e-6
    assert abs(logs["lambda_t"] - 0.3) < 1e-6
