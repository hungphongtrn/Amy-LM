"""Tests for AmyTrainer -- vanilla PyTorch training loop."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


def _make_synthetic_loader(n_samples=4, n_frames=24000, n_prosody=120):
    """Create a DataLoader with synthetic data for testing.

    Returns loader compatible with AmyForProsodyClassification forward():
    audio [B, T_audio], prosody [B, 1, T80], timbre [B, 256], labels [B]
    """
    audio = torch.randn(n_samples, n_frames)
    prosody = torch.randint(0, 1024, (n_samples, 1, n_prosody))
    timbre = torch.randn(n_samples, 256)
    labels = torch.tensor([0, 1, 1, 0][:n_samples])
    ds = TensorDataset(audio, prosody, timbre, labels)
    return DataLoader(ds, batch_size=2)


class TestTrainerInit:
    """Verify trainer initializes with both Amy and Baseline models.

    These tests instantiate the 4B MOSS-Audio model and require GPU.
    """

    def test_amy_trainer_initialization(self, require_gpu, device):
        """Trainer initializes with an Amy model."""
        from src.models import AmyForProsodyClassification
        from src.training.trainer import AmyTrainer

        vectors = torch.randn(1024, 8)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
        trainer = AmyTrainer(model, device=device, is_baseline=False)
        assert trainer.optimizer is not None
        assert trainer.current_epoch == 0
        assert not trainer.is_baseline

    def test_baseline_trainer_initialization(self, require_gpu, device):
        """Trainer initializes with a BaselineClassifier."""
        from src.models import BaselineClassifier
        from src.training.trainer import AmyTrainer

        model = BaselineClassifier(device=device)
        trainer = AmyTrainer(model, device=device, is_baseline=True)
        assert trainer.is_baseline
        assert trainer.current_epoch == 0


class TestTrainerForward:
    """Verify training_step, train_epoch, and evaluate.

    These run full forward/backward passes through the 4B model. GPU required.
    """

    @pytest.fixture
    def amy_model(self, require_gpu, device):
        from src.models import AmyForProsodyClassification

        vectors = torch.randn(1024, 8)
        return AmyForProsodyClassification(warm_start_vectors=vectors, device=device)

    @pytest.fixture
    def loader(self):
        return _make_synthetic_loader()

    def test_train_epoch_returns_metrics(self, amy_model, loader, device):
        """train_epoch runs and returns metrics dict with expected keys."""
        from src.training.trainer import AmyTrainer

        trainer = AmyTrainer(amy_model, device=device, is_baseline=False)
        metrics = trainer.train_epoch(loader)
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "train_f1" in metrics
        assert 0.0 <= metrics["train_accuracy"] <= 1.0

    def test_evaluate_returns_metrics(self, amy_model, loader, device):
        """evaluate returns metrics dict with expected keys."""
        from src.training.trainer import AmyTrainer

        trainer = AmyTrainer(amy_model, device=device, is_baseline=False)
        metrics = trainer.evaluate(loader)
        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert "val_f1" in metrics

    def test_baseline_training_step_runs(self, loader, device):
        """Baseline model's training_step ignores prosody/timbre args."""
        from src.models import BaselineClassifier
        from src.training.trainer import AmyTrainer

        model = BaselineClassifier(device=device)
        trainer = AmyTrainer(model, device=device, is_baseline=True)
        trainer.train_epoch(loader)

    def test_training_step_returns_correct_types(self, amy_model, loader, device):
        """training_step returns (loss, logits [B,2], labels [B])."""
        from src.training.trainer import AmyTrainer

        trainer = AmyTrainer(amy_model, device=device, is_baseline=False)
        batch = next(iter(loader))
        loss, logits, labels = trainer.training_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert logits.shape[1] == 2
        assert logits.shape[0] == labels.shape[0]


class TestCheckpoint:
    """Verify save/load roundtrip.

    Instantiates the 4B MOSS-Audio model. GPU required.
    """

    def test_save_load_roundtrip(self, tmp_path, require_gpu, device):
        from src.models import BaselineClassifier
        from src.training.trainer import AmyTrainer

        model = BaselineClassifier(device=device)
        trainer = AmyTrainer(model, device=device, is_baseline=True)
        trainer.current_epoch = 5
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(path))

        model2 = BaselineClassifier(device=device)
        trainer2 = AmyTrainer(model2, device=device, is_baseline=True)
        trainer2.load_checkpoint(str(path))

        assert trainer2.current_epoch == 5
        assert torch.equal(
            trainer.model.classifier.weight, trainer2.model.classifier.weight
        )


class TestLambdaLogging:
    """Verify lambda_p and lambda_t are logged for Amy, excluded for baseline.

    These instantiate models but do no forward pass. Still require GPU
    for model loading.
    """

    def test_amy_logs_lambdas(self, require_gpu, device):
        """Amy model's _get_lambdas returns lambda_p and lambda_t."""
        from src.models import AmyForProsodyClassification
        from src.training.trainer import AmyTrainer

        vectors = torch.randn(1024, 8)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
        trainer = AmyTrainer(model, device=device, is_baseline=False)
        lambdas = trainer._get_lambdas()
        assert "lambda_p" in lambdas
        assert "lambda_t" in lambdas
        assert lambdas["lambda_p"] == 1.0
        assert lambdas["lambda_t"] == 1.0

    def test_baseline_returns_empty_lambdas(self, require_gpu, device):
        from src.models import BaselineClassifier
        from src.training.trainer import AmyTrainer

        model = BaselineClassifier(device=device)
        trainer = AmyTrainer(model, device=device, is_baseline=True)
        assert trainer._get_lambdas() == {}
