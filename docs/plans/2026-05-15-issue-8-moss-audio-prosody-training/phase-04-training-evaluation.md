# Phase 4: Training & Evaluation

## Phase Goal
Train MOSS-Audio baseline and Amy model on MUStARD. Report accuracy, F1, and loss. Log to W&B (metrics + lambdas). Evaluate on test split.

## Initial Assessment (Kitchen Sink)

### Data Status
Preprocessing running in background (PID 401355, `logs/preprocess_mustard_20260515_142141.log`). Target: 698 rows in `data/processed/mustard-processed/train.parquet`. Verify with:
```bash
uv run python -c "from datasets import Dataset; ds=Dataset.from_parquet('data/processed/mustard-processed/train.parquet'); print(len(ds), ds.column_names)"
# Expected: 698, ['audio', 'label', 'prosody_codebooks_idx', 'timbre_vector', ...]
```

### Model API (Phase 2)
```python
from src.models import AmyForProsodyClassification
from src.models.codebook_utils import load_prosody_codebook_vectors

vectors = load_prosody_codebook_vectors("checkpoints/facodec/ns3_facodec_decoder.bin")
model = AmyForProsodyClassification(warm_start_vectors=vectors, device="cuda")
logits = model(audio, prosody_indices, timbre_vector)  # [B, 2]
```

### Data API (Phase 3)
```python
from src.data.mustard_dataset import MustardDataset, collate_mustard, create_mustard_splits
from torch.utils.data import DataLoader

train, val, test = create_mustard_splits("data/processed/mustard-processed/train.parquet", seed=42)
loader = DataLoader(train, batch_size=2, collate_fn=collate_mustard)
# batch → (audio [B, max_T], prosody [B, 1, max_T80], timbre [B, 256],
#           labels [B], audio_lengths [B], prosody_lengths [B])
```

### Key Architecture Decisions (from CONTEXT.md + decisions.md)
- **Vanilla PyTorch loop** (no Lightning, no HF Trainer) — ~50 lines, easier to debug
- **Frozen MOSS-Audio backbone**, trainable FACodec modules + classifier
- **CrossEntropyLoss**, AdamW optimizer
- **W&B logging** for loss, accuracy, F1, λ_p, λ_t per epoch
- **Online Semantic Encoding** — MOSS-Audio encoder runs during forward pass
- **Gradient accumulation** needed for 4B backbone + small batch sizes (1-2)
- **λ zero-init** guarantees model = MOSS-Audio at step 0
- **dtype bridge** — semantic (bfloat16) → fusion (float32) → LM (bfloat16) → classifier (float32)

## Files to Touch

| File | Action | Purpose |
|------|--------|---------|
| `src/models/baseline_classifier.py` | Create | MOSS-Audio frozen backbone + Linear(2560→2) classifier only. No FACodec streams. Must produce identical output to `AmyForProsodyClassification` with lambdas=0 when given the same classifier weights. |
| `src/models/__init__.py` | Modify | Export `BaselineClassifier` |
| `src/training/__init__.py` | Create | Exports: `AmyTrainer` |
| `src/training/trainer.py` | Create | `AmyTrainer`: vanilla PyTorch training loop for binary classification. Works with both `BaselineClassifier` and `AmyForProsodyClassification`. Handles gradient accumulation, metric computation, W&B logging, checkpoint save/load. |
| `scripts/train_amy.py` | Create | CLI entry point. Parses args, loads data, creates model, runs trainer, evaluates on test set. |
| `tests/models/test_baseline_classifier.py` | Create | Tests for `BaselineClassifier`: forward shape, backbone frozen, classifier trainable, equivalence with Amy at λ=0 given same classifier weights. |
| `tests/training/__init__.py` | Create | Empty |
| `tests/training/test_trainer.py` | Create | Tests for `AmyTrainer`: training step runs, evaluate returns metrics dict, checkpoint save/load roundtrip, lambdas logged for Amy model, lambdas excluded for baseline. |

## Tasks

### Task 1: `BaselineClassifier` Model

**Files:**
- Create: `src/models/baseline_classifier.py`
- Modify: `src/models/__init__.py`
- Test: `tests/models/test_baseline_classifier.py`

**Purpose:** Simple MOSS-Audio + Linear(2560→2) classifier with no FACodec streams. The semantic-only baseline for measuring prosody/timbre contribution. Must produce identical logits to `AmyForProsodyClassification` at λ=0 given identical random seeds for the classifier layer.

**Design**: Forward path is a subset of `AmyForProsodyClassification.forward()`:
1. `semantic = wrapper.encode_semantic(audio)` → [B, T_moss, 2560] bfloat16
2. Convert to LM dtype, feed through `language_model(inputs_embeds=H)` → hidden states
3. `pooled = lm_out.mean(dim=1)` → [B, 2560] float32
4. `logits = classifier(pooled)` → [B, 2]

- [ ] **Step 1: Write failing test — `test_baseline_forward_output_shape`**

```python
# tests/models/test_baseline_classifier.py
import pytest
import torch
from src.models.baseline_classifier import BaselineClassifier


class TestBaselineForwardShape:
    @pytest.fixture
    def model(self):
        return BaselineClassifier(device="cpu")

    def test_forward_output_shape(self, model):
        """Forward pass produces [B, 2] logits."""
        audio = torch.randn(2, 32000)
        logits = model(audio)
        assert logits.shape == (2, 2)
        assert logits.dtype == torch.float32

    def test_single_sample_batch(self, model):
        """Handles batch_size=1."""
        audio = torch.randn(1, 16000)
        logits = model(audio)
        assert logits.shape == (1, 2)
```

Run: `pytest tests/models/test_baseline_classifier.py::TestBaselineForwardShape -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.models.baseline_classifier'`

- [ ] **Step 2: Write minimal implementation**

```python
# src/models/baseline_classifier.py
"""Baseline classifier: MOSS-Audio frozen backbone + Linear(2560→2)."""

from __future__ import annotations

import torch
import torch.nn as nn

from .moss_audio import MossAudioWrapper


class BaselineClassifier(nn.Module):
    """MOSS-Audio semantic encoder → Qwen3 language model → mean-pool → Linear classifier.

    Frozen MOSS-Audio backbone. Trainable: Linear(2560→2) classifier head.
    No FACodec streams. Used as the baseline for measuring prosody/timbre contribution.
    """

    def __init__(
        self,
        moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        device: torch.device | str = "cpu",
        num_classes: int = 2,
        hidden_dim: int = 2560,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        self.wrapper = MossAudioWrapper(model_id=moss_model_id, device=self.device)
        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._freeze_backbone()
        self._ensure_head_trainable()

    def _freeze_backbone(self) -> None:
        for param in self.wrapper.parameters():
            param.requires_grad = False

    def _ensure_head_trainable(self) -> None:
        for param in self.classifier.parameters():
            param.requires_grad = True

    def get_language_model(self) -> nn.Module:
        return self.wrapper.language_model

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Forward pass: raw audio → 2-class logits.

        Args:
            audio: Raw waveform [B, T_audio] at 16kHz.

        Returns:
            Logits [B, 2] for binary classification.
        """
        with torch.no_grad():
            semantic = self.wrapper.encode_semantic(audio)
        semantic = semantic.float()

        language_model = self.get_language_model()
        lm_dtype = next(language_model.parameters()).dtype
        H_lm = semantic.to(dtype=lm_dtype)
        lm_out = language_model(inputs_embeds=H_lm).last_hidden_state
        lm_out = lm_out.float()

        pooled = lm_out.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
```

Run: `pytest tests/models/test_baseline_classifier.py::TestBaselineForwardShape -v`
Expected: PASS

- [ ] **Step 3: Write failing test — `test_backbone_frozen`**

```python
class TestBaselineFreeze:
    @pytest.fixture
    def model(self):
        return BaselineClassifier(device="cpu")

    def test_backbone_fully_frozen(self, model):
        for name, param in model.wrapper.named_parameters():
            assert not param.requires_grad, (
                f"Backbone param '{name}' should be frozen"
            )

    def test_classifier_is_trainable(self, model):
        for param in model.classifier.parameters():
            assert param.requires_grad
```

Run: `pytest tests/models/test_baseline_classifier.py::TestBaselineFreeze -v`
Expected: PASS (these tests verify the constructor logic which is already implemented)

- [ ] **Step 4: Write equivalence test — `test_baseline_equals_amy_at_zero_lambda`**

```python
class TestBaselineAmyEquivalence:
    """BaselineClassifier must produce identical logits to AmyForProsodyClassification
    when lambdas=0 and both have the same classifier weights."""

    @pytest.fixture
    def audio(self):
        return torch.randn(1, 32000)

    @pytest.fixture
    def prosody_indices(self):
        return torch.randint(0, 1024, (1, 1, 160))

    @pytest.fixture
    def timbre_vector(self):
        return torch.randn(1, 256)

    def test_forward_equivalence_at_zero_lambda(self, audio, prosody_indices, timbre_vector):
        """With identical classifier weights, baseline and Amy (λ=0) logits must match."""
        from src.models import AmyForProsodyClassification
        from copy import deepcopy

        vectors = torch.randn(1024, 8)
        baseline = BaselineClassifier(device="cpu")
        amy = AmyForProsodyClassification(warm_start_vectors=vectors, device="cpu")

        # Share classifier weights
        amy.classifier.load_state_dict(deepcopy(baseline.classifier.state_dict()))
        assert torch.equal(baseline.classifier.weight, amy.classifier.weight)

        with torch.no_grad():
            logits_baseline = baseline(audio)
            logits_amy = amy(audio, prosody_indices, timbre_vector)

        assert torch.allclose(logits_baseline, logits_amy, atol=1e-4)
```

Run: `pytest tests/models/test_baseline_classifier.py::TestBaselineAmyEquivalence -v`
Expected: PASS (issue #8 baseline equivalence is a fundamental invariant)

- [ ] **Step 5: Export `BaselineClassifier`**

```python
# In src/models/__init__.py, add:
from .baseline_classifier import BaselineClassifier

# In __all__, add:
"BaselineClassifier",
```

- [ ] **Step 6: Commit**

```bash
git add src/models/baseline_classifier.py src/models/__init__.py tests/models/test_baseline_classifier.py
git commit -m "feat: add BaselineClassifier (MOSS-Audio + Linear head, no FACodec)"
```

---

### Task 2: Training Loop — `AmyTrainer`

**Files:**
- Create: `src/training/__init__.py`
- Create: `src/training/trainer.py`
- Test: `tests/training/__init__.py` (empty)
- Test: `tests/training/test_trainer.py`

**Purpose:** Vanilla PyTorch training loop for binary classification. One `AmyTrainer` class that handles both `BaselineClassifier` and `AmyForProsodyClassification`. Controls gradient accumulation, metric computation, W&B logging, and checkpoint save/load.

**Design:**
```python
class AmyTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 4,
        log_wandb: bool = False,
        is_baseline: bool = False,
    ):
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps
        self.log_wandb = log_wandb
        self.is_baseline = is_baseline
        self.current_epoch = 0

    def training_step(self, batch):
        """Forward + loss computation. Returns (loss, logits, labels)."""
        audio, prosody, timbre, labels, audio_lengths, prosody_lengths = batch
        audio = audio.to(self.device)
        labels = labels.to(self.device)

        if self.is_baseline:
            logits = self.model(audio)
        else:
            prosody = prosody.to(self.device)
            timbre = timbre.to(self.device)
            logits = self.model(audio, prosody, timbre)

        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()

    def train_epoch(self, dataloader):
        """Train for one epoch. Returns metrics dict."""
        self.model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        self.optimizer.zero_grad()
        n_batches = len(dataloader)

        for i, batch in enumerate(dataloader):
            loss, logits, labels = self.training_step(batch)
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == n_batches:
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item()
            all_preds.append(logits)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels, prefix="train")
        metrics["train_loss"] = total_loss / n_batches

        if self.log_wandb:
            import wandb
            wandb.log({**metrics, **self._get_lambdas(), "epoch": self.current_epoch})

        self.current_epoch += 1
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate on validation/test set. Returns metrics dict."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in dataloader:
            loss, logits, labels = self.training_step(batch)
            total_loss += loss.item()
            all_preds.append(logits)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels, prefix="val")
        metrics["val_loss"] = total_loss / len(dataloader)
        return metrics

    def _compute_metrics(self, logits, labels, prefix="train"):
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        f1 = self._compute_f1(preds, labels)
        return {f"{prefix}_accuracy": acc, f"{prefix}_f1": f1}

    def _compute_f1(self, preds, labels):
        """Binary macro F1 score. Manual computation to avoid sklearn dependency."""
        tp = ((preds == 1) & (labels == 1)).float().sum().item()
        fp = ((preds == 1) & (labels == 0)).float().sum().item()
        fn = ((preds == 0) & (labels == 1)).float().sum().item()
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        return 2 * prec * rec / (prec + rec + 1e-8)

    def _get_lambdas(self):
        """Read λ values from Amy model. Baseline returns empty dict."""
        if self.is_baseline:
            return {}
        return {
            "lambda_p": self.model.fusion.lambda_p.item(),
            "lambda_t": self.model.fusion.lambda_t.item(),
        }

    def save_checkpoint(self, path):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "is_baseline": self.is_baseline,
        }, path)

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
```

- [ ] **Step 1: Write failing test — `test_trainer_initialization`**

```python
# tests/training/test_trainer.py
import pytest
import torch
from src.training.trainer import AmyTrainer


class TestTrainerInit:
    def test_amy_trainer_initialization(self):
        """Trainer initializes with an Amy model."""
        from src.models import AmyForProsodyClassification
        vectors = torch.randn(1024, 8)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device="cpu")
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=False)
        assert trainer.optimizer is not None
        assert trainer.current_epoch == 0

    def test_baseline_trainer_initialization(self):
        """Trainer initializes with a BaselineClassifier."""
        from src.models import BaselineClassifier
        model = BaselineClassifier(device="cpu")
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=True)
        assert trainer.is_baseline
```

Run: `pytest tests/training/test_trainer.py::TestTrainerInit -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.training.trainer'`

- [ ] **Step 2: Implement `src/training/trainer.py`** — write the full implementation above.

Run: `pytest tests/training/test_trainer.py::TestTrainerInit -v`
Expected: PASS

- [ ] **Step 3: Write test — `test_train_epoch_runs`**

```python
class TestTrainerForward:
    @pytest.fixture
    def model(self):
        from src.models import AmyForProsodyClassification
        vectors = torch.randn(1024, 8)
        return AmyForProsodyClassification(warm_start_vectors=vectors, device="cpu")

    @pytest.fixture
    def loader(self):
        from torch.utils.data import DataLoader, TensorDataset
        # 4 samples, 1.5s audio each, binary labels
        audio = torch.randn(4, 24000)
        prosody = torch.randint(0, 1024, (4, 1, 120))
        timbre = torch.randn(4, 256)
        labels = torch.tensor([0, 1, 1, 0])
        ds = TensorDataset(audio, prosody, timbre, labels)
        return DataLoader(ds, batch_size=2)

    def test_train_epoch_returns_metrics(self, model, loader):
        """train_epoch runs and returns metrics dict with expected keys."""
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=False)
        metrics = trainer.train_epoch(loader)
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "train_f1" in metrics
        assert 0.0 <= metrics["train_accuracy"] <= 1.0
```

Run: `pytest tests/training/test_trainer.py::TestTrainerForward -v`
Expected: PASS

- [ ] **Step 4: Write test — `test_evaluate_returns_metrics`**

```python
    def test_evaluate_returns_metrics(self, model, loader):
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=False)
        metrics = trainer.evaluate(loader)
        assert "val_loss" in metrics
        assert "val_accuracy" in metrics
        assert "val_f1" in metrics

    def test_baseline_training_step_runs(self, loader):
        """Baseline model's training_step ignores prosody/timbre args."""
        from src.models import BaselineClassifier
        model = BaselineClassifier(device="cpu")
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=True)
        trainer.train_epoch(loader)
        # Should not crash despite collated batch having prosody/timbre fields
```

- [ ] **Step 5: Write test — `test_checkpoint_save_load`**

```python
class TestCheckpoint:
    def test_save_load_roundtrip(self, tmp_path):
        from src.models import BaselineClassifier
        model = BaselineClassifier(device="cpu")
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=True)

        # Advance epoch
        trainer.current_epoch = 5
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(str(path))

        # Load into fresh trainer
        model2 = BaselineClassifier(device="cpu")
        trainer2 = AmyTrainer(model2, device=torch.device("cpu"), is_baseline=True)
        trainer2.load_checkpoint(str(path))

        assert trainer2.current_epoch == 5
        assert torch.equal(
            trainer.model.classifier.weight, trainer2.model.classifier.weight
        )
```

- [ ] **Step 6: Write test — `test_lambda_logging`**

```python
class TestLambdaLogging:
    def test_amy_logs_lambdas(self):
        """Amy model's _get_lambdas returns lambda_p and lambda_t."""
        from src.models import AmyForProsodyClassification
        vectors = torch.randn(1024, 8)
        model = AmyForProsodyClassification(warm_start_vectors=vectors, device="cpu")
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=False)
        lambdas = trainer._get_lambdas()
        assert "lambda_p" in lambdas
        assert "lambda_t" in lambdas
        assert lambdas["lambda_p"] == 0.0  # zero-init
        assert lambdas["lambda_t"] == 0.0  # zero-init

    def test_baseline_returns_empty_lambdas(self):
        from src.models import BaselineClassifier
        model = BaselineClassifier(device="cpu")
        trainer = AmyTrainer(model, device=torch.device("cpu"), is_baseline=True)
        assert trainer._get_lambdas() == {}
```

- [ ] **Step 7: Run all training tests**

```bash
pytest tests/training/test_trainer.py -v
```

Expected: all PASS

- [ ] **Step 8: Commit**

```bash
git add src/training/ tests/training/
git commit -m "feat: add AmyTrainer — vanilla PyTorch training loop with gradient accumulation, metrics, W&B, checkpoints"
```

---

### Task 3: CLI Entry Point — `scripts/train_amy.py`

**Files:**
- Create: `scripts/train_amy.py`

**Purpose:** `argparse` CLI that loads data, creates model, configures trainer, runs training + evaluation, saves checkpoints. Supports `--mode baseline|amy` to switch between model architectures.

**CLI design:**
```
usage: train_amy.py [-h] --data-path PARQUET_PATH --checkpoint-dir DIR
                    [--mode {baseline,amy}] [--epochs N] [--batch-size N]
                    [--lr LR] [--weight-decay WD] [--grad-accum N]
                    [--seed N] [--wandb] [--wandb-project STR]
                    [--facodec-checkpoint PATH] [--output-dir DIR]
```

- [ ] **Step 1: Write the CLI script**

```python
# scripts/train_amy.py
"""Train MOSS-Audio baseline or Amy model on MUStARD for binary sarcasm classification."""

import argparse
import sys
import torch
from pathlib import Path
from torch.utils.data import DataLoader

# Add src to path when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from src.data.mustard_dataset import MustardDataset, collate_mustard, create_mustard_splits
from src.models import AmyForProsodyClassification, BaselineClassifier
from src.models.codebook_utils import load_prosody_codebook_vectors
from src.training.trainer import AmyTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train Amy LM on MUStARD")
    p.add_argument("--data-path", type=str, required=True,
                   help="Path to preprocessed train.parquet")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/training",
                   help="Directory for model checkpoints")
    p.add_argument("--mode", type=str, choices=["baseline", "amy"], default="amy",
                   help="Model mode: baseline (no FACodec) or amy")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true",
                   help="Enable W&B logging")
    p.add_argument("--wandb-project", type=str, default="amy-lm-pilot",
                   help="W&B project name")
    p.add_argument("--facodec-checkpoint", type=str,
                   default="checkpoints/facodec/ns3_facodec_decoder.bin",
                   help="Path to FACodec decoder checkpoint for warm-starting")
    p.add_argument("--output-dir", type=str, default="outputs/training",
                   help="Directory for training artifacts (metrics, config)")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: cuda or cpu")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Data
    train_ds, val_ds, test_ds = create_mustard_splits(
        args.data_path, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_mustard)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            collate_fn=collate_mustard)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             collate_fn=collate_mustard)
    print(f"Data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test")

    # Model
    if args.mode == "amy":
        vectors = load_prosody_codebook_vectors(args.facodec_checkpoint)
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors, device=device,
        )
        is_baseline = False
    else:
        model = BaselineClassifier(device=device)
        is_baseline = True
    model = model.to(device)
    print(f"Model: {'Baseline' if is_baseline else 'Amy'} ({args.mode})")

    # W&B
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args))

    # Trainer
    trainer = AmyTrainer(
        model=model,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_accum_steps=args.grad_accum,
        log_wandb=args.wandb,
        is_baseline=is_baseline,
    )

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_metrics = trainer.train_epoch(train_loader)
        print(f"Train | Loss: {train_metrics['train_loss']:.4f} | "
              f"Acc: {train_metrics['train_accuracy']:.3f} | "
              f"F1: {train_metrics['train_f1']:.3f}")

        val_metrics = trainer.evaluate(val_loader)
        print(f"Val   | Loss: {val_metrics['val_loss']:.4f} | "
              f"Acc: {val_metrics['val_accuracy']:.3f} | "
              f"F1: {val_metrics['val_f1']:.3f}")

        if not is_baseline:
            lambdas = trainer._get_lambdas()
            print(f"λ_p={lambdas['lambda_p']:.6f} λ_t={lambdas['lambda_t']:.6f}")

        # Save best
        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            trainer.save_checkpoint(str(ckpt_dir / "best_model.pt"))
            print(f"Saved best checkpoint (val_acc={best_val_acc:.3f})")

        trainer.save_checkpoint(str(ckpt_dir / f"epoch_{epoch}.pt"))

    # Final evaluation on test set
    print("\n--- Test Evaluation ---")
    trainer.load_checkpoint(str(ckpt_dir / "best_model.pt"))
    test_metrics = trainer.evaluate(test_loader)
    print(f"Test  | Loss: {test_metrics['val_loss']:.4f} | "
          f"Acc: {test_metrics['val_accuracy']:.3f} | "
          f"F1: {test_metrics['val_f1']:.3f}")

    # Save results
    import json
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "mode": args.mode,
        "test_metrics": test_metrics,
        "best_val_accuracy": best_val_acc,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_dir / 'results.json'}")

    if args.wandb:
        wandb.log({"test_" + k: v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_amy.py
git commit -m "feat: add scripts/train_amy.py CLI for training baseline and Amy models"
```

---

## Phase Completion Criteria

- [ ] `BaselineClassifier` passes all tests (forward shape, backbone frozen, classifier trainable, equivalence at λ=0)
- [ ] `AmyTrainer` passes all tests (init, train_epoch, evaluate, checkpoint, lambda logging)
- [ ] `scripts/train_amy.py` runs end-to-end on preprocessed MUStARD data
- [ ] W&B logs include loss, accuracy, F1 per epoch for both modes
- [ ] W&B logs include `lambda_p`, `lambda_t` per epoch for Amy mode
- [ ] Training produces `checkpoints/training/best_model.pt` and per-epoch checkpoints
- [ ] `outputs/training/results.json` contains test metrics
- [ ] All existing tests (147+) still pass
- [ ] Preprocessing pipeline complete (698 rows verified)

## Handoff Notes

### For the next implementer (subagent-driven-development)
1. Preprocessing must be complete before Task 2 tests run (tests use real DataLoader from preprocessed parquet). Wait for PID 401355 to finish. Verify with the command in [Data Status](#data-status).
2. `BaselineClassifier` and `AmyForProsodyClassification` must produce identical logits at λ=0 when given identical classifier weights. This is tested in `TestBaselineAmyEquivalence` and is a fundamental architecture invariant.
3. MOSS-Audio model loading is slow (~30s on cold cache). Tests will take time. Budget ~2-3 min per test class on CPU.
4. W&B is optional (`--wandb` flag). Tests mock it (no `wandb.init()` in tests).
5. Gradient accumulation is used because batch_size=1-2 with the 4B backbone on a single GPU leaves little headroom. Default `grad_accum_steps=4` gives effective batch size of 8 with batch_size=2.

### Known Risks
- **Out of memory (OOM)**: The 4B Qwen3 model forward+backward may exceed VRAM even with batch_size=1. If so, enable gradient checkpointing on the language model: `model.get_language_model().gradient_checkpointing_enable()`.
- **λ drift**: Zero-init guarantees baseline equivalence at step 0, but if λ doesn't drift meaningfully, the prosody hypothesis may be falsified. Expected λ > 0.01 by epoch 5 if signal is useful.
- **Overfitting**: 698 samples is small. Monitor train/val gap. If val F1 << train F1 by epoch 3, consider early stopping or stronger weight decay.
