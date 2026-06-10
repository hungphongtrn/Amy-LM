"""Vanilla PyTorch training loop for Amy LM binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn
from tqdm import tqdm


class AmyTrainer:
    """Vanilla PyTorch training loop with metrics and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        grad_accum_steps: int = 4,
        log_wandb: bool = False,
        is_baseline: bool = False,
        is_lora: bool = False,
        max_grad_norm: float = 1.0,
    ) -> None:
        if grad_accum_steps < 1:
            raise ValueError(f"grad_accum_steps must be >= 1, got {grad_accum_steps}")

        self.model = model
        self.device = device
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params, lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.grad_accum_steps = grad_accum_steps
        self.log_wandb = log_wandb
        self.is_baseline = is_baseline
        self.is_lora = is_lora
        self.max_grad_norm = max_grad_norm
        self.current_epoch = 0
        self._saved_lambda_grads: dict[str, float] = {}
        if not is_baseline:
            self._register_lambda_grad_hooks()

    @property
    def _base_model(self) -> nn.Module:
        return self.model.model if self.is_lora else self.model

    def training_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward pass and compute classification loss.

        Accepts batch of exactly 4 or 6 elements.
        Collated batch: (audio, prosody, timbre, labels, audio_lengths, prosody_lengths)
        Simple batch:   (audio, prosody, timbre, labels)
        """
        n = len(batch)
        if n != 4 and n != 6:
            raise ValueError(
                f"Expected batch of 4 or 6 elements, got {n}. "
                f"Use collate_mustard for collation or TensorDataset with 4 elements."
            )
        audio, prosody, timbre, labels = batch[:4]
        audio = audio.to(self.device)
        labels = labels.to(self.device)

        use_autocast = self.device.type == "cuda"
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_autocast):
            if self.is_baseline:
                logits = self.model(audio)
            else:
                prosody = prosody.to(self.device)
                timbre = timbre.to(self.device)
                logits = self.model(audio=audio, prosody_indices=prosody, timbre_vector=timbre)

        loss = self.criterion(logits, labels)
        return loss, logits.detach().cpu(), labels.detach().cpu()

    def train_epoch(self, dataloader) -> dict[str, float]:
        """Train for one epoch and return train metrics."""
        n_batches = len(dataloader)
        if n_batches == 0:
            raise ValueError("Dataloader is empty — cannot train an epoch.")

        self.model.train()
        total_loss = 0.0
        total_grad_norm = 0.0
        grad_steps = 0
        all_preds = []
        all_labels = []

        self.optimizer.zero_grad()

        for i, batch in enumerate(tqdm(dataloader, desc="  Train", leave=False, unit="batch")):
            loss, logits, labels = self.training_step(batch)
            scaled_loss = loss / self.grad_accum_steps
            scaled_loss.backward()

            if (i + 1) % self.grad_accum_steps == 0 or (i + 1) == n_batches:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()
                self.optimizer.zero_grad()
                total_grad_norm += grad_norm.item()
                grad_steps += 1

            total_loss += loss.item()
            all_preds.append(logits)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels, prefix="train")
        metrics["train_loss"] = total_loss / n_batches
        if grad_steps > 0:
            metrics["train_grad_norm"] = total_grad_norm / grad_steps

        if self.log_wandb:
            import wandb

            wandb.log({**metrics, **self._get_lambdas(), "epoch": self.current_epoch})

        self.current_epoch += 1
        return metrics

    @torch.no_grad()
    def evaluate(self, dataloader) -> dict[str, float]:
        """Evaluate and return validation metrics."""
        n_batches = len(dataloader)
        if n_batches == 0:
            raise ValueError("Dataloader is empty — cannot evaluate.")

        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="  Eval ", leave=False, unit="batch"):
            loss, logits, labels = self.training_step(batch)
            total_loss += loss.item()
            all_preds.append(logits)
            all_labels.append(labels)

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        metrics = self._compute_metrics(all_preds, all_labels, prefix="val")
        metrics["val_loss"] = total_loss / n_batches

        if self.log_wandb:
            import wandb
            wandb.log({**metrics, **self._get_lambdas(), "epoch": self.current_epoch})

        return metrics

    def _compute_metrics(
        self, logits: torch.Tensor, labels: torch.Tensor, prefix: str = "train"
    ) -> dict[str, float]:
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        f1 = self._compute_f1(preds, labels)
        return {f"{prefix}_accuracy": acc, f"{prefix}_f1": f1}

    @staticmethod
    def _compute_f1(preds: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute binary F1 without external dependencies."""
        tp = ((preds == 1) & (labels == 1)).float().sum().item()
        fp = ((preds == 1) & (labels == 0)).float().sum().item()
        fn = ((preds == 0) & (labels == 1)).float().sum().item()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        return 2 * precision * recall / (precision + recall + 1e-8)

    def _register_lambda_grad_hooks(self) -> None:
        """Register backward hooks on lambda_p/lambda_t to capture gradients."""
        fusion = self._base_model.amy_moss.residual_fusion

        def _make_hook(name: str):
            def hook(grad: torch.Tensor) -> None:
                self._saved_lambda_grads[name] = grad.detach().cpu().item()
            return hook

        for name in ("lambda_p", "lambda_t"):
            param = getattr(fusion, name)
            param.register_hook(_make_hook(name))

    def _get_lambdas(self) -> dict[str, float]:
        """Return lambda values and gradients for Amy model; empty for baseline."""
        if self.is_baseline:
            return {}
        fusion = self._base_model.amy_moss.residual_fusion
        result = {
            "lambda_p": fusion.lambda_p.item(),
            "lambda_t": fusion.lambda_t.item(),
        }
        if self._saved_lambda_grads:
            result["lambda_p_grad"] = self._saved_lambda_grads.get("lambda_p", 0.0)
            result["lambda_t_grad"] = self._saved_lambda_grads.get("lambda_t", 0.0)
        return result

    def save_checkpoint(self, path: str) -> None:
        """Save model, optimizer, and epoch state."""
        if self.is_lora:
            model_state = self.model.state_dict()
        else:
            model_state = {
                k: v
                for k, v in self.model.state_dict().items()
                if not k.startswith("amy_moss.moss.")
            }
        facodec_state = self._base_model.amy_moss.facodec_state_dict() if not self.is_baseline else {}
        torch.save(
            {
                "model_state_dict": model_state,
                "facodec_state_dict": facodec_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "is_baseline": self.is_baseline,
                "is_lora": self.is_lora,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model, optimizer, and epoch state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
