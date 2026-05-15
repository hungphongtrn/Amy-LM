"""Vanilla PyTorch training loop for Amy LM binary classification."""

from __future__ import annotations

import torch
import torch.nn as nn


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
    ) -> None:
        self.model = model
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps
        self.log_wandb = log_wandb
        self.is_baseline = is_baseline
        self.current_epoch = 0

    def training_step(
        self, batch: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one forward pass and compute classification loss."""
        audio, prosody, timbre, labels = batch[:4]
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

    def train_epoch(self, dataloader) -> dict[str, float]:
        """Train for one epoch and return train metrics."""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

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
    def evaluate(self, dataloader) -> dict[str, float]:
        """Evaluate and return validation metrics."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

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

    def _get_lambdas(self) -> dict[str, float]:
        """Return lambda values for Amy model; empty for baseline."""
        if self.is_baseline:
            return {}
        return {
            "lambda_p": self.model.fusion.lambda_p.item(),
            "lambda_t": self.model.fusion.lambda_t.item(),
        }

    def save_checkpoint(self, path: str) -> None:
        """Save model, optimizer, and epoch state."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epoch": self.current_epoch,
                "is_baseline": self.is_baseline,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model, optimizer, and epoch state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.current_epoch = ckpt["epoch"]
