"""Train MOSS-Audio baseline or Amy model on MUStARD for binary sarcasm classification.

Usage:
    # Amy model training with W&B
    python scripts/train_amy_classifier.py --data-path data/processed/mustard-processed/train.parquet --mode amy --wandb

    # Baseline training
    python scripts/train_amy_classifier.py --data-path data/processed/mustard-processed/train.parquet --mode baseline
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mustard_dataset import (
    ShuffledFacodecDataset,
    collate_mustard,
    create_mustard_splits,
)
from src.models import AmyForProsodyClassification, BaselineClassifier
from src.training.trainer import AmyTrainer


def parse_args():
    p = argparse.ArgumentParser(
        description="Train Amy LM on MUStARD for binary sarcasm classification"
    )
    p.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to preprocessed FACodec-encoded train.parquet",
    )
    p.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/training",
        help="Directory for model checkpoints",
    )
    p.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "amy"],
        default="amy",
        help="Model mode: baseline (MOSS-Audio + Linear) or amy (with FACodec streams)",
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=8,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
    )
    p.add_argument(
        "--grad-checkpoint",
        action="store_true",
        help="Enable gradient checkpointing on Qwen3 LM to reduce VRAM usage",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    p.add_argument(
        "--wandb-project",
        type=str,
        default="amy-lm-pilot",
        help="W&B project name",
    )
    p.add_argument(
        "--facodec-checkpoint",
        type=str,
        default="checkpoints/facodec/ns3_facodec_decoder.bin",
        help="Path to FACodec decoder checkpoint for warm-starting prosody embedding",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training",
        help="Directory for training artifacts (metrics JSON)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience (0 = disabled, run all epochs)",
    )
    p.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping (default: 1.0)",
    )
    p.add_argument(
        "--checkpoint-metric",
        type=str,
        choices=["val_accuracy", "val_f1"],
        default="val_accuracy",
        help="Metric for best checkpoint selection and early stopping (default: val_accuracy)",
    )
    p.add_argument(
        "--facodec-control",
        type=str,
        choices=["aligned", "shuffled"],
        default="aligned",
        help="FACodec feature alignment: 'aligned' (standard) or 'shuffled' (negative control). Only valid with --mode amy.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Validate facodec-control only valid with amy
    is_baseline = args.mode == "baseline"
    if is_baseline and args.facodec_control == "shuffled":
        raise ValueError("--facodec-control shuffled is only valid with --mode amy (baseline has no FACodec streams)")

    # Data
    train_ds, val_ds, test_ds = create_mustard_splits(args.data_path, seed=args.seed)

    if not is_baseline and args.facodec_control == "shuffled":
        train_ds = ShuffledFacodecDataset(train_ds, seed=args.seed)
        val_ds = ShuffledFacodecDataset(val_ds, seed=args.seed)
        test_ds = ShuffledFacodecDataset(test_ds, seed=args.seed)
        print("FACodec control: shuffled (derangement within each split)")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_mustard,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        collate_fn=collate_mustard,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        collate_fn=collate_mustard,
    )
    print(
        f"Data: {len(train_ds)} train, {len(val_ds)} val, {len(test_ds)} test"
        f" ({len(train_ds) + len(val_ds) + len(test_ds)} total)"
    )

    # Model
    if is_baseline:
        model = BaselineClassifier(
            device=device,
            gradient_checkpointing=args.grad_checkpoint,
        )
    else:
        model = AmyForProsodyClassification(
            prosody_warm_start_vectors_path=args.facodec_checkpoint,
            device=device,
            gradient_checkpointing=args.grad_checkpoint,
        )
    model = model.to(device)

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
        max_grad_norm=args.max_grad_norm,
    )

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    ckpt_metric = args.checkpoint_metric
    best_metric = -float("inf")
    epochs_no_improve = 0
    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epoch", unit="ep")
    for epoch in epoch_bar:
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.evaluate(val_loader)

        epoch_bar.set_postfix(
            train_loss=f"{train_metrics['train_loss']:.3f}",
            val_metric=f"{val_metrics[ckpt_metric]:.3f}",
        )

        tqdm.write(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Acc: {train_metrics['train_accuracy']:.3f} | "
            f"F1: {train_metrics['train_f1']:.3f} | "
            f"Grad: {train_metrics.get('train_grad_norm', float('nan')):.2f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Acc: {val_metrics['val_accuracy']:.3f} | "
            f"F1: {val_metrics['val_f1']:.3f}"
        )

        if not is_baseline:
            lambdas = trainer._get_lambdas()
            if "lambda_p_grad" in lambdas:
                tqdm.write(
                    f"  lambda_p={lambdas['lambda_p']:.6f}  grad_p={lambdas['lambda_p_grad']:.3e}  "
                    f"lambda_t={lambdas['lambda_t']:.6f}  grad_t={lambdas['lambda_t_grad']:.3e}"
                )
            else:
                tqdm.write(f"  lambda_p={lambdas['lambda_p']:.6f}  lambda_t={lambdas['lambda_t']:.6f}")

        if val_metrics[ckpt_metric] >= best_metric:
            best_metric = val_metrics[ckpt_metric]
            epochs_no_improve = 0
            trainer.save_checkpoint(str(ckpt_dir / "best_model.pt"))
            tqdm.write(f"  -> Saved best checkpoint ({ckpt_metric}={best_metric:.3f})")
        else:
            epochs_no_improve += 1

        trainer.save_checkpoint(str(ckpt_dir / f"epoch_{epoch}.pt"))

        if args.patience > 0 and epochs_no_improve >= args.patience:
            tqdm.write(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    # Final evaluation on test set
    tqdm.write("\n--- Test Evaluation (best checkpoint) ---")
    trainer.load_checkpoint(str(ckpt_dir / "best_model.pt"))
    test_metrics = trainer.evaluate(test_loader)
    tqdm.write(
        f"Test  | Loss: {test_metrics['val_loss']:.4f} | "
        f"Acc: {test_metrics['val_accuracy']:.3f} | "
        f"F1: {test_metrics['val_f1']:.3f}"
    )

    # Save results
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "mode": args.mode,
        "seed": args.seed,
        "epochs_requested": args.epochs,
        "epochs_completed": epoch,
        "early_stopped": args.patience > 0 and epochs_no_improve >= args.patience,
        "patience": args.patience,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "test_metrics": test_metrics,
        "checkpoint_metric": ckpt_metric,
        "best_checkpoint_value": best_metric,
        "facodec_control": args.facodec_control,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")

    if args.wandb:
        wandb.log({"test_" + k: v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
