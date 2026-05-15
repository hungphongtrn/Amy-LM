"""Train MOSS-Audio baseline or Amy model on MUStARD for binary sarcasm classification.

Usage:
    # Amy model training with W&B
    python scripts/train_amy.py --data-path data/processed/mustard-processed/train.parquet --mode amy --wandb

    # Baseline training
    python scripts/train_amy.py --data-path data/processed/mustard-processed/train.parquet --mode baseline
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.mustard_dataset import MustardDataset, collate_mustard, create_mustard_splits
from src.models import AmyForProsodyClassification, BaselineClassifier
from src.models.codebook_utils import load_prosody_codebook_vectors
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
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (effective batch = batch_size * grad_accum)",
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
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode}")

    # Data
    train_ds, val_ds, test_ds = create_mustard_splits(args.data_path, seed=args.seed)
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
    is_baseline = args.mode == "baseline"
    if is_baseline:
        model = BaselineClassifier(device=device)
    else:
        vectors = load_prosody_codebook_vectors(args.facodec_checkpoint)
        model = AmyForProsodyClassification(
            warm_start_vectors=vectors,
            device=device,
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
    )

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_metrics = trainer.train_epoch(train_loader)
        print(
            f"Train | Loss: {train_metrics['train_loss']:.4f} | "
            f"Acc: {train_metrics['train_accuracy']:.3f} | "
            f"F1: {train_metrics['train_f1']:.3f}"
        )

        val_metrics = trainer.evaluate(val_loader)
        print(
            f"Val   | Loss: {val_metrics['val_loss']:.4f} | "
            f"Acc: {val_metrics['val_accuracy']:.3f} | "
            f"F1: {val_metrics['val_f1']:.3f}"
        )

        if not is_baseline:
            lambdas = trainer._get_lambdas()
            print(f"lambda_p={lambdas['lambda_p']:.6f}  lambda_t={lambdas['lambda_t']:.6f}")

        if val_metrics["val_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["val_accuracy"]
            trainer.save_checkpoint(str(ckpt_dir / "best_model.pt"))
            print(f"  -> Saved best checkpoint (val_acc={best_val_acc:.3f})")

        trainer.save_checkpoint(str(ckpt_dir / f"epoch_{epoch}.pt"))

    # Final evaluation on test set
    print("\n--- Test Evaluation (best checkpoint) ---")
    trainer.load_checkpoint(str(ckpt_dir / "best_model.pt"))
    test_metrics = trainer.evaluate(test_loader)
    print(
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
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum,
        "effective_batch_size": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "test_metrics": test_metrics,
        "best_val_accuracy": best_val_acc,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_dir / 'results.json'}")

    if args.wandb:
        wandb.log({"test_" + k: v for k, v in test_metrics.items()})
        wandb.finish()


if __name__ == "__main__":
    main()
