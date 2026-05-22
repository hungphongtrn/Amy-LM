#!/usr/bin/env python3
"""AmyLM DPO training script."""

from __future__ import annotations

import argparse
import os
import sys

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig
from trl import DPOConfig

# Add src path
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Add vendor path for MOSS-Audio processor
_VENDOR_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "vendor", "MOSS-Audio", "src")
)
if _VENDOR_SRC not in sys.path:
    sys.path.insert(0, _VENDOR_SRC)

from processing_moss_audio import MossAudioProcessor
from models.amy_lm import AmyLM, AmyLMConfig
from training.dpo_collator import DPOCollator
from training.amy_dpo_trainer import AmyDPOTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="AmyLM DPO Training")

    # Model
    parser.add_argument(
        "--model",
        default="OpenMOSS-Team/MOSS-Audio-4B-Thinking",
        help="Pretrained MOSS-Audio model ID or path",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default="hungphongtrn/nvtts_facodec",
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.85,
        help="Filter samples with cosine_similarity < threshold",
    )

    # QLoRA
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Training
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--num-epochs", type=float, default=3.0, help="Training epochs")
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Max sequence length"
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Logging & checkpointing
    parser.add_argument("--output-dir", default="./output/amy_dpo", help="Output directory")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--wandb-project", default="amy-lm-dpo", help="W&B project")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B")

    # Misc
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_and_filter_dataset(dataset_id: str, cosine_threshold: float):
    """Load NVTTS-FACodec dataset and filter by cosine similarity."""
    ds = load_dataset(dataset_id, split="train")
    ds = ds.filter(lambda x: x["cosine_similarity"] < cosine_threshold)
    total = len(ds)
    print(f"After cosine < {cosine_threshold}: {total} samples")

    # NVTTS split indices: train=0-3640, dev=3641-3686
    train_ds = ds.select(range(min(3641, total)))
    dev_start = min(3641, total)
    dev_end = min(dev_start + 46, total)
    dev_ds = ds.select(range(dev_start, dev_end)) if dev_end > dev_start else None

    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds) if dev_ds else 0}")
    return train_ds, dev_ds


def init_model(args):
    """Initialize AmyLM with 4-bit quantization + QLoRA."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    amy_config = AmyLMConfig.from_pretrained(args.model, trust_remote_code=True)
    amy_config.freeze_audio_encoder = True
    amy_config.freeze_audio_adapter = True
    amy_config.freeze_llm = True

    model = AmyLM.from_pretrained(
        args.model,
        config=amy_config,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Pre-LoRA trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def init_processor_and_collator(model_id: str, max_length: int):
    """Create MOSS-Audio processor and DPO collator."""
    processor = MossAudioProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        enable_time_marker=True,
    )
    tokenizer = processor._base_tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    collator = DPOCollator(
        processor=processor,
        pad_token_id=int(tokenizer.pad_token_id),
        max_length=max_length,
    )
    return processor, collator


def main():
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Load dataset
    train_ds, dev_ds = load_and_filter_dataset(args.dataset, args.cosine_threshold)

    # Init model
    model = init_model(args)

    # Setup processor and collator
    processor, collator = init_processor_and_collator(args.model, args.max_length)

    # DPO config
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        precompute_ref_log_probs=True,
        beta=args.beta,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_epochs,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        loss_type=["sigmoid"],
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.wandb_run_name,
        seed=args.seed,
    )

    trainer = AmyDPOTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor._base_tokenizer,
        data_collator=collator,
        args=dpo_config,
    )

    print(f"Starting DPO training: {len(train_ds)} train, " f"{len(dev_ds) if dev_ds else 0} dev")
    trainer.train()

    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
