#!/usr/bin/env python3
"""AmyLM DPO training script.

Usage:
    # Use a YAML config file (recommended):
    python scripts/train_amy_dpo.py --config configs/dpo/rtx3060_12gb.yaml

    # Override specific values from CLI (highest priority):
    python scripts/train_amy_dpo.py --config configs/dpo/rtx3060_12gb.yaml --beta 0.2 --lora-r 8

    # No config file (uses dataclass defaults):
    python scripts/train_amy_dpo.py --learning-rate 1e-5 --num-epochs 5

    # Debug smoke test:
    python scripts/train_amy_dpo.py --config configs/dpo/debug.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
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

from modeling_moss_audio import MossAudioModel
from processing_moss_audio import MossAudioProcessor
from models.amy_lm import AmyMossLM, AmyMossLMConfig
from training.amy_dpo_trainer import AmyDPOTrainer
from training.config import DPOTrainingConfig, register_config_arg, resolve_config
from training.dpo_collator import DPOCollator


def parse_config(raw_args: list[str] | None = None) -> DPOTrainingConfig:
    """Build argparse, inject YAML defaults, parse CLI → resolved config.

    Priority: CLI flags > YAML file > dataclass defaults.
    """
    parser = argparse.ArgumentParser(description="AmyLM DPO Training")

    # --config flag (handled first to load YAML before full parse)
    register_config_arg(parser)

    # Model
    parser.add_argument(
        "--model",
        default=DPOTrainingConfig.model,
        help="Pretrained MOSS-Audio model ID or path",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        default=DPOTrainingConfig.dataset,
        help="HuggingFace dataset ID",
    )
    parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=DPOTrainingConfig.cosine_threshold,
        help="Filter samples with cosine_similarity < threshold",
    )

    # QLoRA
    parser.add_argument("--lora-r", type=int, default=DPOTrainingConfig.lora_r, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=DPOTrainingConfig.lora_alpha, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=DPOTrainingConfig.lora_dropout, help="LoRA dropout")

    # Training
    parser.add_argument("--beta", type=float, default=DPOTrainingConfig.beta, help="DPO beta")
    parser.add_argument("--learning-rate", type=float, default=DPOTrainingConfig.learning_rate, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=DPOTrainingConfig.warmup_ratio, help="Warmup ratio")
    parser.add_argument("--num-epochs", type=float, default=DPOTrainingConfig.num_epochs, help="Training epochs")
    parser.add_argument("--per-device-batch-size", type=int, default=DPOTrainingConfig.per_device_batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=DPOTrainingConfig.gradient_accumulation_steps)
    parser.add_argument(
        "--max-length", type=int, default=DPOTrainingConfig.max_length, help="Max sequence length"
    )
    parser.add_argument("--max-grad-norm", type=float, default=DPOTrainingConfig.max_grad_norm)

    # Logging & checkpointing
    parser.add_argument("--output-dir", default=DPOTrainingConfig.output_dir, help="Output directory")
    parser.add_argument("--logging-steps", type=int, default=DPOTrainingConfig.logging_steps)
    parser.add_argument("--save-steps", type=int, default=DPOTrainingConfig.save_steps)
    parser.add_argument("--eval-steps", type=int, default=DPOTrainingConfig.eval_steps)
    parser.add_argument("--wandb-project", default=DPOTrainingConfig.wandb_project, help="W&B project")
    parser.add_argument("--wandb-run-name", default=DPOTrainingConfig.wandb_run_name, help="W&B run name")
    parser.add_argument("--no-wandb", action="store_true", default=DPOTrainingConfig.no_wandb, help="Disable W&B")

    # Misc
    parser.add_argument("--seed", type=int, default=DPOTrainingConfig.seed)
    parser.add_argument("--no-grad-checkpoint", action="store_false",
                        dest="gradient_checkpointing",
                        default=DPOTrainingConfig.gradient_checkpointing,
                        help="Disable gradient checkpointing (for debug)")
    parser.add_argument("--num-samples", type=int, default=DPOTrainingConfig.num_samples,
                        help="Limit dataset to first N samples (for debug)")

    return resolve_config(parser, raw_args)


def load_and_filter_dataset(dataset_id: str, cosine_threshold: float, num_samples: int | None = None):
    """Load NVTTS-FACodec dataset and filter by cosine similarity."""
    ds = load_dataset(dataset_id, split="train")
    ds = ds.filter(lambda x: x["cosine_similarity"] < cosine_threshold)
    total = len(ds)
    print(f"After cosine < {cosine_threshold}: {total} samples")

    if num_samples is not None:
        total = min(num_samples, total)

    # NVTTS split indices: train=0-3640, dev=3641-3686
    train_ds = ds.select(range(min(3641, total)))
    dev_start = min(3641, total)
    dev_end = min(dev_start + 46, total)
    dev_ds = ds.select(range(dev_start, dev_end)) if dev_end > dev_start else None

    # Add a 'prompt' column so TRL's _prepare_dataset skips extract_prompt.
    # Without this, extract_prompt finds the longest common prefix between
    # 'chosen' and 'rejected' text strings and truncates both, corrupting the
    # responses seen by DPOCollator. The actual prompt (system text + audio
    # placeholders) is constructed dynamically in DPOCollator._tokenize_sample.
    def _add_prompt(example):
        example["prompt"] = ""
        return example

    train_ds = train_ds.map(_add_prompt)
    if dev_ds is not None:
        dev_ds = dev_ds.map(_add_prompt)

    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds) if dev_ds else 0}")
    return train_ds, dev_ds


def init_model(config: DPOTrainingConfig):
    """Initialize AmyMossLM with 4-bit quantized backbone via constructor injection.

    Loads MossAudioModel (backbone only) with load_in_4bit=True, then
    wraps in AmyMossLM composition via AmyMossLM(config, moss=moss_4bit).
    No __class__ mutation, no _upgrade_from_moss hack.

    LoRA is scoped to MossAudioModel backbone only via target_modules regex
    (^moss\..*), leaving FACodec modules fully trainable at full precision.
    """
    moss = MossAudioModel.from_pretrained(
        config.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        load_in_4bit=True,
    )
    amy_config = AmyMossLMConfig(
        moss_config=moss.config,
        freeze_audio_encoder=True,
        freeze_audio_adapter=True,
        freeze_llm=True,
    )
    model = AmyMossLM(amy_config, moss=moss)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Pre-LoRA trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=r"^moss\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$",
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


def build_dpo_config(config: DPOTrainingConfig) -> DPOConfig:
    """Build TRL DPOConfig from resolved DPOTrainingConfig."""
    gc_kwargs = {"use_reentrant": False} if config.gradient_checkpointing else None
    return DPOConfig(
        output_dir=config.output_dir,
        precompute_ref_log_probs=True,
        beta=config.beta,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        per_device_train_batch_size=config.per_device_batch_size,
        per_device_eval_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_length,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_epochs,
        bf16=True,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=gc_kwargs if gc_kwargs else {},
        loss_type=["sigmoid"],
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        save_total_limit=config.save_total_limit,
        remove_unused_columns=False,
        report_to="wandb" if not config.no_wandb else "none",
        run_name=config.wandb_run_name,
        seed=config.seed,
    )


def print_config(config: DPOTrainingConfig) -> None:
    """Print resolved config as a table."""
    print("\n─ Resolved Config ─")
    for field in config.__dataclass_fields__:
        print(f"  {field:30s} = {getattr(config, field)}")
    print()


def main(raw_args: list[str] | None = None):
    config = parse_config(raw_args)
    print_config(config)

    if config.seed is not None:
        torch.manual_seed(config.seed)

    # Load dataset
    train_ds, dev_ds = load_and_filter_dataset(config.dataset, config.cosine_threshold, config.num_samples)

    # Init model
    model = init_model(config)

    # Setup processor and collator
    processor, collator = init_processor_and_collator(config.model, config.max_length)

    # DPO config
    dpo_config = build_dpo_config(config)

    trainer = AmyDPOTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        processing_class=processor._base_tokenizer,
        data_collator=collator,
        args=dpo_config,
    )

    print(f"Starting DPO training: {len(train_ds)} train, "
          f"{len(dev_ds) if dev_ds else 0} dev")
    trainer.train()

    final_path = os.path.join(config.output_dir, "final")
    trainer.save_model(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
