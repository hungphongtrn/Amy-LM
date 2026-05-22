# Phase 3: Training Script

## Phase Goal

`python scripts/train_amy_dpo.py` loads the real NVTTS-FACodec dataset, filters by cosine threshold, initializes AmyLM with 4-bit QLoRA, and starts a DPO training run on a single 3060 GPU.

## Files to Touch

- `scripts/train_amy_dpo.py` — **Create**: Training entry point
- `src/training/__init__.py` — Update if needed

## Tasks

### Task 1: Create training script

**Files:**
- Create: `scripts/train_amy_dpo.py`

- [ ] **Step 1: Imports and argument parser**

```python
#!/usr/bin/env python3
"""AmyLM DPO training script.

Loads NVTTS-FACodec preference dataset, initializes 4-bit QLoRA AmyLM,
and trains via AmyDPOTrainer on a single GPU.

Usage:
    python scripts/train_amy_dpo.py \
        --model "OpenMOSS-Team/MOSS-Audio-4B-Thinking" \
        --dataset "hungphongtrn/nvtts_facodec" \
        --output-dir ./output/amy_dpo \
        --cosine-threshold 0.85
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import BitsAndBytesConfig, PreTrainedTokenizerBase
from trl import DPOConfig

# Add vendor path for MOSS-Audio processor
_VENDOR_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vendor", "MOSS-Audio", "src"))
if _VENDOR_SRC not in sys.path:
    sys.path.insert(0, _VENDOR_SRC)

from processing_moss_audio import MossAudioProcessor

# Add src path
_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from models.amy_lm import AmyLM, AmyLMConfig
from training.dpo_collator import DPOCollator
from training.amy_dpo_trainer import AmyDPOTrainer
```

- [ ] **Step 2: Config dataclass and argument parsing**

```python
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AmyLM DPO Training")
    # Model
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Audio-4B-Thinking",
                        help="Pretrained MOSS-Audio model ID or path")
    parser.add_argument("--from-checkpoint", type=str, default=None,
                        help="Resume from HF checkpoint directory")

    # Dataset
    parser.add_argument("--dataset", default="hungphongtrn/nvtts_facodec",
                        help="HuggingFace dataset ID")
    parser.add_argument("--cosine-threshold", type=float, default=0.85,
                        help="Filter samples with cosine_similarity < threshold")

    # QLoRA
    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")

    # Training hyperparameters
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="Learning rate for FACodec modules (LoRA uses paged_adamw_8bit default)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", help="LR scheduler type")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--num-epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--per-device-batch-size", type=int, default=1,
                        help="Per-device train batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--max-length", type=int, default=1024,
                        help="Maximum sequence length")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Max gradient norm")

    # Logging & checkpointing
    parser.add_argument("--output-dir", default="./output/amy_dpo", help="Output directory")
    parser.add_argument("--logging-steps", type=int, default=10, help="Log every N steps")
    parser.add_argument("--save-steps", type=int, default=500, help="Save checkpoint every N steps")
    parser.add_argument("--eval-steps", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--wandb-project", default="amy-lm-dpo", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name")

    # Misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B logging")

    return parser.parse_args()
```

- [ ] **Step 3: Dataset loading and filtering**

```python
def load_and_filter_dataset(
    dataset_id: str,
    cosine_threshold: float,
) -> tuple:
    """Load NVTTS-FACodec dataset and filter by cosine similarity.

    Returns:
        (train_dataset, eval_dataset) — HF Dataset objects.
    """
    ds = load_dataset(dataset_id, split="train")  # full dataset
    ds = ds.filter(lambda x: x["cosine_similarity"] < cosine_threshold)
    print(f"After cosine < {cosine_threshold}: {len(ds)} samples")

    # Split using existing NVTTS list indices (train: 0-3640, dev: 3641-3686, test: 3687-4045)
    TRAIN_COUNT = 3641
    DEV_COUNT = 46

    train_ds = ds.select(range(TRAIN_COUNT))
    dev_ds = ds.select(range(TRAIN_COUNT, TRAIN_COUNT + DEV_COUNT))

    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")
    return train_ds, dev_ds
```

- [ ] **Step 4: Model initialization with 4-bit QLoRA**

```python
def init_model(args: argparse.Namespace) -> AmyLM:
    """Initialize AmyLM with 4-bit quantization + QLoRA adapters.

    The AmyLM class inherits MossAudioModel (HF PreTrainedModel).
    We load the base MOSS-Audio checkpoint, then wrap in AmyLM config,
    apply 4-bit quantization, freeze backbone, add LoRA adapters.

    Returns:
        PEFT-wrapped AmyLM ready for training.
    """
    # Load a fresh AmyLM instance with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    amy_config = AmyLMConfig.from_pretrained(args.model, trust_remote_code=True)
    # Override freeze defaults: backbone frozen, FACodec trainable
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

    # Verify trainable params are FACodec only (pre-LoRA)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Pre-LoRA: {trainable_params:,}/{total_params:,} trainable params "
          f"({100 * trainable_params / total_params:.1f}%)")

    if trainable_params == 0:
        raise RuntimeError(
            "No trainable parameters after freeze! Check freeze flags in AmyLMConfig."
        )

    # Apply LoRA adapters
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
```

- [ ] **Step 5: Processor and collator setup**

```python
def init_processor_and_collator(
    model_id: str,
    max_length: int,
) -> tuple[MossAudioProcessor, DPOCollator]:
    """Create MOSS-Audio processor and custom DPO collator.

    The processor handles mel extraction and tokenizer access.
    The collator bridges raw dataset rows → DPO batch format.
    """
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
```

- [ ] **Step 6: Main training entry point**

```python
def main() -> None:
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Initialize W&B
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Load dataset
    train_ds, dev_ds = load_and_filter_dataset(args.dataset, args.cosine_threshold)

    # Initialize model
    model = init_model(args)

    # Setup processor and collator
    processor, collator = init_processor_and_collator(args.model, args.max_length)

    # Training arguments
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        beta=args.beta,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler,
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
        precompute_ref_log_probs=True,
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

    # Train
    print(f"Starting DPO training with {len(train_ds)} training samples, "
          f"{len(dev_ds)} dev samples")
    trainer.train()

    # Save final checkpoint
    final_path = os.path.join(args.output_dir, "final")
    trainer.save_model(final_path)
    print(f"Final model saved to {final_path}")


if __name__ == "__main__":
    main()
```

### Task 2: Verify script syntax and imports

- [ ] **Step 1: Verify all imports resolve (no GPU needed)**

```bash
uv run python -c "
import ast, sys
with open('scripts/train_amy_dpo.py') as f:
    ast.parse(f.read())
print('Syntax OK')
"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_amy_dpo.py
git commit -m "feat: add AmyLM DPO training script"
```

## Phase Completion Criteria
- [ ] Training script parses all arguments correctly
- [ ] Dataset loads and filters by cosine threshold
- [ ] Model initializes with 4-bit QLoRA and prints trainable param count
- [ ] Script is import-clean and syntax-valid
- [ ] Ready for a manual dry-run with `--no-wandb` and real GPU

## Handoff Notes

A full training run (3 epochs, B=1, 3641 samples, grad_accum=4) on a 3060 takes ~12-24 hours. First do a smoke test: 1 epoch on 10 samples with batch_size=1, max_length=512, verify loss decreases and lambda metrics change.
