#!/usr/bin/env python3
"""Standalone gradient inspection for AmyMossLM DPO model.

Constructs the same tiny model as pytest test_dpo_forward_backward_gradient_flow,
runs a forward+backward pass, and prints gradient norms, parameter counts, dtypes,
and trainable status for every named parameter group in the model.

Usage:
    python scripts/test_gradients.py          # auto-detect GPU
    python scripts/test_gradients.py --cpu    # force CPU (slower)
"""

from __future__ import annotations

import argparse
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
from peft import LoraConfig, TaskType, get_peft_model

from scripts import train_amy_dpo
from src.models.moss_audio_model import MossAudioConfig, MossAudioModel
from src.models.amy_lm import AmyMossLM, AmyMossLMConfig
from src.training.config import DPOTrainingConfig


def _tiny_moss() -> MossAudioModel:
    config = MossAudioConfig(
        language_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
        },
        audio_config={
            "d_model": 32,
            "output_dim": 32,
            "num_mel_bins": 128,
            "encoder_layers": 1,
            "encoder_attention_heads": 2,
            "encoder_ffn_dim": 64,
            "downsample_hidden_size": 8,
            "deepstack_encoder_layer_indexes": [],
        },
        adapter_hidden_size=32,
    )
    return MossAudioModel(config)


def _unwrap_amy(base) -> torch.nn.Module:
    """Return the AmyMossLM module inside PEFT without chasing base_model forever.

    PEFT/HF wrappers expose nested ``base_model`` properties, and MossAudioModel's
    PreTrainedModel base property can point back to itself. Searching registered
    modules is bounded by PyTorch's memoized traversal and finds the module that
    actually owns the FACodec enrichment layers.
    """
    for module in base.modules():
        if all(
            hasattr(module, attr)
            for attr in (
                "moss",
                "prosody_embedding",
                "timbre_projection",
                "residual_fusion",
            )
        ):
            return module
    raise RuntimeError("Could not find AmyMossLM inside PEFT-wrapped model")


def build_model(device: torch.device) -> torch.nn.Module:
    moss = _tiny_moss().to(device)
    amy_config = AmyMossLMConfig(
        moss_config=moss.config,
        hidden_dim=moss.config.language_config.hidden_size,
        freeze_audio_encoder=True,
        freeze_audio_adapter=True,
        freeze_llm=True,
    )
    model = AmyMossLM(amy_config, moss=moss)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Pre-LoRA trainable: {trainable:,}/{total:,} ({100 * trainable / total:.1f}%)")

    lora_config = LoraConfig(
        r=2,
        lora_alpha=4,
        lora_dropout=0.0,
        target_modules=r"^moss\.language_model\..*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj|gate_proj)$",
        modules_to_save=[
            "prosody_embedding",
            "timbre_projection",
            "temporal_pool",
            "residual_fusion",
        ],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model


def format_num(n: float | int) -> str:
    if isinstance(n, (int, bool)):
        return f"{n:>12,}"
    return f"{n:>12.6f}"


def print_header() -> None:
    print()
    hdr = (
        f"{'PARAMETER NAME':<85s} "
        f"{'DTYPE':>8s} "
        f"{'TRAINABLE':>9s} "
        f"{'#ELEMS':>12s} "
        f"{'GRAD NORM':>12s} "
        f"{'GRAD MAX':>12s} "
        f"{'GRAD MIN':>12s} "
        f"{'GRAD MEAN':>12s}"
    )
    print(hdr)
    print("-" * len(hdr))


def log_gradient(row_name: str, param: torch.nn.Parameter | None) -> None:
    dtype_str = str(param.dtype).removeprefix("torch.") if param is not None else "  NONE  "
    trainable = param.requires_grad if param is not None else False
    num_elems = param.numel() if param is not None else 0

    if param is not None and param.grad is not None:
        g = param.grad
        g_norm = g.norm().item()
        g_max = g.max().item()
        g_min = g.min().item()
        g_mean = g.mean().item()
    elif param is not None and param.requires_grad:
        g_norm, g_max, g_min, g_mean = 0.0, 0.0, 0.0, 0.0
        dtype_str = dtype_str + " *ZERO*"
    else:
        g_norm = g_max = g_min = g_mean = float("nan")

    vals = (
        row_name[:84].ljust(85),
        dtype_str[:7].rjust(8),
        str(trainable).rjust(5),
        format_num(num_elems) if num_elems > 0 else " ".rjust(12),
        format_num(g_norm) if not (isinstance(g_norm, float) and g_norm != g_norm) else " ".rjust(12),
        format_num(g_max) if not (isinstance(g_max, float) and g_max != g_max) else " ".rjust(12),
        format_num(g_min) if not (isinstance(g_min, float) and g_min != g_min) else " ".rjust(12),
        format_num(g_mean) if not (isinstance(g_mean, float) and g_mean != g_mean) else " ".rjust(12),
    )
    print(f"{vals[0]} {vals[1]} {vals[2]} {vals[3]} {vals[4]} {vals[5]} {vals[6]} {vals[7]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradient inspection for AmyMossLM DPO model")
    parser.add_argument("--cpu", action="store_true", help="Force CPU (default: GPU if available)")
    args = parser.parse_args()

    device = torch.device("cpu") if args.cpu else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(0)
    model = build_model(device)
    model.train()

    # Un-prompt lambda gates for gradient flow
    amy = _unwrap_amy(model)
    for gate_name in ("lambda_p", "lambda_t", "lambda_c", "lambda_a"):
        gate = getattr(amy.residual_fusion, gate_name)
        print(f"residual_fusion.{gate_name} init value: {gate.item():.6f}")
        gate.data.fill_(1.0)

    B, S = 1, 8
    mel_len = 20

    input_ids = torch.randint(0, 63, (B, S), device=device)
    attention_mask = torch.ones(B, S, dtype=torch.long, device=device)
    labels = input_ids.clone()

    audio_data = torch.randn(B, 128, mel_len, device=device)
    audio_data_seqlens = torch.full((B,), mel_len, dtype=torch.long, device=device)

    audio_input_mask = torch.zeros(B, S, dtype=torch.bool, device=device)
    audio_input_mask[0, 2:5] = True

    prosody_indices = torch.randint(0, 1024, (B, 1, 20), device=device)
    timbre_vector = torch.randn(B, 256, device=device)

    print("\nRunning forward pass ...")
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        audio_data=audio_data,
        audio_data_seqlens=audio_data_seqlens,
        audio_input_mask=audio_input_mask,
        labels=labels,
        prosody_indices=prosody_indices,
        timbre_vector=timbre_vector,
    )
    print(f"Loss: {output.loss.item():.6f}")

    print("Running backward pass ...")
    output.loss.backward()

    # ── Collect and log ──────────────────────────────────────────────────
    print_header()

    # Group and log params
    sections = {
        "── LoRA adapters (Qwen3) ──": lambda n: "lora_" in n,
        "── FACodec: prosody_embedding ──": lambda n: "prosody_embedding" in n,
        "── FACodec: timbre_projection ──": lambda n: "timbre_projection" in n,
        "── FACodec: residual_fusion gates ──": lambda n: any(
            g in n for g in ("lambda_p", "lambda_t", "lambda_c", "lambda_a")
        ),
        "── FACodec: residual_fusion.norm ──": lambda n: "residual_fusion.norm" in n,
        "── lm_head ──": lambda n: "lm_head" in n and "lora_" not in n,
        "── moss.audio_encoder ──": lambda n: "moss.audio_encoder" in n and "lora_" not in n,
        "── moss.audio_adapter ──": lambda n: "moss.audio_adapter" in n and "lora_" not in n,
        "── moss.language_model (base, non-LoRA) ──": lambda n: (
            "moss.language_model" in n and "lora_" not in n
        ),
        "── deepstack_audio_merger ──": lambda n: "deepstack_audio_merger" in n,
    }

    seen: set[str] = set()
    for section_label, match_fn in sections.items():
        params = [(n, p) for n, p in model.named_parameters() if match_fn(n) and n not in seen]
        if not params:
            continue
        print(f"\n{section_label}")
        for name, param in params:
            log_gradient(name, param)
            seen.add(name)

    # ── Catch any not covered ────────────────────────────────────────────
    remaining = [(n, p) for n, p in model.named_parameters() if n not in seen]
    if remaining:
        print("\n── OTHER (unclassified) ──")
        for name, param in remaining:
            log_gradient(name, param)

    # ── Summary stats ────────────────────────────────────────────────────
    print("\n── SUMMARY ──")
    trainable_w_grad = sum(
        1 for _, p in model.named_parameters() if p.requires_grad and p.grad is not None
    )
    trainable_no_grad = sum(
        1 for _, p in model.named_parameters() if p.requires_grad and p.grad is None
    )
    frozen_w_grad = sum(
        1 for _, p in model.named_parameters() if not p.requires_grad and p.grad is not None
    )
    frozen_no_grad = sum(
        1 for _, p in model.named_parameters() if not p.requires_grad and p.grad is None
    )
    print(f"  Trainable + has grad : {trainable_w_grad}")
    print(f"  Trainable + NO grad  : {trainable_no_grad}  <-- PROBLEM if > 0")
    print(f"  Frozen    + has grad : {frozen_w_grad}       <-- PROBLEM if > 0")
    print(f"  Frozen    + no grad  : {frozen_no_grad}")


if __name__ == "__main__":
    main()
