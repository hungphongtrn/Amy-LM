"""Configuration loader for AmyLM DPO training.

Provides a typed dataclass + YAML loader so configs can be saved, versioned,
shared, and overridden via CLI.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DPOTrainingConfig:
    """All tunable hyperparameters for AmyLM DPO training."""

    # ── Model ───────────────────────────────────────────────────────────
    model: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking"

    # ── Dataset ─────────────────────────────────────────────────────────
    dataset: str = "hungphongtrn/nvtts_facodec"
    cosine_threshold: float = 0.85

    # ── LoRA ────────────────────────────────────────────────────────────
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05

    # ── DPO ─────────────────────────────────────────────────────────────
    beta: float = 0.1
    precompute_ref_batch_size: int | None = None  # Larger batch for ref log-prob precomputation (None = use training batch)

    # ── Optimization ────────────────────────────────────────────────────
    learning_rate: float = 5.0e-5
    warmup_ratio: float = 0.1
    max_length: int = 1024
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    num_epochs: float = 3.0
    max_grad_norm: float = 1.0
    optim: str = "paged_adamw_8bit"

    # ── Logging & Checkpointing ─────────────────────────────────────────
    output_dir: str = "./output/amy_dpo"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    wandb_project: str = "amy-lm-dpo"
    wandb_run_name: str | None = None
    no_wandb: bool = False

    # ── Debug ────────────────────────────────────────────────────────────
    num_samples: int | None = None  # Limit dataset to first N samples for smoke tests
    gradient_checkpointing: bool = True

    # ── Misc ────────────────────────────────────────────────────────────
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> DPOTrainingConfig:
        """Load config from a YAML file.

        Only keys present in the YAML override defaults; missing keys keep
        their dataclass default values.
        """
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)
        if raw is None:
            raise ValueError(f"Config file is empty: {path}")

        # Filter to only known fields (ignore unknown keys gracefully)
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in raw.items() if k in known}
        return cls(**filtered)

    def merge_cli(self, args: argparse.Namespace) -> DPOTrainingConfig:
        """Override config values with any non-default CLI arguments.

        args should be the output of parse_args() from the training script.
        Expects argparse defaults to match this dataclass defaults, so we
        only override when the CLI value differs from its argparse default.
        """
        return self

    def as_dict(self) -> dict[str, Any]:
        """Return config as a plain dict (for DPOConfig construction)."""
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}


def load_config(
    config_path: str | Path | None = None,
    cli_overrides: argparse.Namespace | None = None,
) -> DPOTrainingConfig:
    """Load DPO config from YAML file, optionally overridden by CLI args.

    Priority (highest to lowest):
      1. CLI arguments (non-default values)
      2. YAML file values
      3. Dataclass defaults

    Args:
        config_path: Path to YAML config file (None → use dataclass defaults).
        cli_overrides: argparse.Namespace from the training script.

    Returns:
        DPOTrainingConfig with resolved values.
    """
    if config_path is not None:
        config = DPOTrainingConfig.from_yaml(config_path)
    else:
        config = DPOTrainingConfig()

    if cli_overrides is not None:
        config = _apply_cli_overrides(config, cli_overrides)

    return config


def _apply_cli_overrides(
    config: DPOTrainingConfig, cli: argparse.Namespace
) -> DPOTrainingConfig:
    """Override config fields where CLI args differ from argparse defaults.

    We rely on argparse's `get_default()` to detect explicit user overrides.
    """
    for field in config.__dataclass_fields__:
        if not hasattr(cli, field):
            continue
        cli_val = getattr(cli, field)
        # argparse stores its own default separately — compare
        default_val = cli.__dict__.get(f"__default_{field}", None)
        if default_val is None:
            default_val = _get_argparse_default(cli, field)
        if cli_val != default_val:
            setattr(config, field, cli_val)
    return config


def _get_argparse_default(namespace: argparse.Namespace, key: str) -> Any:
    """Retrieve the argparse-level default for a given argument name.

    argparse tracks defaults on the parser actions, not on the Namespace.
    This walks the parser actions attached to the namespace to find the match.
    """
    parser: argparse.ArgumentParser | None = getattr(
        namespace, "__parser__", None
    )
    if parser is None:
        return None
    for action in parser._actions:
        if action.dest == key:
            return action.default
    return None


# ── Helpers for training script ─────────────────────────────────────────────


def register_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the --config argument to an existing argument parser."""
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (overrides defaults; CLI flags take highest priority)",
    )


def resolve_config(
    parser: argparse.ArgumentParser, raw_args: list[str] | None = None
) -> DPOTrainingConfig:
    """Two-pass parse: first load YAML, then let CLI override.

    Returns a fully resolved DPOTrainingConfig.
    """
    # Pass 1: extract --config path without full parse
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_ns, _ = config_parser.parse_known_args(raw_args)

    # Load YAML defaults
    if config_ns.config:
        config = DPOTrainingConfig.from_yaml(config_ns.config)
    else:
        config = DPOTrainingConfig()

    # Set argparse defaults to YAML values so CLI overrides work naturally
    for field in config.__dataclass_fields__:
        for action in parser._actions:
            if action.dest == field:
                action.default = getattr(config, field)
                break

    # Pass 2: full parse (CLI overrides YAML by design)
    args = parser.parse_args(raw_args)
    # Rebuild config from final parsed values
    return DPOTrainingConfig(
        **{f: getattr(args, f) for f in config.__dataclass_fields__ if hasattr(args, f)}
    )
