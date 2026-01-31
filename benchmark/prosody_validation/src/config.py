"""
Configuration management for Prosody Validation Benchmark
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration dataclass for the benchmark pipeline."""

    # Directory paths (relative to benchmark/prosody_validation/)
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "data"
    )
    OUTPUTS_DIR: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "outputs"
    )
    AUDIO_DIR: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "outputs" / "audio"
    )
    RESPONSES_DIR: Path = field(
        default_factory=lambda: Path(__file__).parent.parent / "outputs" / "responses"
    )

    # Model names
    REWRITE_MODEL: str = "google/gemini-2.5-flash"
    TEXT_MODEL: str = "openai/gpt-4o-mini"
    E2E_MODEL: str = "google/gemini-2.5-flash"

    # Batch sizes
    BATCH_SIZE_TTS: int = 10
    BATCH_SIZE_WHISPER: int = 20

    # Concurrency settings
    MAX_CONCURRENT_LLM: int = 100
    MAX_CONCURRENT_TTS: int = 5
    MAX_CONCURRENT_ASR: int = 10

    # API settings
    OPENROUTER_API_KEY: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_API_KEY", "")
    )
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

    # Qwen3-TTS settings (to be configured on GPU machine)
    QWEN3_TTS_REPO_PATH: Optional[str] = None
    QWEN3_TTS_DEVICE: str = "cuda"

    # Whisper settings
    WHISPER_MODEL_SIZE: str = "base"
    WHISPER_DEVICE: str = "cpu"

    # Logging settings
    LOG_LEVEL: str = "INFO"

    def __post_init__(self):
        """Ensure directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self.AUDIO_DIR.mkdir(parents=True, exist_ok=True)
        self.RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "Config":
        """Load configuration from YAML file."""
        import yaml

        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Convert Path strings to Path objects
        if "BASE_DIR" in data:
            data["BASE_DIR"] = Path(data["BASE_DIR"])
        if "DATA_DIR" in data:
            data["DATA_DIR"] = Path(data["DATA_DIR"])
        if "OUTPUTS_DIR" in data:
            data["OUTPUTS_DIR"] = Path(data["OUTPUTS_DIR"])
        if "AUDIO_DIR" in data:
            data["AUDIO_DIR"] = Path(data["AUDIO_DIR"])
        if "RESPONSES_DIR" in data:
            data["RESPONSES_DIR"] = Path(data["RESPONSES_DIR"])

        return cls(**data)


# Global configuration instance
config = Config()


def get_config() -> Config:
    """Get the global configuration instance."""
    return config


def set_config(new_config: Config) -> None:
    """Set the global configuration instance."""
    global config
    config = new_config
