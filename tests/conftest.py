"""Shared test fixtures and utilities.

Provides auto-detected GPU device fixtures. All model-loading tests must
use these fixtures to ensure models load on GPU when available.
"""

from __future__ import annotations

import pytest
import torch


def _detect_device() -> torch.device:
    """Detect the best available device (GPU > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Auto-detected device fixture (session-scoped for reuse across tests)."""
    return _detect_device()


@pytest.fixture(scope="session")
def require_gpu() -> None:
    """Skip the test if no GPU is available."""
    if not torch.cuda.is_available():
        pytest.skip("Test requires a GPU")
