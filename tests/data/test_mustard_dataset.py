from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset as HFDataset

from src.data.mustard_dataset import (
    MustardDataset,
    collate_mustard,
    create_mustard_splits,
)


def make_synthetic_parquet(path: Path, num_samples: int = 5) -> None:
    samples = []
    for i in range(num_samples):
        sr = 16000
        n_samples = int((2.0 + i * 0.5) * sr)
        audio = np.random.randn(n_samples).astype(np.float32)
        n_frames = n_samples // 200

        samples.append(
            {
                "dataset": "test",
                "id": f"sample_{i:03d}",
                "audio": {"array": audio, "sampling_rate": sr},
                "prosody_codebooks_idx": [j % 1024 for j in range(n_frames)],
                "content_codebooks_idx": [
                    [j % 1024 for j in range(n_frames)] for _ in range(2)
                ],
                "acoustic_codebooks_idx": [
                    [j % 1024 for j in range(n_frames)] for _ in range(3)
                ],
                "timbre_vector": [float(j % 256) / 256.0 for j in range(256)],
                "label": i % 2,
            }
        )

    ds = HFDataset.from_list(samples)
    ds.to_parquet(str(path))


def test_mustard_dataset_loads_and_returns_expected_shapes(tmp_path: Path) -> None:
    parquet_path = tmp_path / "mustard.parquet"
    make_synthetic_parquet(parquet_path, num_samples=3)

    dataset = MustardDataset(parquet_path)
    assert len(dataset) == 3

    audio, prosody, timbre, label = dataset[0]
    assert isinstance(audio, torch.Tensor)
    assert audio.dtype == torch.float32
    assert audio.ndim == 1

    assert isinstance(prosody, torch.Tensor)
    assert prosody.dtype == torch.long
    assert prosody.ndim == 2
    assert prosody.shape[0] == 1

    assert isinstance(timbre, torch.Tensor)
    assert timbre.dtype == torch.float32
    assert timbre.shape == (256,)

    assert isinstance(label, int)
    assert label in (0, 1)


def test_mustard_dataset_raises_on_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        MustardDataset(tmp_path / "missing.parquet")


def test_collate_mustard_pads_audio_and_prosody() -> None:
    batch = [
        (
            torch.ones(10, dtype=torch.float32),
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            torch.ones(256, dtype=torch.float32),
            0,
        ),
        (
            torch.ones(15, dtype=torch.float32),
            torch.tensor([[4, 5]], dtype=torch.long),
            torch.zeros(256, dtype=torch.float32),
            1,
        ),
    ]

    audio, prosody, timbre, labels = collate_mustard(batch)
    assert audio.shape == (2, 15)
    assert prosody.shape == (2, 1, 3)
    assert timbre.shape == (2, 256)
    assert labels.shape == (2,)
    assert labels.dtype == torch.long

    assert torch.equal(audio[0, 10:], torch.zeros(5, dtype=torch.float32))
    assert torch.equal(prosody[1, 0, 2:], torch.zeros(1, dtype=torch.long))


def test_create_mustard_splits_sizes_and_non_overlap(tmp_path: Path) -> None:
    parquet_path = tmp_path / "mustard.parquet"
    make_synthetic_parquet(parquet_path, num_samples=10)

    train_ds, val_ds, test_ds = create_mustard_splits(parquet_path, seed=42)

    assert len(train_ds) == 8
    assert len(val_ds) == 1
    assert len(test_ds) == 1
    assert len(train_ds) + len(val_ds) + len(test_ds) == 10


def test_create_mustard_splits_validates_fractions(tmp_path: Path) -> None:
    parquet_path = tmp_path / "mustard.parquet"
    make_synthetic_parquet(parquet_path, num_samples=10)

    with pytest.raises(ValueError, match=r"train_frac must be in \(0, 1\)"):
        create_mustard_splits(parquet_path, train_frac=0.0, val_frac=0.1)

    with pytest.raises(ValueError, match=r"val_frac must be in \(0, 1\)"):
        create_mustard_splits(parquet_path, train_frac=0.8, val_frac=1.0)

    with pytest.raises(ValueError, match=r"train_frac \+ val_frac"):
        create_mustard_splits(parquet_path, train_frac=0.8, val_frac=0.2)
