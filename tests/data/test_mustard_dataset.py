from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset as HFDataset

from src.data.mustard_dataset import (
    MustardDataset,
    ShuffledFacodecDataset,
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

    audio, prosody, timbre, labels, audio_lengths, prosody_lengths = collate_mustard(batch)
    assert audio.shape == (2, 15)
    assert prosody.shape == (2, 1, 3)
    assert timbre.shape == (2, 256)
    assert labels.shape == (2,)
    assert labels.dtype == torch.long
    assert torch.equal(audio_lengths, torch.tensor([10, 15], dtype=torch.long))
    assert torch.equal(prosody_lengths, torch.tensor([3, 2], dtype=torch.long))

    assert torch.equal(audio[0, 10:], torch.zeros(5, dtype=torch.float32))
    assert torch.equal(prosody[1, 0, 2:], torch.zeros(1, dtype=torch.long))


def test_create_mustard_splits_sizes(tmp_path: Path) -> None:
    """create_mustard_splits should produce expected split sizes."""
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


class TestShuffledFacodecDataset:
    """Tests for ShuffledFacodecDataset — FACodec negative control."""

    def test_preserves_audio_and_label(self, tmp_path: Path) -> None:
        """Audio and label come from the original index, not the donor."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=10)
        base = MustardDataset(parquet_path)
        shuffled = ShuffledFacodecDataset(base, seed=42)

        for i in range(len(base)):
            base_audio, _, _, base_label = base[i]
            shuf_audio, _, _, shuf_label = shuffled[i]
            assert torch.equal(shuf_audio, base_audio), f"Audio mismatch at index {i}"
            assert shuf_label == base_label, f"Label mismatch at index {i}"

    def test_prosody_timbre_are_from_same_donor(self, tmp_path: Path) -> None:
        """Prosody and timbre come from the same donor index."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=10)
        base = MustardDataset(parquet_path)
        shuffled = ShuffledFacodecDataset(base, seed=42)

        for i in range(len(base)):
            _, s_pros, s_timb, _ = shuffled[i]
            donor_idx = int(shuffled._perm[i])
            _, d_pros, d_timb, _ = base[donor_idx]
            assert torch.equal(s_pros, d_pros), f"Prosody donor mismatch at index {i}"
            assert torch.equal(s_timb, d_timb), f"Timbre donor mismatch at index {i}"

    def test_is_derangement(self, tmp_path: Path) -> None:
        """No index maps to itself when n > 1."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=10)
        base = MustardDataset(parquet_path)
        shuffled = ShuffledFacodecDataset(base, seed=42)

        for i in range(len(base)):
            donor = int(shuffled._perm[i])
            assert donor != i, f"Index {i} maps to itself (not a derangement)"

    def test_deterministic(self, tmp_path: Path) -> None:
        """Same seed produces same derangement."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=10)
        base = MustardDataset(parquet_path)

        s1 = ShuffledFacodecDataset(base, seed=42)
        s2 = ShuffledFacodecDataset(base, seed=42)
        assert torch.equal(s1._perm, s2._perm)

    def test_single_sample_no_derangement(self, tmp_path: Path) -> None:
        """Single-sample split returns identity permutation."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=1)
        base = MustardDataset(parquet_path)
        shuffled = ShuffledFacodecDataset(base, seed=42)

        assert len(shuffled) == 1
        assert int(shuffled._perm[0]) == 0
        base_audio, _, _, base_label = base[0]
        shuf_audio, shuf_pros, shuf_timb, shuf_label = shuffled[0]
        assert torch.equal(shuf_audio, base_audio)
        assert shuf_label == base_label
        _, d_pros, d_timb, _ = base[int(shuffled._perm[0])]
        assert torch.equal(shuf_pros, d_pros)
        assert torch.equal(shuf_timb, d_timb)

    def test_different_seed_different_derangement(self, tmp_path: Path) -> None:
        """Different seeds produce different derangements."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=10)
        base = MustardDataset(parquet_path)

        s1 = ShuffledFacodecDataset(base, seed=42)
        s2 = ShuffledFacodecDataset(base, seed=99)
        assert not torch.equal(s1._perm, s2._perm)

    def test_works_with_subset(self, tmp_path: Path) -> None:
        """ShuffledFacodecDataset wraps a Subset (from random_split)."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=10)
        train_ds, _, _ = create_mustard_splits(parquet_path, seed=42)

        shuffled = ShuffledFacodecDataset(train_ds, seed=42)
        assert len(shuffled) == len(train_ds)

        for i in range(len(shuffled)):
            audio, pros, timb, label = shuffled[i]
            assert audio.dtype == torch.float32
            assert pros.dtype == torch.long
            assert timb.dtype == torch.float32
            assert label in (0, 1)
            donor = int(shuffled._perm[i])
            assert donor != i, f"Derangement failed for Subset index {i}"

    def test_collate_works_with_shuffled(self, tmp_path: Path) -> None:
        """collate_mustard works on batches from ShuffledFacodecDataset."""
        parquet_path = tmp_path / "mustard.parquet"
        make_synthetic_parquet(parquet_path, num_samples=5)
        base = MustardDataset(parquet_path)
        shuffled = ShuffledFacodecDataset(base, seed=42)

        batch = [shuffled[i] for i in range(3)]
        audio, prosody, timbre, labels, a_lens, p_lens = collate_mustard(batch)

        assert audio.shape[0] == 3
        assert prosody.shape[0] == 3
        assert timbre.shape[0] == 3
        assert labels.shape[0] == 3
        assert a_lens.shape[0] == 3
        assert p_lens.shape[0] == 3
