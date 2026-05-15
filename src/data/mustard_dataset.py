from pathlib import Path
from typing import Union

import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset, random_split


class MustardDataset(Dataset):
    """PyTorch Dataset for FACodec-preprocessed MUStARD parquet."""

    def __init__(self, parquet_path: Union[str, Path]):
        super().__init__()
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"Parquet file not found: {path}")

        dataset = HFDataset.from_parquet(str(path))
        self.samples = [
            (
                s["audio"]["array"],
                s["prosody_codebooks_idx"],
                s["timbre_vector"],
                int(s["label"]),
            )
            for s in dataset
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        audio_arr, prosody_list, timbre_list, label = self.samples[idx]
        audio = torch.tensor(audio_arr, dtype=torch.float32)
        prosody = torch.tensor(prosody_list, dtype=torch.long).unsqueeze(0)
        timbre = torch.tensor(timbre_list, dtype=torch.float32)
        return audio, prosody, timbre, label


def collate_mustard(
    batch: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length MUStARD samples into a batch."""
    audios, prosodies, timbres, labels = zip(*batch)

    max_audio = max(a.shape[0] for a in audios)
    audio_padded = torch.stack([F.pad(a, (0, max_audio - a.shape[0])) for a in audios])

    max_pro = max(p.shape[1] for p in prosodies)
    prosody_padded = torch.stack([F.pad(p, (0, max_pro - p.shape[1])) for p in prosodies])

    timbre_stacked = torch.stack(timbres)
    labels_stacked = torch.tensor(labels, dtype=torch.long)

    return audio_padded, prosody_padded, timbre_stacked, labels_stacked


def create_mustard_splits(
    parquet_path: Union[str, Path],
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
):
    """Create random train/val/test dataset splits."""
    if not (0 < train_frac < 1):
        raise ValueError(f"train_frac must be in (0, 1), got {train_frac}")
    if not (0 < val_frac < 1):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    if train_frac + val_frac >= 1.0:
        raise ValueError(
            f"train_frac + val_frac ({train_frac + val_frac}) must be < 1.0 "
            f"to leave room for test split"
        )

    dataset = MustardDataset(parquet_path)
    n = len(dataset)
    train_n = int(n * train_frac)
    val_n = int(n * val_frac)
    test_n = n - train_n - val_n

    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_n, val_n, test_n], generator=generator)
