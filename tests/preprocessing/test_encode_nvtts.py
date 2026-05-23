from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset

import scripts.encode_nvtts as encode_nvtts


def _build_synthetic_split(start: int, size: int) -> Dataset:
    items = []
    sr = 48000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    for i in range(start, start + size):
        audio = (0.25 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
        items.append({"index": str(i), "audio": {"array": audio, "sampling_rate": sr}})
    return Dataset.from_list(items)


def _build_pairs_df(total: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [str(i) for i in range(total)],
            "chosen": [f"chosen-{i}" for i in range(total)],
            "rejected": [f"rejected-{i}" for i in range(total)],
            "cosine_similarity": np.linspace(0.1, 0.9, total, dtype=np.float32),
        }
    )


def test_script_smoke_with_mock(tmp_path: Path, monkeypatch):
    splits = {
        "train": _build_synthetic_split(0, 3),
        "dev": _build_synthetic_split(3, 2),
        "test": _build_synthetic_split(5, 1),
    }
    pairs_df = _build_pairs_df(6)

    monkeypatch.setattr(encode_nvtts, "load_dataset", lambda *args, split=None, **kwargs: splits[split])
    monkeypatch.setattr(encode_nvtts.pd, "read_parquet", lambda *_args, **_kwargs: pairs_df)

    import sys

    monkeypatch.setattr(
        sys,
        "argv",
        ["encode_nvtts.py", "--mock", "--no-push", "--device", "cpu", "--output-dir", str(tmp_path)],
    )

    assert encode_nvtts.main() == 0

    for split in ("train", "dev", "test"):
        out_file = tmp_path / f"{split}.parquet"
        assert out_file.exists()
        ds = Dataset.from_parquet(str(out_file))
        expected_cols = {
            "index", "audio", "prosody_codebooks_idx", "content_codebooks_idx",
            "acoustic_codebooks_idx", "timbre_vector", "chosen", "rejected", "cosine_similarity",
        }
        assert expected_cols.issubset(set(ds.column_names))


def test_mock_stream_shapes():
    split = _build_synthetic_split(0, 1)
    pairs_df = _build_pairs_df(1)
    encoder = encode_nvtts.FACodecEncoder(device="cpu", force_mock=True)
    rows = encode_nvtts.encode_split(split, pairs_df, encoder, batch_size=1)

    row = rows[0]
    assert len(row["prosody_codebooks_idx"]) > 0
    assert len(row["content_codebooks_idx"]) == 2
    assert len(row["acoustic_codebooks_idx"]) == 3
    assert len(row["timbre_vector"]) == 256


def test_split_preservation(tmp_path: Path, monkeypatch):
    splits = {
        "train": _build_synthetic_split(0, 4),
        "dev": _build_synthetic_split(4, 3),
        "test": _build_synthetic_split(7, 2),
    }
    pairs_df = _build_pairs_df(9)
    monkeypatch.setattr(encode_nvtts, "load_dataset", lambda *args, split=None, **kwargs: splits[split])
    monkeypatch.setattr(encode_nvtts.pd, "read_parquet", lambda *_args, **_kwargs: pairs_df)

    import sys

    monkeypatch.setattr(sys, "argv", ["encode_nvtts.py", "--mock", "--no-push", "--device", "cpu", "--output-dir", str(tmp_path)])
    encode_nvtts.main()

    assert len(Dataset.from_parquet(str(tmp_path / "train.parquet"))) == 4
    assert len(Dataset.from_parquet(str(tmp_path / "dev.parquet"))) == 3
    assert len(Dataset.from_parquet(str(tmp_path / "test.parquet"))) == 2


def test_join_integrity():
    split = _build_synthetic_split(0, 3)
    pairs_df = _build_pairs_df(3)
    encoder = encode_nvtts.FACodecEncoder(device="cpu", force_mock=True)
    rows = encode_nvtts.encode_split(split, pairs_df, encoder, batch_size=2)
    got = [r["index"] for r in rows]
    assert got == ["0", "1", "2"]
    assert rows[1]["chosen"] == "chosen-1"
    assert rows[1]["rejected"] == "rejected-1"
