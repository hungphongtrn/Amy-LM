import json
import os

import numpy as np
import pandas as pd
import pytest
import torch


class TestScorePairs:
    """Tests for embedding + cosine similarity scoring script."""

    # --- validation ---

    def test_validate_pairs_passes_on_good_data(self):
        from scripts.score_pairs import validate_pairs

        df = pd.DataFrame({
            "chosen": ["hello", "world"],
            "rejected": ["hi", "there"],
        })
        validate_pairs(df)

    def test_validate_pairs_raises_on_empty_chosen(self):
        from scripts.score_pairs import validate_pairs

        df = pd.DataFrame({
            "chosen": ["hello", ""],
            "rejected": ["hi", "there"],
        })
        with pytest.raises(ValueError, match="chosen"):
            validate_pairs(df)

    def test_validate_pairs_raises_on_empty_rejected(self):
        from scripts.score_pairs import validate_pairs

        df = pd.DataFrame({
            "chosen": ["hello", "world"],
            "rejected": ["hi", "  "],
        })
        with pytest.raises(ValueError, match="rejected"):
            validate_pairs(df)

    def test_validate_pairs_raises_on_nan(self):
        from scripts.score_pairs import validate_pairs

        df = pd.DataFrame({
            "chosen": ["hello", np.nan],
            "rejected": ["hi", "there"],
        })
        with pytest.raises(ValueError):
            validate_pairs(df)

    # --- cosine similarity computation ---

    def test_cosine_similarity_range(self):
        from scripts.score_pairs import compute_cosine_similarity

        rng = np.random.RandomState(42)
        a = torch.from_numpy(rng.randn(100, 768).astype(np.float32))
        b = torch.from_numpy(rng.randn(100, 768).astype(np.float32))

        sim = compute_cosine_similarity(a, b)
        assert sim.shape == (100,)
        assert sim.dtype == np.float32
        assert np.all(sim >= -1.0)
        assert np.all(sim <= 1.0)

    def test_cosine_similarity_identical(self):
        from scripts.score_pairs import compute_cosine_similarity

        a = torch.randn(10, 768)
        sim = compute_cosine_similarity(a, a)
        assert np.allclose(sim, 1.0, atol=1e-4)

    def test_cosine_similarity_orthogonal(self):
        from scripts.score_pairs import compute_cosine_similarity

        a = torch.eye(10)[:5].repeat(2, 1)
        b = torch.eye(10)[5:].repeat(2, 1)
        sim = compute_cosine_similarity(a, b)
        assert np.allclose(sim, 0.0, atol=1e-4)

    # --- output schema (integration with synthetic data + mock model) ---

    def test_output_parquet_schema(self, tmp_path, monkeypatch):
        import scripts.score_pairs as sp

        output_dir = tmp_path / "nvtts_preference_pairs"
        output_path = output_dir / "pairs_scored.parquet"

        monkeypatch.setattr(sp, "INPUT_PATH", str(tmp_path / "pairs.jsonl"))
        monkeypatch.setattr(sp, "OUTPUT_DIR", str(output_dir))
        monkeypatch.setattr(sp, "OUTPUT_PATH", str(output_path))

        pairs = [
            {"id": "a", "chosen": "great response", "rejected": "ok response",
             "rationale_chosen": "r1", "rationale_rejected": "r2"},
            {"id": "b", "chosen": "awesome", "rejected": "fine",
             "rationale_chosen": "r3", "rationale_rejected": "r4"},
        ]
        os.makedirs(os.path.dirname(str(tmp_path / "pairs.jsonl")), exist_ok=True)
        with open(tmp_path / "pairs.jsonl", "w") as f:
            for record in pairs:
                json.dump(record, f)
                f.write("\n")

        class MockModel:
            def encode(self, texts, **kwargs):
                n = len(texts)
                emb = torch.randn(n, 768) if n > 0 else torch.empty(0, 768)
                if kwargs.get("convert_to_tensor"):
                    return emb
                return emb.numpy()

        monkeypatch.setattr(sp, "SentenceTransformer", lambda name, device: MockModel())

        sp.main()

        df = pd.read_parquet(output_path)
        expected_cols = {"id", "chosen", "rejected", "rationale_chosen",
                          "rationale_rejected", "cosine_similarity"}
        assert set(df.columns) == expected_cols
        assert len(df) == 2
        assert df["cosine_similarity"].dtype == np.float32
        assert df["cosine_similarity"].between(-1.0, 1.0).all()
