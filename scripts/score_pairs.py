"""Script 3: Embed chosen/rejected response pairs with EmbeddingGemma and compute cosine similarity.

Reads JSONL from data/nvtts_pairs/pairs.jsonl (output of Script 2).
Embeds chosen and rejected responses using google/embeddinggemma-300m,
computes pairwise cosine similarity, and outputs a parquet with all
provenance columns plus `cosine_similarity`.

All samples are retained — no hard filtering is applied.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

INPUT_PATH = "data/nvtts_pairs/pairs.jsonl"
OUTPUT_DIR = "data/nvtts_preference_pairs"
OUTPUT_PATH = f"{OUTPUT_DIR}/pairs_scored.parquet"
MODEL_NAME = "google/embeddinggemma-300m"
PROMPT_TEMPLATE = "task: sentence similarity | query: {text}"
BATCH_SIZE = 64


def validate_pairs(df: pd.DataFrame) -> None:
    """Raise ValueError if any chosen or rejected field is empty."""
    for col in ("chosen", "rejected"):
        if df[col].isna().any() or (df[col].astype(str).str.strip() == "").any():
            raise ValueError(f"Found empty values in '{col}' column")


def compute_cosine_similarity(
    embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
) -> np.ndarray:
    """Compute pairwise cosine similarity between two sets of normalized embeddings."""
    a = torch.nn.functional.normalize(embeddings_a.float(), p=2, dim=1)
    b = torch.nn.functional.normalize(embeddings_b.float(), p=2, dim=1)
    return (a * b).sum(dim=1).cpu().numpy().astype(np.float32)


def main() -> None:
    records = []
    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    validate_pairs(df)

    model = SentenceTransformer(MODEL_NAME, device="cuda")

    chosen_texts = [PROMPT_TEMPLATE.format(text=t) for t in df["chosen"]]
    rejected_texts = [PROMPT_TEMPLATE.format(text=t) for t in df["rejected"]]

    chosen_embs = model.encode(
        chosen_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_tensor=True
    )
    rejected_embs = model.encode(
        rejected_texts, batch_size=BATCH_SIZE, show_progress_bar=True, convert_to_tensor=True
    )

    df["cosine_similarity"] = compute_cosine_similarity(chosen_embs, rejected_embs)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
