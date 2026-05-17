"""Utilities for extracting FACodec codebook vectors from Amphion checkpoints."""

import torch


def load_prosody_codebook_vectors(checkpoint_path: str) -> torch.Tensor:
    """Load prosody codebook vectors from FACodec decoder checkpoint.

    The FACodec decoder checkpoint (ns3_facodec_decoder.bin) stores
    factorized codebook vectors at 8 dimensions per entry. The prosody
    codebook is the first quantizer in the ResidualVQ group (index 0).

    State dict key: ``quantizer.0.layers.0._codebook.weight``
    Shape: ``[1024, 8]``  (vocab_size=1024, codebook_dim=8)

    Args:
        checkpoint_path: Path to ns3_facodec_decoder.bin.

    Returns:
        Float32 tensor [1024, 8] — raw FACodec prosody codebook vectors.

    Raises:
        FileNotFoundError: If checkpoint_path does not exist.
        KeyError: If the prosody codebook key is not found in the checkpoint.
    """
    import os
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"FACodec decoder checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    key = "quantizer.0.layers.0._codebook.weight"

    if key not in state_dict:
        raise KeyError(
            f"Prosody codebook key '{key}' not found in checkpoint at "
            f"{checkpoint_path}. "
            f"Available keys: {list(state_dict.keys())[:10]}..."
        )

    return state_dict[key].float().detach()
