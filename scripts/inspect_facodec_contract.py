"""Print FACodec tensor shapes for issue #12 contract verification.

This script is intentionally read-only. It does not preprocess datasets or save
model outputs; it only loads FACodec, runs one synthetic waveform, and prints the
returned tensor shapes.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.preprocessing.facodec_encoder import FACodecEncoder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seconds", type=float, default=1.0)
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    args = parser.parse_args()

    encoder = FACodecEncoder(
        device=args.device,
        checkpoint_path=str(args.checkpoint_path) if args.checkpoint_path else None,
    )
    if encoder._mock:
        raise SystemExit(
            "FACodec real checkpoint unavailable; inspector requires real Amphion FACodec."
        )

    samples = int(encoder.SAMPLE_RATE * args.seconds)
    audio = torch.zeros(samples, device=args.device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        enc_out = encoder._encoder(audio)
        vq_post_emb, vq_id, unknown, quantized, spk_embs = encoder._decoder(
            enc_out, eval_vq=False, vq=True
        )

    print(f"enc_out:           {tuple(enc_out.shape)} {enc_out.dtype}")
    print(f"vq_post_emb:       {tuple(vq_post_emb.shape)} {vq_post_emb.dtype}")
    print(f"vq_id:             {tuple(vq_id.shape)} {vq_id.dtype}")
    print(f"unknown_return_3:  {type(unknown).__name__}")
    print(f"quantized:         list (len={len(quantized)})")
    if isinstance(quantized, list) and len(quantized) > 0:
        print(f"quantized[0]:      {tuple(quantized[0].shape)} {quantized[0].dtype}")
    print(f"spk_embs:          {tuple(spk_embs.shape)} {spk_embs.dtype}")
    print(f"timbre_vector:     {tuple(spk_embs.squeeze().shape)} {spk_embs.dtype}")
    print(f"prosody vq_id[:1]: {tuple(vq_id[:1].shape)}")
    print(f"content vq_id[1:3]:{tuple(vq_id[1:3].shape)}")
    print(f"acoustic vq_id[3:]:{tuple(vq_id[3:].shape)}")


if __name__ == "__main__":
    main()
