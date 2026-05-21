#!/usr/bin/env python3
"""Evaluate trained Amy LM on DynamicSuperb SarcasmDetection_Mustard using speech + instruction.

Instead of the classifier head, this script feeds the fused FACodec-augmented
semantic stream as input embeddings to the Qwen3 language model, appends the
dataset's instruction as text tokens, and lets the LM generate a yes/no response.

Usage:
    python scripts/infer_amy.py \
        --checkpoint checkpoints/training/best_model.pt \
        --device cuda \
        --max-samples 50
"""

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models import AmyForProsodyClassification
from src.models.codebook_utils import load_prosody_codebook_vectors
from src.inference import AmyInference, resample_audio
from src.preprocessing.facodec_encoder import FACodecEncoder

AUDIO_MAX_SAMPLES = 160000


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate Amy LM on speech + instruction sarcasm detection"
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (e.g. best_model.pt)",
    )
    p.add_argument(
        "--facodec-checkpoint",
        type=str,
        default="checkpoints/facodec/ns3_facodec_decoder.bin",
        help="Path to FACodec decoder checkpoint for warm-starting prosody embedding",
    )
    p.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Max samples to evaluate (None = all 200)",
    )
    p.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inference",
        help="Directory for inference results JSON",
    )
    return p.parse_args()


def load_model(checkpoint_path: str, facodec_checkpoint: str, device: torch.device):
    vectors = load_prosody_codebook_vectors(facodec_checkpoint)
    model = AmyForProsodyClassification(warm_start_vectors=vectors, device=device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_state = {
        k: v
        for k, v in ckpt["model_state_dict"].items()
        if not k.startswith("wrapper.")
    }
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()
    return model


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    print("Loading trained Amy model ...")
    model = load_model(args.checkpoint, args.facodec_checkpoint, device)

    # --- Load tokenizer ---
    model_id = model.wrapper.model_id
    print(f"Loading tokenizer from {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- FACodec encoder ---
    print("Initialising FACodec encoder ...")
    encoder = FACodecEncoder(device=device)
    if encoder._mock:
        print("WARNING: Using mock FACodec encoder — results will be invalid.")

    # --- Dataset ---
    print("Loading DynamicSuperb SarcasmDetection_Mustard ...")
    ds = load_dataset("DynamicSuperb/SarcasmDetection_Mustard", split="test")
    if args.max_samples:
        ds = ds.select(range(args.max_samples))
    print(f"Evaluating on {len(ds)} samples")

    inference = AmyInference(model=model, encoder=encoder, tokenizer=tokenizer, device=device)

    correct = 0
    total = 0
    results = []

    for idx, sample in enumerate(ds):
        audio_dict = sample["audio"]
        audio_arr = audio_dict.get("array", audio_dict) if isinstance(audio_dict, dict) else audio_dict
        samplerate = audio_dict.get("sampling_rate", 16000) if isinstance(audio_dict, dict) else 16000
        instruction = sample.get("instruction", "Is the speaker being sarcastic?")
        ground_truth = sample["label"]
        gt_label = int(ground_truth)

        # Add constraint to instruction for cleaner LM output
        instruction = instruction.strip().rstrip(".")
        instruction += ".\nAnswer with only 'Yes' or 'No'."

        try:
            # --- Audio preprocessing ---
            audio = torch.tensor(audio_arr, dtype=torch.float32)
            if audio.numel() == 0:
                raise ValueError("Empty audio")
            audio = resample_audio(audio, samplerate)
            if audio.numel() > AUDIO_MAX_SAMPLES:
                audio = audio[:AUDIO_MAX_SAMPLES]

            streams = encoder.encode(audio)
            out = inference.predict(
                audio=audio.unsqueeze(0),
                instruction=instruction,
                prosody_indices=streams.prosody_codebooks_idx.unsqueeze(0),
                timbre_vector=streams.timbre_vector.unsqueeze(0),
            )
            pred = int(out["prediction"])
            response = str(out["response"])

            if pred == gt_label:
                correct += 1
            total += 1

            results.append({
                "idx": idx,
                "file": sample.get("file", f"sample_{idx}"),
                "utterance": sample.get("utterance", "")[:120],
                "instruction": sample.get("instruction", ""),
                "ground_truth": bool(ground_truth),
                "prediction": bool(pred),
                "response": response,
                "correct": pred == gt_label,
            })

            if (idx + 1) % 10 == 0:
                print(f"  [{idx + 1}/{len(ds)}] Acc so far: {correct / total:.3f}")

        except Exception as exc:
            print(f"  [ERROR] Sample {idx} ({sample.get('file', '?')}): {exc}")
            total += 1
            results.append({
                "idx": idx,
                "file": sample.get("file", f"sample_{idx}"),
                "error": str(exc),
                "ground_truth": bool(ground_truth),
                "correct": False,
            })

    acc = correct / total if total > 0 else 0
    print(f"\nAccuracy: {acc:.3f}  ({correct}/{total} correct)")

    # --- Save results ---
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "inference_results.json", "w") as f:
        json.dump({
            "accuracy": acc,
            "correct": correct,
            "total": total,
            "checkpoint": args.checkpoint,
            "results": results,
        }, f, indent=2)
    print(f"Results saved to {out_dir / 'inference_results.json'}")

    return 0 if acc >= 0.5 else 1


if __name__ == "__main__":
    sys.exit(main())
