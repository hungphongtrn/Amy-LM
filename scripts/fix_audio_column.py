#!/usr/bin/env python3
"""Fix audio column on CREMA-D to proper Audio(sampling_rate=16000) format.

The existing dataset has audio stored as:
    {"array": {"bytes": ..., "path": ...}, "sampling_rate": 16000}
    
This needs to be flattened to:
    {"bytes": ..., "path": ...}
    
with the Audio(sampling_rate=16000) feature type.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datasets import load_dataset, Dataset, Audio, Features, Value, Sequence


def fix_audio(sample):
    audio = sample["audio"]
    if isinstance(audio, dict) and "array" in audio:
        nested = audio["array"]
        if isinstance(nested, dict) and "bytes" in nested:
            return {
                "audio": {
                    "bytes": nested["bytes"],
                    "path": nested.get("path") or "unknown.wav",
                }
            }
    return sample


def main():
    print("Loading dataset...")
    ds = load_dataset("hungphongtrn/crema_d_facodec", split="train")
    print(f"  {len(ds)} rows, features: {ds.features}")

    print("Flattening audio column...")
    ds = ds.map(fix_audio)
    print(f"  Sample audio: {ds[0]['audio']}")

    print("Casting to Audio(sampling_rate=16000)...")
    # Updated schema per Issue #12: FACodec Stream Contract
    features = Features({
        "dataset": Value("string"),
        "id": Value("string"),
        "audio": Audio(sampling_rate=16000),
        "prosody_codebooks_idx": Sequence(Value("int64")),  # [T80]
        "content_codebooks_idx": Sequence(Sequence(Value("int64"))),  # [2, T80]
        "acoustic_codebooks_idx": Sequence(Sequence(Value("int64"))),  # [3, T80]
        "timbre_vector": Sequence(Value("float32")),  # [256]
    })
    ds = ds.cast(features)
    print(f"  Features: {ds.features}")
    print(f"  Audio type: {type(ds[0]['audio'])}")

    print("Pushing to hub...")
    ds.push_to_hub("hungphongtrn/crema_d_facodec")
    print("Done!")


if __name__ == "__main__":
    main()