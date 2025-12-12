# **Distributed Data Processing Plan**.

This architecture uses **Hugging Face Hub** as the synchronization layer. You will not manually copy files between machines. Instead, Machine A pushes results to the Hub, and Machine B pulls them.

## **Hardware Requirements**

* **RAM:** **512GB+ System RAM**. We are processing large batches; running out of RAM causes crashes.
* **GPU:** **24GB+ VRAM** (A10G, A100, or 3090/4090). Crucial for large batch inference.
* **CPU:** **4-8 Cores** is sufficient. We are GPU-bound; we just need enough CPU to feed the DataLoader.
* **Network:** Fast internet (for pulling/pushing to HF Hub).

-----

## **0. Preparation: Upload Raw Data**

Before starting, ensure your raw audio and text are on the Hub. This acts as the "Source of Truth."

* **Dataset:** `speechcolab/gigaspeech` (dev subset, e.g., `validation`)
* **Columns:** `audio` (Audio), `text` (string), `segment_id` (string)

-----

## **Script 1: Alignment (NeMo Parakeet)**

**Dependency:** None (uses Raw Audio).
**Output:** `hungphongtrn/speech-time-alignment` (Contains word-level timestamps).

This script downloads audio, runs transcription/alignment in batches using the TDT model's native timestamp capability, and pushes the results.

**File:** `src/data/1_extract_alignment.py`

```python
import os
import torch
import json
import tempfile
from datasets import load_dataset, Dataset, Features, Value, Sequence
from nemo.collections.asr.models import ASRModel

# CONFIG
SRC_REPO = "speechcolab/gigaspeech"
TGT_REPO = "hungphongtrn/speech-time-alignment"
BATCH_SIZE = 16  # Adjust based on VRAM
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    print(f"🚀 Starting Alignment on {torch.cuda.get_device_name(0)}")
    
    # 1. Load NeMo Model (Parakeet TDT)
    # Automatically downloads the model
    model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3").cuda()
    model.eval()

    # 2. Load Source Data
    # using "validation" split as the dev subset
    ds = load_dataset(SRC_REPO, split="validation", token=HF_TOKEN)
    
    # 3. Batch Processing Function
    def align_batch(batch):
        audio_paths = []
        ids = batch['segment_id']
        
        # Save temp files for NeMo (it expects file paths)
        temp_dir = tempfile.mkdtemp()
        for i, audio_array in enumerate(batch['audio']):
            # Assuming audio is decoded by HF to {'array': ..., 'sampling_rate': ...}
            # We need to save it as wav for NeMo
            import soundfile as sf
            path = os.path.join(temp_dir, f"{ids[i]}.wav")
            sf.write(path, audio_array['array'], audio_array['sampling_rate'])
            audio_paths.append(path)

        # Run Transcription with Timestamps
        # Parakeet TDT v3 supports native timestamp generation
        with torch.no_grad():
             outputs = model.transcribe(paths2audio_files=audio_paths, batch_size=len(audio_paths), timestamps=True)
        
        aligned_data = []
        # outputs is a list of objects with .text and .timestamp attributes
        for out in outputs:
            # out.timestamp['word'] contains list of {'word': ..., 'start': ..., 'end': ...}
            word_timestamps = out.timestamp['word']
            aligned_data.append(json.dumps(word_timestamps)) # Store as JSON string
            
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir)
        
        return {"alignment_json": aligned_data}

    # 4. Process
    # We use batching to maximize throughput
    ds_aligned = ds.map(align_batch, batched=True, batch_size=BATCH_SIZE, remove_columns=["audio"])

    # 5. Push
    ds_aligned.push_to_hub(TGT_REPO, token=HF_TOKEN)
    print("✅ Alignment Data Pushed!")

if __name__ == "__main__":
    main()
```

*Note: For strict alignment, calling the `nemo_forced_aligner` CLI inside the loop on the temp files is the most robust method.*

-----

## **Script 2: Acoustic Extraction (WavLM)**

**Dependency:** None (uses Raw Audio).
**Output:** `hungphongtrn/mimi-wavlm` (Contains 50Hz features).
**Key:** High VRAM usage.

**File:** `src/data/2_extract_wavlm.py`

```python
import torch
from datasets import load_dataset, Features, Value, Array2D
from transformers import WavLMModel, Wav2Vec2Processor
import numpy as np

# CONFIG
SRC_REPO = "speechcolab/gigaspeech"
TGT_REPO = "hungphongtrn/mimi-wavlm"
BATCH_SIZE = 32
MODEL_ID = "microsoft/wavlm-large"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Extracting WavLM on {device}")

    # 1. Model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = WavLMModel.from_pretrained(MODEL_ID).to(device)
    model.eval()

    # 2. Data
    # 'streaming=True' is good if local disk is small, but 'False' is faster if you have RAM.
    ds = load_dataset(SRC_REPO, split="validation") 

    def extract_batch(batch):
        # HF Audio feature automatically decodes to array
        arrays = [x['array'] for x in batch['audio']]
        
        # Processor handles padding/batching
        inputs = processor(
            arrays, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=16000*20 # Limit to 20s to prevent OOM
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # [Batch, Time, 1024]
            hidden_states = outputs.last_hidden_state

        # Post-process: Remove padding based on attention mask
        results = []
        for i, feat in enumerate(hidden_states):
            # Get valid length
            valid_len = inputs.attention_mask[i].sum().item()
            # WavLM downsampling factor is 320 (20ms)
            feat_len = int(valid_len / 320) # Approximation, or rely on output shape
            
            # Slice and cast to FP16 to save storage
            clean_feat = feat[:feat_len].cpu().numpy().astype(np.float16)
            results.append(clean_feat)

        return {"wavlm_feat": results}

    # 3. Process
    # Define features explicitly to allow Array2D storage
    features = ds.features.copy()
    features["wavlm_feat"] = Array2D(shape=(None, 1024), dtype="float16")
    del features["audio"] # Don't save audio again

    ds_wavlm = ds.map(
        extract_batch, 
        batched=True, 
        batch_size=BATCH_SIZE, 
        features=features,
        remove_columns=["audio", "text"] # Keep only ID and Features
    )

    # 4. Push
    ds_wavlm.push_to_hub(TGT_REPO)
    print("✅ WavLM Features Pushed!")

if __name__ == "__main__":
    main()
```

-----

## **Script 3: Semantic Extraction (LLM + Bridge)**

**Dependency:** `hungphongtrn/speech-time-alignment` (Needs timestamps to align).
**Output:** `hungphongtrn/mimi-llm` (Contains expanded/aligned tokens).

**File:** `src/data/3_extract_llm.py`

```python
import torch
import json
import numpy as np
from datasets import load_dataset, Features, Value, Array2D
from transformers import AutoModelForCausalLM, AutoTokenizer

# CONFIG
ALIGN_REPO = "hungphongtrn/speech-time-alignment" # Contains text & timestamps
TGT_REPO = "hungphongtrn/mimi-llm"
BATCH_SIZE = 8 # LLMs are memory hungry!
MODEL_ID = "Qwen/Qwen3-1.7B"

def main():
    device = "cuda"
    print(f"🚀 Extracting LLM Feats on {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Fix padding
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(device)
    model.eval()

    ds = load_dataset(ALIGN_REPO, split="train")

    def extract_batch(batch):
        # Batch['text'] comes from the alignment dataset
        inputs = tokenizer(batch['text'], return_tensors="pt", padding=True, truncation=True).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Layer 27 (index -2)
            states = outputs.hidden_states[-2] # [B, T, 2048]

        results_feat = []
        results_time = []

        for i, text in enumerate(batch['text']):
            # 1. Get Embeddings
            # Remove padding
            valid_len = inputs.attention_mask[i].sum().item()
            feat = states[i, :valid_len, :].cpu().numpy().astype(np.float16)
            
            # 2. Get Timestamps (The Bridge)
            # Parse the JSON string we saved in Step 1
            align_data = json.loads(batch['alignment_json'][i])
            
            # ... Implement the Token-to-Word alignment logic here ...
            # (As discussed: Heuristic map token indices to word time boundaries)
            # For this script, we placeholder the shape:
            times = np.zeros((valid_len, 2), dtype=np.float32) 
            
            results_feat.append(feat)
            results_time.append(times)

        return {"llm_feat": results_feat, "llm_times": results_time}

    # Define schema
    features = ds.features.copy()
    features["llm_feat"] = Array2D(shape=(None, 2048), dtype="float16")
    features["llm_times"] = Array2D(shape=(None, 2), dtype="float32")
    
    ds_llm = ds.map(extract_batch, batched=True, batch_size=BATCH_SIZE)
    
    ds_llm.push_to_hub(TGT_REPO)
    print("✅ LLM Features Pushed!")

if __name__ == "__main__":
    main()
```

-----

## **How to Run in Parallel**

Since we use Hugging Face `map` with batching, parallelization happens **per-machine** easily.

1. **Sharding:** If you have 4 machines, you don't need complex logic. Just use HF `shard`:

    ```python
    # Inside the script, change:
    ds = load_dataset(..., split="train")

    # TO:
    import argparse
    parser.add_argument("--rank", type=int)
    parser.add_argument("--world_size", type=int)
    args = parser.parse_args()

    ds = load_dataset(..., split=f"train[{args.rank}%{args.world_size}::{args.world_size}]")
    ```

2. **Execution:**

      * **Machine 1:** `python src/data/2_extract_wavlm.py --rank 0 --world_size 4`
      * **Machine 2:** `python src/data/2_extract_wavlm.py --rank 1 --world_size 4`
      * ...

The HF `push_to_hub` will handle the merging smartly (or you can push to different branches/subsets if you want strict safety, but usually sequential pushes work fine for datasets).

**Final Note on Resources:**
Ensure your machines have ample **System RAM** (64GB+). When `load_dataset` processes audio, it can spike memory usage before sending tensors to the GPU. If you see "Killed" errors, reduce `BATCH_SIZE` or increase `num_proc` in the `.map()` call carefully.
