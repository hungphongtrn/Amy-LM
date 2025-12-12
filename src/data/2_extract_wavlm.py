import torch
from datasets import load_dataset, Features, Value, Array2D
from transformers import WavLMModel, Wav2Vec2Processor
import numpy as np

# CONFIG
SRC_REPO = "speechcolab/gigaspeech"
TGT_REPO = "hungphongtrn/wavlm-features"
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
    ds = load_dataset(SRC_REPO, "dev", split="validation") 

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

