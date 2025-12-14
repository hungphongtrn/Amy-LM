import torch
from datasets import load_dataset
from transformers import WavLMModel, Wav2Vec2FeatureExtractor
import numpy as np

# CONFIG
SRC_REPO = "fixie-ai/gigaspeech"
TGT_REPO = "hungphongtrn/wavlm-features"
BATCH_SIZE = 16
MODEL_ID = "microsoft/wavlm-large"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Extracting WavLM on {device}")
    
    # 1. Model
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
    model = WavLMModel.from_pretrained(MODEL_ID).to(device)
    model.eval()
    
    # 2. Data
    ds = load_dataset(SRC_REPO, "dev", split="dev") 
    
    def extract_batch(batch):
        # HF Audio feature automatically decodes to array
        arrays = [x['array'] for x in batch['audio']]
        
        # Feature extractor handles padding/batching
        inputs = feature_extractor(
            arrays, 
            sampling_rate=16000, 
            return_tensors="pt", 
            padding=True
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Extract frame-level features from final transformer layer
            # Shape: [Batch, Time, 1024]
            hidden_states = outputs.last_hidden_state
        
        # Post-process: Remove padding based on attention mask
        results = []
        for i, feat in enumerate(hidden_states):
            # attention_mask is already at feature level (after CNN downsampling)
            # Just sum to get the number of valid frames
            feat_len = inputs.attention_mask[i].sum().item()
            
            # Slice to valid length and convert to float16
            clean_feat = feat[:feat_len].cpu().numpy().astype(np.float16)
            results.append(clean_feat)
        
        # Save as numpy array
        return {"wavlm_feat": results}
    

    columns_to_remove = [col for col in ds.column_names if col not in ['segment_id']]
    
    ds_wavlm = ds.map(
        extract_batch, 
        batched=True, 
        batch_size=BATCH_SIZE,
        remove_columns=columns_to_remove  # Keep only ID and Features
    )
    
    # 4. Push
    ds_wavlm.push_to_hub(TGT_REPO)
    print("✅ WavLM Features Pushed!")

if __name__ == "__main__":
    main()