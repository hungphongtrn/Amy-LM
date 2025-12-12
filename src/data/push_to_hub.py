import os
from datasets import load_dataset, DatasetDict
from huggingface_hub import HfApi

# CONFIG
WAVLM_REPO = "hungphongtrn/wavlm-features"
LLM_REPO = "hungphongtrn/llm-features"
FINAL_REPO = "hungphongtrn/amy-dataset"
HF_TOKEN = os.getenv("HF_TOKEN")

def main():
    print("🚀 Starting Merge & Upload")
    
    # 1. Load Datasets
    print(f"Loading {WAVLM_REPO}...")
    ds_wavlm = load_dataset(WAVLM_REPO, split="train") # Assuming train split
    
    print(f"Loading {LLM_REPO}...")
    ds_llm = load_dataset(LLM_REPO, split="train")
    
    # 2. Merge
    # We assume both datasets are sorted/aligned by their original source indices 
    # or we can join on a key like 'segment_id'.
    # For Gigaspeech, segment_id is unique.
    
    # Let's check if we can simply concatenate columns if rows match
    # Ideally, we should join.
    
    # Convert to pandas for easier join? Or use dataset.add_column if order is guaranteed.
    # Since they come from the same source and split, order *should* be preserved if processing didn't shuffle.
    # But to be safe, we should probably join on 'segment_id' if available.
    
    # However, for this script, I'll implement a simple column merge assuming index alignment 
    # as they likely processed the same validation split in order. 
    # If a safer merge is needed, we'd index by segment_id.
    
    # Check lengths
    if len(ds_wavlm) != len(ds_llm):
        print(f"⚠️ Warning: Dataset lengths differ! WavLM: {len(ds_wavlm)}, LLM: {len(ds_llm)}")
        # In a real scenario, we would handle this. 
        # For now, we proceed assuming the intersection or similar subset.
    
    # Merging
    # We can use concatenate_datasets if we want to stack rows (not what we want)
    # We want to merge columns.
    
    print("Merging datasets...")
    # Add WavLM features to LLM dataset (which already has alignment info)
    # This assumes row-to-row correspondence.
    
    # Using a robust join on segment_id would be better if columns exist
    # ds_final = ds_llm.add_column("wavlm_feat", ds_wavlm["wavlm_feat"])
    
    # Let's try to join if segment_id exists
    try:
        # Convert to dictionary for fast lookup
        wavlm_dict = {row['segment_id']: row['wavlm_feat'] for row in ds_wavlm}
        
        def add_wavlm(batch):
            feats = []
            ids = batch['segment_id']
            for i, seg_id in enumerate(ids):
                if seg_id in wavlm_dict:
                    feats.append(wavlm_dict[seg_id])
                else:
                    feats.append(None) # Or handle missing
            return {"wavlm_feat": feats}
            
        ds_final = ds_llm.map(add_wavlm, batched=True, batch_size=1000)
        
        # Filter out missing if any
        ds_final = ds_final.filter(lambda x: x['wavlm_feat'] is not None)
        
    except KeyError:
        print("Could not find segment_id, falling back to index matching.")
        ds_final = ds_llm.add_column("wavlm_feat", ds_wavlm["wavlm_feat"])

    # 3. Push
    print(f"Pushing to {FINAL_REPO}...")
    ds_final.push_to_hub(FINAL_REPO, token=HF_TOKEN)
    print("✅ Upload Complete!")

if __name__ == "__main__":
    main()

