import torch
from datasets import load_from_disk
import numpy as np
from src.trainer.compressor_data_loader import CompressorDataLoader

def analyze_variance():
    print("Loading dataset...")
    # Initialize dataloader to reuse its logic
    dl = CompressorDataLoader(
        data_path="data/Amy-LM-Dataset", 
        batch_size=1, 
        segment_duration=1.0
    )
    dl.setup()
    
    # Get a few samples
    print("Fetching samples...")
    train_loader = dl.train_dataloader()
    
    avg_cosine_diff_llm = []
    avg_cosine_diff_wavlm = []
    
    for i, batch in enumerate(train_loader):
        if i >= 10: break
        
        # llm_feat: (B, T, D)
        llm = batch.llm_feat[0]
        wavlm = batch.wavlm_feat[0]
        
        # Calculate temporal smoothness: Cosine sim between t and t+1
        # Normalize first
        llm_norm = torch.nn.functional.normalize(llm, dim=-1)
        wavlm_norm = torch.nn.functional.normalize(wavlm, dim=-1)
        
        # Sim between adjacent frames
        llm_sim = torch.nn.functional.cosine_similarity(llm_norm[:-1], llm_norm[1:], dim=-1)
        wavlm_sim = torch.nn.functional.cosine_similarity(wavlm_norm[:-1], wavlm_norm[1:], dim=-1)
        
        avg_cosine_diff_llm.append(llm_sim.mean().item())
        avg_cosine_diff_wavlm.append(wavlm_sim.mean().item())

    print(f"\n--- Analysis (Avg Cosine Sim between adjacent frames) ---")
    print(f"LLM Feat Temporal Similarity: {np.mean(avg_cosine_diff_llm):.4f} (Higher = Smoother/Constant)")
    print(f"WavLM Feat Temporal Similarity: {np.mean(avg_cosine_diff_wavlm):.4f} (Lower = More changing)")
    
    if np.mean(avg_cosine_diff_llm) > np.mean(avg_cosine_diff_wavlm):
        print("\nCONCLUSION: LLM features change much slower than WavLM features.")
        print("This makes them easier to predict/distill, leading to lower loss.")

if __name__ == "__main__":
    analyze_variance()
