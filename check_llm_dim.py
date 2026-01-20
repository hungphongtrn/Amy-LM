from datasets import load_from_disk
import numpy as np

ds = load_from_disk("data/Amy-LM-Dataset")
item = ds[0]
llm_feat = np.array(item["llm_feat"])
print(f"LLM Feature Shape: {llm_feat.shape}")
print(f"D_llm: {llm_feat.shape[1]}")
