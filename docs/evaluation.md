┌────────────────────────────────────────────────────────────────────┐
│                     Amy Validation Pipeline                        │
├────────────────────────────────────────────────────────────────────┤
│  Stage 1: Intrinsic (Cheap, Fast)                                  │
│    ├── Probing Classifiers (Phoneme vs Pitch on each head)         │
│    ├── Reconstruction Ablation (SI-SDR per codebook)               │
│    └── Semantic Similarity (same text, different prosody)          │
├────────────────────────────────────────────────────────────────────┤
│  Stage 2: Cross-Modal (Medium Cost)                                │
│    ├── Audio↔Text Retrieval (Head 0 vs Qwen embeddings)            │
│    └── Emotion Classification (Head 1 only)                        │
├────────────────────────────────────────────────────────────────────┤
│  Stage 3: Qualitative (Compelling Demos)                           │
│    └── Prosody Swap Reconstruction                                 │
├────────────────────────────────────────────────────────────────────┤
│  Stage 4: Proxy LLM (Before Full Commitment)                       │
│    └── Small Transformer on Prosody Tokens → Conversational Tasks  │
└────────────────────────────────────────────────────────────────────┘