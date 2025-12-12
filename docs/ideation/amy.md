# Mimi Modification

## **Architecture Overview**

We modify the standard Mimi architecture to support **9 Codebooks** (1 Semantic + 1 Prosody + 7 Acoustic).

## **1. The "Surgery" Strategy (Initialization)**

To avoid training from scratch, we transplant weights from the official Kyutai Mimi checkpoint (8-codebook version).

| New Layer Index | Role | Initialization Source |
| :--- | :--- | :--- |
| **Head 0** | **Semantic (Content)** | **Random / Fresh Init** (Targeting Qwen Space) |
| **Head 1** | **Prosody (Intonation)** | **Mimi Original CB 0** (Semantic/Coarse) |
| **Head 2** | Acoustic Residual 1 | Mimi Original CB 1 |
| **Head 3** | Acoustic Residual 2 | Mimi Original CB 2 |
| ... | ... | ... |
| **Head 8** | Acoustic Residual 7 | Mimi Original CB 7 |

## **2. The `CausalBridgeQuantizer` Class**

```python
import torch
import torch.nn as nn
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

class CausalBridgeQuantizer(nn.Module):
    def __init__(self, codec_dim=512, qwen_dim=1536, wavlm_dim=1024):
        super().__init__()
        
        # --- The Split Heads ---
        # Head 0: Strictly Semantic (Context)
        self.head_content = VectorQuantize(dim=codec_dim, codebook_size=2048)
        
        # Head 1: Strictly Prosodic (Pitch/Rhythm)
        self.head_prosody = VectorQuantize(dim=codec_dim, codebook_size=2048)
        
        # Heads 2-8: Acoustic Texture (Timbre/Noise)
        # We use 7 layers here to reach total 9 codebooks
        self.acoustic_residual = ResidualVQ(
            dim=codec_dim, 
            num_quantizers=7, 
            codebook_size=2048
        )
        
        # --- The Bridges (Projections) ---
        # Maps Codec Space -> Teacher Space
        self.proj_qwen = nn.Linear(codec_dim, qwen_dim)   # 512 -> 1536
        self.proj_wavlm = nn.Linear(codec_dim, wavlm_dim) # 512 -> 1024

    def forward(self, x, qwen_hidden=None, durations=None):
        """
        x: [Batch, Frames, 512]
        qwen_hidden: [Batch, Tokens, 1536] (Training Only)
        durations: [Batch, Tokens] (Training Only)
        """
        
        # 1. Quantize Content
        q_c, idx_c, _ = self.head_content(x)
        
        # 2. Quantize Prosody (Residual of Content)
        resid_1 = x - q_c.detach()
        q_p, idx_p, _ = self.head_prosody(resid_1)
        
        # 3. Quantize Acoustics (Residual of Prosody)
        resid_2 = x - (q_c + q_p)
        q_a, idx_a, _ = self.acoustic_residual(resid_2)
        
        # Reconstruct total quantized representation
        q_total = q_c + q_p + q_a
        
        # 4. Distillation Losses (The "Bridge")
        distill_loss = 0.0
        
        if self.training and qwen_hidden is not None:
            # Expand Qwen States to Audio Frame Rate (12.5 Hz)
            # Logic: Repeat hidden state 'N' times based on duration of token
            aligned_qwen = torch.repeat_interleave(qwen_hidden, durations, dim=1)
            
            # Loss A: Content Head must match Qwen Hidden State
            pred_qwen = self.proj_qwen(q_c)
            distill_loss += nn.MSELoss()(pred_qwen, aligned_qwen)
            
            # Loss B: Content+Prosody must match WavLM Features
            # (Assuming wavlm_target is passed in context or computed)
            # pred_wavlm = self.proj_wavlm(q_c + q_p)
            # distill_loss += nn.MSELoss()(pred_wavlm, wavlm_target)
            
        return q_total, torch.cat([idx_c, idx_p, idx_a], dim=1), distill_loss
```

## **3. Training Considerations**

* **Optimizer:** AdamW (`beta1=0.8`, `beta2=0.99`).
* **Learning Rate:** `1e-4` for Heads 0/1; `1e-5` for pre-trained Heads 2-8.
* **Gradient Accumulation:** Required due to 12.5Hz sequence length (sequences are long in time, though short in tokens).
