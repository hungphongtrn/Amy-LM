# Amy LM Pilot Validation - Strategy

## Goal

Validate the "Extension Architecture" hypothesis: Adding FACodec prosody/timbre embeddings to MOSS-Audio via residual summation improves prosody understanding without degrading semantic performance. Success = λ gate grows from zero, indicating the model learns to use the FACodec signal.

## Architecture

The Amy LM Extension Architecture adds discrete prosody codes and global timbre vectors to MOSS-Audio's semantic stream at the LLM input layer (analogous to positional encodings).

**Fusion formula:** `H_t = MOSS-Audio(S_t) + λ · (E_prosody[p_t^pooled] + h_timbre)`

Where:
- FACodec prosody: 80 Hz → pooled to 12.5 Hz via average pooling
- Timbre: 256-dim global utterance vector → projected to 2560-dim
- λ: Zero-initialized learnable gate

## Tech Stack

| Component | Source |
|-----------|--------|
| Semantic backbone | MOSS-Audio-4B-Instruct |
| Prosody/Timbre | FACodec (Amphion checkpoint) |
| Primary dataset | MUStARD++ (~1K sarcasm clips) |
| Benchmark suite | MUStARD++, MCR-Bench, MSP-Podcast, MSPB, SpeechWellness, SUPERB-prosody, NonVerbalSpeech-38K, Vox-Profile |

## Hypothesis Matrix

Three dimensions forming a 3×3×3 experimental matrix (progressive execution, stops at diminishing returns):

1. **Embedding Init**: Random | FACodec warm-start | Continuous projector
2. **Training Strategy**: Frozen MOSS-Audio | LoRA (rank 64) | Full fine-tune
3. **Loss Objective**: Classification head | LM loss | Combined

## Constraints & Assumptions

- FACodec encoder runs offline during preprocessing (not in training loop)
- Training starts with λ = 0 (model = MOSS-Audio at step 0)
- Pilot scope: If λ stays near zero, hypothesis is falsified — no further matrix rows needed
- Compute constraint: Experiments stop at diminishing returns, not exhaustive completion
- Hardware: 12GB GPU assumed; larger infrastructure deferred

## Phases (High-Level)

### Phase 1: MUStARD++ Preprocessing - Foundation
**Outcome:** Preprocessed `.pt` files ready for training with aligned (semantic, prosody, timbre, label) tuples.
**Rough scope:**
- Download MUStARD++ from HuggingFace
- Run FACodec offline to extract prosody indices (80 Hz) + timbre vectors
- Run MOSS-Audio encoder offline to extract semantic frames (12.5 Hz, 2560-dim)
- Temporal pooling alignment (80 Hz → 12.5 Hz)
- Save per-utterance `.pt` files with all features and metadata

**Depends on:** None (entry point)

### Phase 2: Embedding Tables & Residual Fusion - Core Feature
**Outcome:** Trainable embedding tables with warm-start initialization, residual fusion module with λ gate, end-to-end forward pass working on preprocessed data.
**Rough scope:**
- Embedding lookup tables for FACodec prosody (1024 vocab → 2560-dim)
- Three init strategies: random, FACodec warm-start via projection, continuous projector
- Temporal pooling module (swappable strategies)
- Residual fusion: summation + λ gate + LayerNorm
- Integration tests: λ=0 produces identity, known inputs produce expected outputs

**Depends on:** Phase 1 completion

### Phase 3: Training Loop & Hypothesis Matrix - Validation
**Outcome:** MUStARD++ training with observable λ growth, hypothesis matrix execution from simplest to complex, benchmark evaluation across 8 datasets.
**Rough scope:**
- Training loop plugging fusion into MOSS-Audio forward pass
- Three loss modes: classification head, LM loss, combined
- Three training modes: frozen, LoRA, full fine-tune
- Hypothesis matrix runner (3×3×3 with early stopping)
- Benchmark evaluator for all 8 datasets
- Metrics: in-domain (MUStARD++) + out-of-domain (generalization)

**Depends on:** Phase 2 completion

## Open Questions

1. Will λ grow from zero, or does the hypothesis need revision?
2. Which embedding init strategy converges fastest?
3. Is LoRA sufficient, or is full fine-tune required?
4. Does sarcasm training on MUStARD++ generalize to broader prosody tasks?
5. At what point do returns diminish in the hypothesis matrix?

---

*This strategy follows progressive disclosure. See phase documents for implementation detail. Do NOT read ahead — Phase 2 and 3 will be detailed after Phase 1 completes and learnings are incorporated.*
