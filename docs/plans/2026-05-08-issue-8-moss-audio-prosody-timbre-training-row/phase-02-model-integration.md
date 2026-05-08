# Phase 2: MOSS-Audio Internal Residual Extension

## Phase Goal

Build `AmyForProsodyClassification` so raw audio produces frozen MOSS-Audio Semantic Stream frames, optional FACodec Prosody/Timbre streams are fused at the MOSS-Audio LLM input, and the model returns binary logits.

## Current Status

Stub only. Detail this phase after Phase 1 confirms the exact dataset schema and any FACodec checkpoint/codebook-vector constraints.

## Expected Scope

- Add a MOSS-Audio loading adapter using `AutoConfig.from_pretrained("OpenMOSS-Team/MOSS-Audio-4B-Thinking", trust_remote_code=True)` and corresponding model loading with `trust_remote_code=True`.
- Identify and test the audio encoder/adapter path that yields `S_t [B, T_moss, 2560]` without invoking the generative forward path directly.
- Add a stream activation config that constructs only Prosody/Timbre modules for issue #8.
- Load FACodec prosody codebook vectors and initialize `ProsodyEmbedding` rows through a one-time projection to 2560; fail clearly if vectors are unavailable.
- Use `TimbreProjection(256 -> 2560)`, broadcast to MOSS-Audio frame count, and pass pre-broadcast tensors into `ResidualFusion`.
- Mean-pool the frame axis and apply `nn.Linear(2560, 2)`.

## Completion Criteria

- Unit tests cover disabled stream exclusion, lambda zero initialization, output logits shape, and trainable/frozen parameter partitioning.
- A smoke forward pass works with synthetic FACodec fields and either a tiny mocked MOSS-Audio adapter or a real local MOSS-Audio load when feasible.
