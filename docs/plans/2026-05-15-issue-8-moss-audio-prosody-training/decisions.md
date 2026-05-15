# Decision Log

## 2026-05-15: Plan structure — 4 phases
**Context:** Issue #8 has 4 major work areas: MOSS-Audio loading, model assembly, data pipeline, training loop.
**Decision:** 4 phases, each producing one independently testable artifact.
**Rationale:** Each phase has its own verification criteria (can test MOSS-Audio forward pass without data pipeline; can test model assembly without training loop).
**Consequences:** Phase 4 (training) is gated on Phases 2+3 both being complete. Phase 3 can run in parallel with Phase 2 if needed.

## 2026-05-15: Vanilla PyTorch training loop (no Lightning, no HF Trainer)
**Context:** CONTEXT.md and issue #8 body both specify "Vanilla PyTorch training loop, no Lightning and no HF Trainer."
**Decision:** Write a ~50-line training loop in a standalone script.
**Rationale:** No GAN dual-optimizer complexity to justify Lightning, avoids Lightning's memory overhead with 4B backbone, easier to debug.
**Consequences:** Must manually handle device placement, gradient accumulation, logging. W&B must be called explicitly.

## 2026-05-15: MOSS-Audio sub-module extraction (not calling generative forward())
**Context:** CONTEXT.md specifies: "We extract sub-modules (audio_encoder, audio_adapter, language_model) from the loaded MossAudioModel rather than calling its generative forward()."
**Decision:** Create `MossAudioWrapper` that holds `audio_encoder`, `audio_adapter`, `language_model` as separate frozen attributes.
**Rationale:** We need frame-level hidden states to fuse with FACodec embeddings before the LLM, not the final text output. The generative forward path includes the LM head and token decoding, which we don't need.
**Consequences:** Must understand MossAudioModel internal structure. If sub-module names differ across model versions, the wrapper must detect and map them.

## 2026-05-15: FACodec warm-start for ProsodyEmbedding
**Context:** Issue #8 specifies FACodec codebook vectors projected to D=2560 as initial ProsodyEmbedding weights.
**Decision:** Use the existing `init="warm_start"` path in `ProsodyEmbedding`, which loads codebook vectors and applies a frozen or trainable `nn.Linear(D_codebook, 2560, bias=False)` projector.
**Rationale:** Existing code already implements this; just needs the actual FACodec codebook vectors extracted from the Amphion checkpoint.
**Consequences:** Must locate the prosody codebook weight tensor in the FACodecDecoder state dict. The key name might vary; need to verify against actual checkpoint.

## 2026-05-15: Stream activation config via YAML
**Context:** Issue #8 specifies `facodec_streams: {prosody: true, content: false, acoustic: false, timbre: true}`. CONTEXT.md says disabled streams are excluded from both module instantiation and forward computation.
**Decision:** The `AmyForProsodyClassification` constructor takes a stream config dict and only creates modules for active streams. Forward pass only fuses active streams.
**Rationale:** Prevents wasted memory/compute, clear semantics — a disabled stream isn't "zeroed out," it literally doesn't exist in the computation graph.
**Consequences:** Model state dict will differ based on active streams. Checkpoint compatibility depends on matching config.

## 2026-05-15: Online Semantic Encoding
**Context:** CONTEXT.md specifies issue #8 uses MOSS-Audio to compute the Semantic Stream per batch, not precomputed to disk.
**Decision:** Training batch loads only audio waveform + FACodec indices + labels. MOSS-Audio encoder runs during forward pass to produce S_t.
**Rationale:** Semantic frames are large ([T, 2560] per sample), and MOSS-Audio encoding is considered fast enough. Keeps the data pipeline simple.
**Consequences:** Forward pass includes the audio encoder computation. Gradient checkpointing on the encoder may be needed for memory.
