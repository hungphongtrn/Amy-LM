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

## 2026-05-15: Phase 1 complete — MOSS-Audio vendor approach
**Context:** `AutoModel.from_pretrained()` failed because the MOSS-Audio HF repo lacks `modeling_moss_audio.py`. The model class lives in the GitHub source repo (`OpenMOSS/MOSS-Audio`).
**Decision:** Vendored the entire MOSS-Audio source repo to `vendor/MOSS-Audio/` (matching the existing `vendor/Amphion/` pattern). Added `vendor/MOSS-Audio/src/` to `sys.path` for local imports.
**Rationale:** `MossAudioModel.from_pretrained()` with `trust_remote_code=True` requires the full model class. The GitHub repo provides it; vendoring keeps the dependency self-contained.
**Consequences:** Patched one vendored import: `from src.configuration_moss_audio` → `from configuration_moss_audio` in `vendor/MOSS-Audio/src/modeling_moss_audio.py`. All MOSS-Audio source files tracked in git despite `.gitignore` (via force-add, same as Amphion).

## 2026-05-15: Phase 1 complete — MOSS-Audio frame rate and encoding
**Context:** Assumed ~12.5 Hz frame rate from Whisper 200x downsample. Actual MOSS-Audio encoder pipeline differs.
**Decision:** Confirmed actual path: 16kHz audio → mel spectrogram (hop=160, ~100 Hz) → conv downsample (/8) → ~12.5 Hz. 2s audio → ~25 frames (not ~250 as originally estimated).
**Rationale:** Verified empirically through tests. The 25-frame count for 2s audio is now the ground truth encoded in test assertions.
**Consequences:** `TemporalPool` downsampling from FACodec 80 Hz to MOSS-Audio's actual 12.5 Hz will use `target_len = MOSS_output_frames` (not a hardcoded constant). Phase 2 must use the actual frame count from `encode_semantic()`.

## 2026-05-15: Phase 1 complete — GatedMLP adapter structure
**Context:** Assumed `audio_adapter` would have `.in_features` and `.out_features` attributes like a plain Linear. MOSS-Audio uses a `GatedMLP` structure.
**Decision:** Access input dim via `wrapper.audio_adapter.gate_proj.in_features` and output dim via `wrapper.audio_adapter.down_proj.out_features` in tests. The wrapper API (`encode_semantic()`) abstracts this entirely.
**Rationale:** These are test-only internals; the public API is unaffected. Adapter output dim (2560) matches Qwen3 hidden_size as required.
**Consequences:** If MOSS-Audio updates its GatedMLP structure, the structural tests will need updating. The wrapper's `encode_semantic()` will continue working as long as `audio_encoder()` and `audio_adapter()` accept the same call signatures.

## 2026-05-15: Phase 2 detailed — Codebook extraction as standalone utility
**Context:** The `ProsodyEmbedding` warm-start path needs raw FACodec codebook vectors `[1024, 8]` from `quantizer.0.layers.0._codebook.weight` in `ns3_facodec_decoder.bin`.
**Decision:** Create `src/models/codebook_utils.py` with `load_prosody_codebook_vectors()` as a standalone utility, not embedded inside `AmyForProsodyClassification`.
**Rationale:** Codebook extraction is a one-time initialization step. Separating it keeps the model constructor clean (accepts `warm_start_vectors: Tensor`) and makes the utility reusable across different model variants and scripts.
**Consequences:** The model constructor requires pre-loaded vectors; the caller is responsible for providing them. This matches issue #8's spec: "FACodec encoder is not loaded during training."

## 2026-05-15: Phase 2 detailed — TemporalPool auto-alignment (no target_len override)
**Context:** Need pool output frames to equal MOSS-Audio's T_moss. `TemporalPool` computes `target_len = round(duration_sec * output_rate)` internally.
**Decision:** Use `TemporalPool` as-is without adding a `target_len` parameter. The math works: `round(T80 * 12.5 / 80)` always equals the actual MOSS-Audio frame count for the same audio duration.
**Rationale:** Both FACodec (80 Hz) and MOSS-Audio (~12.5 Hz) use deterministic conv-based downsampling from the same 16kHz input. The ratio is consistent. Adding a target_len override introduces an untested code path with no real benefit.
**Consequences:** Phase 2 includes a temporal alignment test that asserts `P.shape[1] == T_moss` to catch any edge case early. If an edge case is found later, we can add alignment padding/trimming in the model forward.

## 2026-05-15: Phase 2 detailed — Timbre broadcast in model forward, not in fusion
**Context:** `ResidualFusion.forward()` expects `timbre: [B, T, D]` (pre-broadcast). `TimbreProjection.forward()` returns utterance-level `[B, 2560]`.
**Decision:** Handle the broadcast from `[B, 2560]` → `[B, T_moss, 2560]` inside `AmyForProsodyClassification.forward()`, not inside fusion or projection.
**Rationale:** Fusion is a general-purpose module that shouldn't know about utterance-vs-frame semantics. Projection transforms the vector. The model orchestrator (Amy) owns the spatial broadcasting logic.
**Consequences:** Broadcasting uses `torch.unsqueeze(1).expand(-1, T_moss, -1)`. This is correct because timbre is per-utterance — same vector repeats across all frames.

## 2026-05-15: Phase 2 detailed — Language model called with inputs_embeds, input_ids=None
**Context:** MOSS-Audio's own forward path calls `self.language_model(input_ids=None, attention_mask=..., inputs_embeds=inputs_embeds, ...)`. We need frame-level hidden states, not text generation.
**Decision:** Call `language_model(inputs_embeds=H)` without `input_ids` or `attention_mask`. Extract `last_hidden_state` for pooling.
**Rationale:** Qwen3 handles `input_ids=None` + `inputs_embeds` natively. No text tokens are involved. Mean-pooling over the frame dimension produces the utterance representation for the classifier. The default attention mask behavior is acceptable for this pilot — later experiments can tune mask strategies.
**Consequences:** If Qwen3 defaults to causal attention, each output frame sees itself and earlier frames. This is acceptable for a classification pilot. Bidirectional attention can be explored later.

## 2026-05-15: Phase 2 complete — dtype bridging and frame alignment
**Context:** MOSS-Audio produces `bfloat16` semantic embeddings; FACodec modules produce `float32`. The `language_model` has `bfloat16` weights. Pooled prosody frames can differ from T_moss (e.g., 1s audio: MOSS gives 13 frames, pool computes `round(12.5)=12`).
**Decision:** In `AmyForProsodyClassification.forward()`: convert semantic to `float32` for fusion, then convert fused `H` to `lm_dtype` before LM forward, convert LM output back to `float32` for classifier. Add an explicit `F.adaptive_avg_pool1d` alignment step when `P.shape[1] != T_moss`.
**Rationale:** Fusion and FACodec modules operate in float32 (training precision). Language model expects its native dtype. The classifier is float32. Frame alignment via adaptive pooling is robust to edge cases without modifying `TemporalPool`.
**Consequences:** The dtype bridge adds `.float()` and `.to(dtype=lm_dtype)` calls. These are cheap (tensor metadata ops) and handled in no-grad context where possible.