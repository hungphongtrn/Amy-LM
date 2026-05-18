# Decision Log

## 2026-05-18: Phase ordering and parallelism

**Context:** Issues #17, #18, #19 are all unblocked.
**Decision:** #17 and #18 can run truly parallel (no shared code, no shared tests). #19 depends on understanding the embedding module changes (warm_start projector unfreeze) — but does NOT depend on #17 or #18 code. Recommended: run #17 + #18 in parallel, then #19.
**Rationale:** #17 is trivial (~50 lines); #18 is data plumbing; #19 is the complex model change. Doing #17/#18 first confirms the test infrastructure is working before attempting #19.
**Consequences:** If #19 needs to import from #17 or #18, no coupling — they are independent.

## 2026-05-18: NV tag text format

**Context:** What canonical text should represent each NV type?
**Decision:** `[Breathing]`, `[Laughter]`, `[Sigh]`, `[Sneeze]`, `[Cough]`, `[Throat clear]`, `[Groan]`, `[Grunt]`, `[Snore]`, `[Sniff]`
**Rationale:** These match the NVTTS paper terminology and are human-readable for LLM prompts. The format `[Tag]` follows bracket-convention used in diarization and ASR tagging.
**Consequences:** DeepSeek prompt templates will use these bracket-wrapped tags. The transform function strips the emoji prefix and wraps in brackets.

## 2026-05-18: Speaker cache cascading fallback priority

**Context:** Three VoxCeleb metadata sources have overlapping speaker coverage.
**Decision:** enrichment (hechmik) → vox1_meta (ProgramComputer) → language-metadata (johbac) → Expresso hardcoded
**Rationale:** enrichment has the richest fields (age, birth year) cross-validated across KG/DBpedia/Wikidata. vox1_meta is the canonical source. language-metadata adds names for VoxCeleb2 from non-vox1-meta coverage. Expresso has no upstream source — hardcoded.
**Consequences:** A speaker may get age from enrichment but gender from vox1_meta if enrichment is missing gender. Each field is resolved independently through the cascade.

## 2026-05-18: AmyLM trainable/frozen split

**Context:** Which modules are trainable in the DPO phase?
**Decision:** Trainable: ProsodyEmbedding (including warm_start projector), TimbreProjection, ResidualFusion (λ gates). Frozen: audio_encoder, audio_adapter, Qwen3 language_model.
**Rationale:** The frozen backbone preserves MOSS-Audio's pretrained representations. Only the new FACodec enrichment pathway is trained. The warm_start projector must be trainable because its static random-init projector would pass no useful gradient during DPO — defeating the purpose of training.
**Consequences:** `src/models/embedding.py:46-48` must change from `requires_grad = False` to `requires_grad = True`. The test `test_warm_start_projector_is_frozen` must be updated.

## 2026-05-18: No content/acoustic streams in AmyLM

**Context:** The Stream Activation Config allows per-stream enabling.
**Decision:** AmyLM only instantiates prosody + timbre modules. No content or acoustic streams.
**Rationale:** The initial experiment isolates prosody as the social-prosody signal and timbre as the speaker-identity signal. Content/acoustic are left for ablation studies later.
**Consequences:** AmyLM does not need `AcousticEmbedding` or `ContentEmbedding` modules. If Stream Activation Config later enables them, they would be added then — not now.

## 2026-05-18: AmyLM forward() enrichment location

**Context:** Where in `MossAudioModel.forward()` should FACodec enrichment happen?
**Decision:** Between `audio_adapter(audio_embeds)` and `inputs_embeds.masked_scatter_(...)`. Specifically: after `audio_embeds = self.audio_adapter(audio_embeds)` (line 470), enrich with prosody/timbre, then proceed to the audio token count check and `masked_scatter_`.
**Rationale:** The audio embeddings at this point are in LLM embedding space (2560 dim), matching the output of `ProsodyEmbedding` and `TimbreProjection`. Enrichment after the adapter but before scattering ensures FACodec streams add to the right token positions.
**Consequences:** `AmyLM.forward()` overrides the parent's forward method entirely (copy the base logic, add enrichment step). This means we cannot `super().forward()` — we must reimplement the full forward path.
