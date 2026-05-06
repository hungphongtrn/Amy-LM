# Decision Log

## 2026-05-06: Grill session resolved FACodec stream contract

**Context:** Issue #12 was created because the codebase conflated FACodec residual acoustic VQ codes with the utterance-level timbre vector. The field `timbre_codebooks_idx` stored averaged `vq_id[3:]` values, and `TimbreEmbedding` performed a discrete lookup on those integer indices. The real timbre vector (`spk_embs`) was being ignored.

**Decision:** Published the corrected contract (see `strategy.md` Resolved Contract section):
- `vq_id[3:]` → **Acoustic Stream** (a_t), 3 codebooks, 80 Hz, int64
- `spk_embs` → **Timbre Vector** (h_t), utterance-level, float32
- `TimbreEmbedding` (discrete) → **TimbreProjection** (continuous linear)
- Dataset field `timbre_codebooks_idx` → `acoustic_codebooks_idx`

**Rationale:** Matches Amphion FACodec's documented API. The discrete embedding approach was a fundamental misunderstanding of FACodec's architecture.

**Consequences:**
- Issue #6 preprocessing must rename fields and add `timbre_vector`
- Issue #7 models must replace `TimbreEmbedding` with `TimbreProjection`
- Issue #8 is blocked until #6 and #7 corrections land
- `CONTEXT.md` updated with canonical terms

## 2026-05-06: Spike scope limited to documentation

**Context:** The acceptance criteria for issue #12 explicitly say "verify contract and publish corrected spec" — not to implement the corrections.

**Decision:** This spike produces a report, an optional inspector script, and documented specs. No renames, schema migrations, or module changes in this issue. Actual code changes belong to follow-up updates on issues #6 and #7.

**Rationale:** Keeps the spike fast and focused on evidence. Prevents unverified assumptions from entering production code before the FACodec contract is confirmed against real tensor shapes.

**Consequences:** Issue #8 remains blocked until #6/#7 are corrected in their respective issues.

## 2026-05-06: Phase 1 — Contract verified via local checkpoint run

**Context:** The inspector script (`scripts/inspect_facodec_contract.py`) ran successfully against a real Amphion FACodec checkpoint available locally.

**Decision:** All open questions about FACodec tensor shapes are now resolved:
- `vq_id` = `(6, 1, 80)` — 6 codebooks × 80 Hz, confirmed
- `spk_embs` = `(1, 256)` → `(256,)` float32 — D_timbre = 256, confirmed
- VQ slicing: prosody `(1,1,80)`, content `(2,1,80)`, acoustic `(3,1,80)`, confirmed

**Rationale:** The executable run confirms the Amphion README evidence with actual tensor shapes from the project's FACodec checkpoint.

**Consequences:**
- `D_timbre = 256` is now a hard constant in the corrected schema and module specs
- The contract moves from "provisional" to "confirmed"
- Phase 2 can proceed with full confidence in shape dimensions

## 2026-05-06: Phase 2 — Acoustic Stream preserves codebook axis

**Context:** The current `facodec_encoder.py` averages `vq_id[3]`, `vq_id[4]`, `vq_id[5]` into one integer per frame, destroying per-codebook structure. The Stream Dimensionality Contract requires each codebook to have its own embedding table, summed after lookup.

**Decision:** The corrected `acoustic_codebooks_idx` must store `[3, T80]` as a nested sequence preserving the codebook axis. No averaging.

**Rationale:** Averaging loses the independent codebook structure that the Stream Dimensionality Contract depends on. Each of the 3 acoustic codebooks gets its own embedding table; averaging them before storage makes it impossible to reconstruct.

**Consequences:**
- `facodec_encoder.py:367-369` averaging logic must be removed
- `acoustic_codebooks_idx` schema becomes `Sequence(Sequence(Value("int64")))` (nested, 2D)
- Dataset rows for existing processed data will need reprocessing or migration
- `AcousticEmbedding` must handle 3 separate embedding tables, not 1

## 2026-05-06: Phase 2 — Codebook axis preserved on all VQ streams

**Context:** The current preprocessing flattens multi-codebook VQ streams into scalar integers per frame. `content_codebooks_idx` (2 codebooks) is averaged, and `acoustic_codebooks_idx` (3 codebooks) was also averaged (and misnamed `timbre_codebooks_idx`). Only `prosody_codebooks_idx` (1 codebook) was correct but omitting the codebook axis.

**Decision:** All VQ streams preserve their codebook axis: prosody `[1, T80]`, content `[2, T80]`, acoustic `[3, T80]`. No averaging across codebooks. The embedding modules are responsible for summing per-codebook embeddings.

**Rationale:** The Stream Dimensionality Contract (each codebook gets its own independent embedding table) requires the codebook axis to be preserved through preprocessing. Averaging at storage time makes per-codebook embedding impossible.

**Consequences:** More complex HF Features nesting (`Sequence(Sequence(Value("int64")))`), but semantically correct per the FACodec architecture.

## 2026-05-06: Phase 2 — Issue #8 approved to proceed with conditions

**Context:** After Phase 2, all corrected specs are published. The question is whether issue #8 must wait for #6/#7 code changes, or can start in parallel.

**Decision:** Issue #8 is approved to proceed with conditions: #6 and #7 code must land before integration tests pass, but #8 module code can be written against the corrected interface specs now.

**Rationale:** The interface specs are precise enough (shapes, types, method signatures) that #8 can code against them in parallel. The actual dataset and module implementations won't match until #6/#7 land, so integration tests will fail — but that's expected parallel development.

**Consequences:**
- #6 and #7 should be prioritized (they unblock integration)
- #8 can begin implementation in parallel
- First experiment: Prosody only (λ_p enabled)
