# Decision Log

## 2026-05-05: Remove MOSS-Audio, use FACodec for all 3 codebooks

**Context:** Original plan used MOSS-Audio for semantic/content extraction and FACodec for prosody + timbre. This required temporal alignment (80 Hz → 12.5 Hz) between two separate encoders.

**Decision:** Use FACodec as the single encoder for content, prosody, and timbre codebooks. MOSS-Audio removed entirely.

**Rationale:**
- FACodec outputs all three codebooks in a single forward pass — simpler, faster
- No cross-encoder temporal alignment needed
- Same frame rate for all three codebooks (native FACodec output)
- During training, only prosody and timbre codebooks will be used, but content is extracted anyway since it's a single pass

## 2026-05-05: Store only codebook indices, not vectors

**Context:** Storing full codebook vectors (float32, 512-dim, 12.5 Hz) would consume ~38 GB for 100K utterances.

**Decision:** Only store integer indices per frame (list[int]). Vectors are reconstructed at training time from codebook lookup tables (2048 entries × 512-dim per codebook head).

**Rationale:**
- Indices are ~2500× smaller than vectors (~150 B vs ~385 KB per utterance)
- Codebook lookup tables are tiny (~4 MB per head) — can be bundled separately
- Reconstruction is a trivial embedding lookup, negligible overhead

## 2026-05-05: Generic dataset loader instead of MUStARD-specific downloader

**Context:** Original plan had `MustardDownloader` class tied to a specific HF repo.

**Decision:** Use `datasets.load_dataset()` with configurable dataset name. Pipeline is dataset-agnostic.

**Rationale:**
- Supports processing multiple HF datasets without code changes
- `dataset` column in output records the source repo for traceability
- Same CLI entry point handles any HF audio dataset

## 2026-05-05: HF Dataset as output format instead of .pt files

**Context:** Original plan saved individual `.pt` files per utterance.

**Decision:** Assemble output as a HuggingFace `Dataset` object (Apache Arrow/Parquet) and push to Hub.

**Rationale:**
- Memory-mapped by default — doesn't load entire dataset into RAM
- Works natively with HF training loops
- Single CLI command to push to Hub for reuse
- Standard format, versioned, shareable

## 2026-05-04: Restructured plan to follow progressive disclosure

**Context:** Original plan was a single 1958-line document with all 10 tasks fully detailed upfront.

**Decision:** Split into layered structure:
- `README.md` — Entry point with status
- `phase-01-preprocessing.md` — Detailed tasks for current phase
- `decisions.md` — This file
- `OLD-full-plan.md` — Archived original (now outdated)

**Rationale:**
- Implementer shouldn't need to read 50 tasks to understand the first 3
- Adaptability to reality as implementation reveals it

---

*Future decisions will be added here as the plan evolves.*
