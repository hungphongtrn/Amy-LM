# Decision Log

## 2026-05-08: Plan Phase 1 Around MUStARD Dataset Foundation

**Context:** Issue #8 acceptance criteria begin with MUStARD extraction and FACodec preprocessing. The repository has generic FACodec preprocessing primitives but no MUStARD-specific parser or MOSS-Audio pilot runner.

**Decision:** Detail Phase 1 for MUStARD source parsing, FACodec row construction, and a preparation CLI before detailing model/training phases.

**Rationale:** Model and training work depends on the exact dataset schema and local availability of real FACodec outputs. Locking the data contract first reduces risk and gives later phases a stable collator target.

**Consequences:** Phase 2 through Phase 4 are intentionally stubbed and must be detailed only after Phase 1 lands.

## 2026-05-08: Use A MUStARD-Specific Preparation Path

**Context:** Existing `DatasetProcessor` loads Hugging Face datasets and currently does not preserve issue #8's binary `label`. MUStARD is sourced from a GitHub repository with its own metadata/audio layout, not necessarily from a HF dataset.

**Decision:** Add `scripts/prepare_mustard_dataset.py` and `src/preprocessing/mustard.py` instead of overloading the generic HF `DatasetProcessor` for source-repository extraction.

**Rationale:** A dedicated path keeps MUStARD label/audio resolution explicit while still reusing `FACodecEncoder` and stream dataclasses.

**Consequences:** Later phases should treat the MUStARD parquet/HF Dataset output as the canonical issue #8 training input.

## 2026-05-08: Preserve Content/Acoustic Fields As Optional Dataset Fields

**Context:** Issue #8 says Content and Acoustic streams may exist in the preprocessed dataset but are disabled for this experiment.

**Decision:** Phase 1 may persist Content and Acoustic stream fields alongside required Prosody/Timbre fields, but Phase 2 stream activation must exclude them from module construction and forward computation.

**Rationale:** FACodec already produces these fields during preprocessing, and keeping them can support later ablations without changing the current experiment contract.

**Consequences:** The training collator and model config must not accidentally enable Content/Acoustic for the main issue #8 row.
