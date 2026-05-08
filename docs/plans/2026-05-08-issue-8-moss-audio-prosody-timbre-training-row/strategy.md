# MOSS-Audio Prosody/Timbre Training Row - Strategy

## Goal

Implement the first Amy LM pilot training row for binary MUStARD sarcasm classification using frozen MOSS-Audio semantics plus precomputed FACodec Prosody Stream and Timbre Vector residual extensions.

## Architecture

The implementation has two offline/online boundaries. Offline preprocessing extracts MUStARD audio and labels, then runs FACodec once to persist Prosody Stream indices and Timbre Vector fields in a Hugging Face Dataset/parquet. Online training loads raw waveform batches, computes the Semantic Stream from frozen MOSS-Audio, aligns FACodec Prosody Stream embeddings to the actual MOSS-Audio frame count, broadcasts Timbre Vector projections, applies Residual Summation, and trains a binary classifier head.

The baseline and Amy LM row share the same MOSS-Audio semantic path, pooling, classifier protocol, data splits, loss, and metrics. The Amy LM row differs only by enabling Prosody/Timbre residual streams and trainable stream modules/gates.

## Tech Stack

- Python 3.11
- PyTorch
- Hugging Face `datasets`, `transformers`, and parquet output
- FACodec wrapper in `src/preprocessing/facodec_encoder.py`
- MOSS-Audio via `AutoConfig`/model loading with `trust_remote_code=True`
- wandb for metrics and lambda logging
- pytest for unit and contract tests

## Constraints & Assumptions

- `STATE.md` is absent in the repository checkout used for planning, despite `AGENTS.md` saying to read it first.
- Issue #8 is the source of truth; comments were intentionally not read per user request.
- Current code has FACodec preprocessing primitives, `ProsodyEmbedding`, `TimbreProjection`, `TemporalPool`, and `ResidualFusion`; it does not yet have MUStARD extraction, MOSS-Audio integration, or a pilot training runner.
- Phase 1 must preserve raw audio as HF `Audio` at 16 kHz and add binary `label`; the current generic `DatasetProcessor` persists FACodec streams but does not preserve labels.
- Prosody warm-start is required for the main Amy LM run. If FACodec codebook vectors cannot be extracted from the local checkpoint format, later phases must fail clearly or document an explicit fallback before any reported experiment.
- MOSS-Audio internals may require exploration because the issue requires extracting/freeze-using submodules rather than calling the generative forward path directly.
- Training can be long-running; when launched, use `nohup` and return the log path.

## Phases (High-Level)

### Phase 1: MUStARD Dataset Foundation

**Outcome:** A tested local command can clone/read MUStARD, construct binary sarcasm examples with raw 16 kHz audio, run FACodec preprocessing, and write HF Dataset/parquet rows matching issue #8.

**Rough scope:** Add a MUStARD-specific preprocessing path rather than forcing the generic HF dataset processor to understand the source repository layout. Extend reusable processing only where needed to preserve labels and exact Prosody Stream shape. Validate schema with tests using synthetic MUStARD-like files and mock FACodec.

### Phase 2: MOSS-Audio Internal Residual Extension

**Outcome:** `AmyForProsodyClassification` exposes binary logits from raw audio and optional FACodec Prosody/Timbre streams, with disabled streams excluded from module construction and computation.

**Rough scope:** Load MOSS-Audio config/model with `trust_remote_code=True`, identify the audio encoder/adapter and LLM `inputs_embeds` path, freeze MOSS-Audio parameters, and compose existing embedding/pooling/fusion modules around actual Semantic Stream frame counts.

**Depends on:** Phase 1

### Phase 3: Baseline And Amy Training

**Outcome:** The same train/eval protocol reports accuracy and F1 for frozen MOSS-Audio baseline and Amy LM Prosody/Timbre row.

**Rough scope:** Add collator, split handling, vanilla PyTorch training loop, wandb logging, checkpoint saving, and optional cross-validation aggregation. Log `lambda_p` and `lambda_t` per epoch for Amy LM.

**Depends on:** Phase 2

### Phase 4: Results And Artifacts

**Outcome:** Results are documented as hypothesis-supporting, neutral, or null, and any worthwhile artifacts are optionally pushed to HF Hub when credentials are available.

**Rough scope:** Add result records, command provenance, metrics tables, lambda interpretation, and optional HF Hub upload path for checkpoints/weights.

**Depends on:** Phase 3

## Open Questions

- Which local path should hold the MUStARD source checkout by default: `data/raw/MUStARD`, an explicit CLI path only, or a cached temporary directory?
- Does the local FACodec checkpoint expose prosody codebook vectors in a stable state-dict key usable for warm-start?
- Which MOSS-Audio remote-code submodule returns the exact LLM-input Semantic Stream required by issue #8?
- Should Phase 3 use k-fold cross-validation by default or a deterministic train/validation/test split if MUStARD lacks official folds in the source repository?
