# Amy-LM

Speech Language Model that understands prosody and timbre natively, solving **social deafness** — the failure to process emotional tone and subtext in spoken language.

## Motivation

Current speech-to-text models (Whisper, MOSS-Audio) use a **Projection Architecture**: audio features are compressed through a text-aligned projector (e.g., GatedMLP) into an LLM's embedding space. This is fundamentally lossy for prosody — pitch, rhythm, intonation, and speaker identity are discarded during the forced alignment to semantic tokens. The result: models that transcribe words correctly but respond as if every utterance is emotionally neutral, missing sarcasm, distress, hesitation, and other paralinguistic cues.

## Hypothesis

**Extension Architecture**: Instead of projecting audio into a text-aligned space, add *parallel embedding dimensions* that exist explicitly for prosody and timbre. Using FACodec's factorized discrete codebooks, we derive a **Prosody Stream** (pitch/rhythm/intonation indices at 80 Hz), a **Timbre Vector** (utterance-level speaker embedding), and optionally Content/Acoustic streams. These are fused with MOSS-Audio's **Semantic Stream** via **Residual Summation** at the LLM input — think of it like **Position Embeddings**: just as position shifts where a token sits in embedding space, prosody and timbre rotate/shift the semantic vector to encode *how* something was said and *who* said it. The formulation is `LayerNorm(S_t + λ·P_t + λ·T_t)`, with zero-initialized learnable λ gates so the model starts identical to MOSS-Audio at step 0.

If this holds, the same architecture generalizes from classification (sarcasm detection) to generation (context-aware response preference).

## Current Experiments

### Experiment 1 — Classification Probe (Issue #8)

**Goal**: Validate that FACodec prosody + timbre improves binary sarcasm classification over MOSS-Audio alone.

| Component | Detail |
|-----------|--------|
| Model | `AmyForProsodyClassification` — frozen MOSS-Audio backbone + FACodec streams + `Linear(2560→2)` head |
| Data | MUStARD (binary sarcasm) preprocessed via FACodec encoder |
| Streams | Prosody (warm-started from FACodec codebook), Timbre (projected `spk_embs`) |
| Baseline | Frozen MOSS-Audio + `Linear(2560→2)`, no FACodec streams |
| Status | Ready for agent — implementation can begin |

### Experiment 2 — Generative DPO (Issue #14)

**Goal**: Train AmyLM to prefer context-aware responses over literal ones, using prosody/timbre as the sole disambiguation signal.

**Phase 1 — Preference Pair Pipeline** (complete):
1. **NV Tag Emoji Mapping** (#17) — NVTTS emoji tags → `[Tag]` text labels
2. **Speaker Context Lookup** (#18) — VoxCeleb/Expresso enrichment tables
3. **AmyLM Model + Config** (#19) — HF-compatible `AmyLM` with `save_pretrained`/`from_pretrained`/`generate`
4. **Enrich NVTTS** (#20) — Build enriched dataframe with speaker context
5. **DeepSeek Pair Gen** (#22) — LLM generates chosen/rejected responses
6. **Cosine Scorer** (#23) — Embed + filter pairs by cosine similarity

**Phase 2 — Training Infrastructure** (in progress):
- **PreferenceDatasetProcessor** (#21) — FACodec encoding for preference pairs
- **DPO Collator + Training Script** (#24) — `AmyDPOTrainer` subclass with frozen backbone
- **LLM-as-Judge** (#25) — Pre/post training comparison on 50 hardest held-out samples

### Preprocessing Pipeline

All experiments share the same FACodec encoding pipeline:

```bash
# Process a dataset through FACodec → parquet with codebook indices
uv run python scripts/preprocess.py \
    --dataset huggingface/dataset-name \
    --split train \
    --output-repo org/processed-dataset
```

Output schema includes `prosody_codebooks_idx` `[1, T80]`, `content_codebooks_idx` `[2, T80]`, `acoustic_codebooks_idx` `[3, T80]`, and `timbre_vector` `[256]`.

## Installation

```bash
pip install uv
uv sync
```

## Project Structure

```
Amy-LM/ (branch: exp/amylm-facodec)
├── CONTEXT.md           # Domain glossary (start here)
├── AGENTS.md            # Agent instructions
├── docs/
│   ├── ideation/        # Architecture docs and proposals
│   └── training_records/# Training logs
├── src/
│   ├── models/          # AmyForProsodyClassification, AmyLM, fusion, pooling
│   ├── preprocessing/   # FACodec encoder, dataset processors, reporting
│   ├── data/            # Feature extraction utilities
│   ├── training/        # Training loops and trainers
│   └── inference/       # Generation scripts
├── scripts/             # Preprocessing, training, inference entry points
└── tests/               # Unit and integration tests (108)
```
