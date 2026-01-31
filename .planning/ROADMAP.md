# Roadmap: Proactive-SAT

## Overview

Proactive-SAT ships an MVP benchmark that isolates prosody as the signal for hidden intent: neutralized transcripts + controlled TTS variants + three evaluation tracks + comparable outputs (JSON + charts).

```mermaid
flowchart LR
  A[.data/ source] --> B[Phase 1: Dataset (neutral text + speaker instructions)]
  B --> C[Phase 2: TTS (control + trigger audio)]
  C --> D[Phase 3: Evaluation (3 tracks, 2 tasks)]
  D --> E[Phase 4: Results (JSON + visualizations)]
```

## Phases

**Phase Numbering:**
- Integer phases (1, 2, 3): Planned milestone work
- Decimal phases (2.1, 2.2): Urgent insertions (marked with INSERTED)

- [x] **Phase 1: Data Pipeline** - Turn `.data/` into a 200-sample HF dataset with neutralized text and prosody instructions (Completed: 2026-01-31)
- [ ] **Phase 2: Speech Synthesis** - Generate Control vs Trigger audio variants for the dataset using Qwen3-TTS
- [ ] **Phase 3: Benchmark Evaluation** - Run the three model tracks on both benchmark tasks and capture per-sample outputs
- [ ] **Phase 4: Results & Visualization** - Export benchmark results JSON and produce comparison charts

## Phase Details

### Phase 1: Data Pipeline
**Goal**: Users can produce a 200-sample dataset where the text is neutral but the intended meaning can be expressed via prosody instructions
**Depends on**: Nothing (first phase)
**Requirements**: DATA-01, DATA-02, DATA-03, DATA-04
**Success Criteria** (what must be TRUE):
  1. User can run a single command to load/parse `.data/` into a structured sample table without manual editing
  2. User can inspect any sample and see both original text and a lexical-neutralized transcript
  3. User can inspect any sample and see generated speaker instructions for prosodic injection (e.g., sarcastic/frustrated/distressed)
  4. User can load a HuggingFace dataset with exactly 200 samples that includes Control/Trigger variant metadata
**Plans**: 3 plans

Plans:
- [x] 01-01: Parse `.data/` into canonical sample schema (Completed: 2026-01-31)
- [x] 01-02: Lexical neutralizer + speaker-instruction generation (Completed: 2026-01-31)
- [x] 01-03: Build/export HF dataset (200 samples; control/trigger fields) (Completed: 2026-01-31)

### Phase 2: Speech Synthesis
**Goal**: Users can generate paired Control and Trigger audio for every dataset item
**Depends on**: Phase 1
**Requirements**: TTS-01, TTS-02, TTS-03
**Success Criteria** (what must be TRUE):
  1. User can generate speech audio for dataset samples using Qwen3-TTS from the local repo
  2. User can batch-generate Control variants (flat/factual delivery) for the full dataset with no missing files
  3. User can batch-generate Trigger variants (prosodically rich delivery) for the full dataset with no missing files
  4. User can play back a random pair (Control vs Trigger) and observe an audible style difference consistent with the instructions
**Plans**: 3 plans

Plans:
- [ ] 02-01: Integrate Qwen3-TTS + synthesis runner
- [ ] 02-02: Generate Control audio batch
- [ ] 02-03: Generate Trigger audio batch

### Phase 3: Benchmark Evaluation
**Goal**: Users can run all benchmark tracks end-to-end and obtain per-sample predictions for both tasks
**Depends on**: Phase 2
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05
**Success Criteria** (what must be TRUE):
  1. User can run the Text-only track (GPT-5) on neutral transcripts and obtain intent + proactive-trigger predictions
  2. User can run the ASR+LLM track (Whisper -> GPT-5) on audio and obtain intent + proactive-trigger predictions
  3. User can run the End-to-End SLM track (Gemini 3 Flash) on raw audio and obtain intent + proactive-trigger predictions
  4. User can run a single benchmark command that executes all tracks and writes per-sample outputs plus aggregate metrics to disk
**Plans**: 3 plans

Plans:
- [ ] 03-01: Implement the three evaluation tracks (text-only, ASR+LLM, E2E audio)
- [ ] 03-02: Implement both tasks + orchestration runner
- [ ] 03-03: Metrics aggregation + standardized run artifacts

### Phase 4: Results & Visualization
**Goal**: Users can compare tracks via a portable results artifact and clear visual summaries
**Depends on**: Phase 3
**Requirements**: RSLT-01, RSLT-02
**Success Criteria** (what must be TRUE):
  1. User can produce a single JSON benchmark results file that captures metrics and track-by-track comparisons
  2. User can generate bar-chart comparisons for key metrics across tracks from the JSON results
  3. User can generate confusion matrices for the intent classification task (at least per track)
**Plans**: 2 plans

Plans:
- [ ] 04-01: Export JSON benchmark results file
- [ ] 04-02: Generate comparison visualizations (bar charts, confusion matrices)

## Coverage

13/13 v1 requirements mapped (no duplicates, no orphans).

| Requirement | Phase |
|-------------|-------|
| DATA-01 | Phase 1 |
| DATA-02 | Phase 1 |
| DATA-03 | Phase 1 |
| DATA-04 | Phase 1 |
| TTS-01 | Phase 2 |
| TTS-02 | Phase 2 |
| TTS-03 | Phase 2 |
| EVAL-01 | Phase 3 |
| EVAL-02 | Phase 3 |
| EVAL-03 | Phase 3 |
| EVAL-04 | Phase 3 |
| EVAL-05 | Phase 3 |
| RSLT-01 | Phase 4 |
| RSLT-02 | Phase 4 |

## Progress

**Execution Order:**
Phases execute in numeric order: 1 -> 2 -> 3 -> 4

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 1. Data Pipeline | 3/3 | ✓ Complete | 2026-01-31 |
| 2. Speech Synthesis | 0/3 | Not started | - |
| 3. Benchmark Evaluation | 0/3 | Not started | - |
| 4. Results & Visualization | 0/2 | Not started | - |
