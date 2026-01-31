# Requirements: Proactive-SAT

**Defined:** 2025-01-31
**Core Value:** AI agents must detect hidden intent from prosody alone to enable proactive assistance in high-stakes scenarios

## v1 Requirements

Requirements for initial MVP (200 samples, 1-day timeline).

### Data Pipeline

- [ ] **DATA-01**: Load and parse source data from `.data/` directory
- [ ] **DATA-02**: Build Lexical Neutralizer to rewrite utterances with neutral text
- [ ] **DATA-03**: Generate Speaker Instructions for prosodic injection (sarcastic, frustrated, distressed)
- [ ] **DATA-04**: Create HuggingFace dataset with 200 samples (Control + Trigger variants)

### Speech Synthesis

- [ ] **TTS-01**: Integrate Qwen3-TTS for speech generation
- [ ] **TTS-02**: Generate Control variants (flat, factual delivery)
- [ ] **TTS-03**: Generate Trigger variants (prosodically rich delivery)

### Benchmark Evaluation

- [ ] **EVAL-01**: Implement Text-only LLM track (GPT-5 on neutral transcripts)
- [ ] **EVAL-02**: Implement ASR+LLM track (Whisper → GPT-5)
- [ ] **EVAL-03**: Implement End-to-End SLM track (Gemini 3 Flash on raw audio)
- [ ] **EVAL-04**: Run Intent Classification task across all tracks
- [ ] **EVAL-05**: Run Proactive Triggering task across all tracks

### Results & Visualization

- [ ] **RSLT-01**: Generate JSON benchmark results file
- [ ] **RSLT-02**: Create comparison visualizations (bar charts, confusion matrices)

## v2 Requirements

Deferred to future release.

### Dataset Scale

- **DATA-05**: Expand to 1000+ samples
- **DATA-06**: Multi-language support
- **DATA-07**: Long-form audio support (>30s)

### Advanced Evaluation

- **EVAL-06**: Additional SLM models (Qwen-Audio, etc.)
- **EVAL-07**: Fine-grained prosody analysis
- **EVAL-08**: Real-time inference benchmarking

## Out of Scope

| Feature | Reason |
|---------|--------|
| Custom TTS training | Using Qwen3-TTS off-the-shelf is sufficient |
| Real-time optimization | Batch processing sufficient for benchmark |
| Multi-modal evaluation | Focus on audio-only for v1 |
| Human evaluation baseline | Automated metrics only for MVP |

## Traceability

Which phases cover which requirements. Updated during roadmap creation.

| Requirement | Phase | Status |
|-------------|-------|--------|
| DATA-01 | Phase 1 | Pending |
| DATA-02 | Phase 1 | Pending |
| DATA-03 | Phase 1 | Pending |
| DATA-04 | Phase 1 | Pending |
| TTS-01 | Phase 2 | Pending |
| TTS-02 | Phase 2 | Pending |
| TTS-03 | Phase 2 | Pending |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| EVAL-04 | Phase 3 | Pending |
| EVAL-05 | Phase 3 | Pending |
| RSLT-01 | Phase 4 | Pending |
| RSLT-02 | Phase 4 | Pending |

**Coverage:**
- v1 requirements: 13 total
- Mapped to phases: 13
- Unmapped: 0 ✓

---
*Requirements defined: 2025-01-31*
*Last updated: 2026-01-31 after roadmap creation*
