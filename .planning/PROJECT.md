# Proactive-SAT

## What This Is

A benchmark for evaluating Speech-Language Models' ability to detect intent through prosodic (acoustic) cues rather than just text. Tests the "Modality Gap" — whether models can understand indirect speech acts where neutral text hides the true intent revealed only in voice (sarcasm, frustration, distress).

## Core Value

AI agents must detect hidden intent from prosody alone to enable proactive assistance in high-stakes scenarios.

## Requirements

### Validated

(None yet — ship to validate)

### Active

- [ ] Process source data from `.data/` into structured format
- [ ] Build Lexical Neutralizer to rewrite utterances with neutral text
- [ ] Generate Speaker Instructions for prosodic injection
- [ ] Integrate Qwen3-TTS for speech synthesis (Control vs Trigger variants)
- [ ] Create HuggingFace dataset with 200 samples
- [ ] Implement Text-only LLM evaluation (GPT-5)
- [ ] Implement ASR+LLM evaluation (Whisper → GPT-5)
- [ ] Implement End-to-End SLM evaluation (Gemini 3 Flash)
- [ ] Generate JSON benchmark results
- [ ] Create visualizations for results comparison

### Out of Scope

- Full-scale dataset (1000+ samples) — MVP only for v1
- Real-time inference optimization — batch processing sufficient
- Custom TTS training — using Qwen3-TTS off-the-shelf
- Multi-language support — English only for v1
- Long-form audio (>30s) — short utterances only

## Context

**Timeline:** 1 day remaining — aggressive MVP scope required

**Data Pipeline:**
- Source: `.data/` directory (existing pragmatic dataset)
- Processing: Lexical neutralization + prosodic instruction generation
- Output: HuggingFace dataset format

**Evaluation Architecture:**
| Track | Input | Model |
|-------|-------|-------|
| Text-Only | Neutral transcript | GPT-5 |
| ASR+LLM | Audio → Text | Whisper → GPT-5 |
| End-to-End | Raw audio | Gemini 3 Flash |

**TTS Strategy:**
- Control variant: Flat, factual delivery
- Trigger variant: Prosodically rich (sarcastic, frustrated, distressed)

**Benchmark Tasks:**
1. Intent Classification (hidden speech act detection)
2. Proactive Triggering (binary intervention decision)

## Constraints

- **Timeline:** 1 day — prioritize working pipeline over perfection
- **API Priority:** Commercial APIs for speed (GPT-5, Gemini 3 Flash)
- **TTS:** Qwen3-TTS only (https://github.com/QwenLM/Qwen3-TTS)
- **Sample Size:** 200 samples for MVP
- **Output Format:** JSON results + visualizations

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| GPT-5 for text-only | Commercial API, fast, state-of-the-art | — Pending |
| Whisper → GPT-5 for ASR track | Industry standard ASR + same LLM for fair comparison | — Pending |
| Gemini 3 Flash for E2E | Native audio support, fast, cost-effective | — Pending |
| Qwen3-TTS for synthesis | Instruction-following TTS enables prosodic control | — Pending |
| 200 samples for MVP | Achievable in 1 day, sufficient for proof-of-concept | — Pending |

---
*Last updated: 2025-01-31 after initialization*
