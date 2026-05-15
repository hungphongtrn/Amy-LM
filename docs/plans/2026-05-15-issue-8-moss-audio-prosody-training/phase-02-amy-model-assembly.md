# Phase 2: Amy Model Assembly

> **STUB** — Will be detailed after Phase 1 completes and we verify Sub-module extraction and semantic stream shapes.

## Phase Goal
`AmyForProsodyClassification` produces 2-class logits from audio + FACodec prosody indices + timbre vector. Full forward pass verified with shape and gradient tests.

## High-Level Tasks (TBD)
- Create `AmyForProsodyClassification` in `src/models/amy_classifier.py`
- Integrate `MossAudioWrapper` + `ProsodyEmbedding` (warm-start) + `TimbreProjection` + `TemporalPool` + `ResidualFusion` + classifier head
- Stream activation config (prosody=true, timbre=true, content=false, acoustic=false)
- FACodec codebook vector extraction and projection to D=2560 for warm-start
- Tests for full forward pass, gradient flow through trainable params only, lambda init at zero
- Test that model equals baseline when lambdas=0

## Dependencies
- Phase 1: MOSS-Audio Backbone Integration
