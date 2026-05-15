# Comprehensive Research: Benchmarking Prosody and Timbre Performance in Speech/Audio Models

*Foundation Documentation for Testing "Residual Embedding Extension" vs "Text-Aligned Projection" Architectures*

---

## Executive Summary

This document provides a comprehensive review of benchmarks for evaluating prosody (rhythm, intonation, stress) and timbre (voice characteristics) understanding in speech/audio models. The research is organized to support testing whether a **"residual embedding extension"** architecture (which preserves full acoustic information) outperforms a **"text-aligned projection"** architecture (which maps speech to text-like representations) on tasks requiring prosodic and timbral understanding.

---

## Table of Contents

1. [Emotion Recognition Benchmarks](#1-emotion-recognition-benchmarks)
2. [Sarcasm, Irony & Contradiction Detection](#2-sarcasm-irony--contradiction-detection)
3. [Paralinguistic Challenge Tasks](#3-paralinguistic-challenge-tasks)
4. [Speaker and Voice Characteristics](#4-speaker-and-voice-characteristics)
5. [Prosody-Specific Benchmarks](#5-prosody-specific-benchmarks)
6. [Non-Verbal Vocalizations](#6-non-verbal-vocalizations)
7. [Multi-Modal Contradiction Benchmarks](#7-multi-modal-contradiction-benchmarks)
8. [Evaluation Methodology](#8-evaluation-methodology)
9. [Existing Speech LM Evaluations](#9-existing-speech-language-model-evaluations)
10. [Clinical/Crisis-Relevant Benchmarks](#10-clinicalcrisis-relevant-benchmarks)
11. [Recommended Minimal Benchmark Suite](#11-recommended-minimal-benchmark-suite)

---

## 1. Emotion Recognition Benchmarks

### 1.1 Standard Datasets Overview

| Dataset | Size | Speakers | Emotions | Type | License | Language |
|---------|------|----------|----------|------|---------|----------|
| **IEMOCAP** | ~12h, 10k utts | 10 (5M/5F) | 9 (excited, frustrated, etc.) | Acted + Improvised | ❌ Academic only | English |
| **RAVDESS** | ~1h, 1,440 utts | 24 (12M/12F) | 8 (speech + song) | Acted | ✅ Open | English |
| **CREMA-D** | ~6h, 7,400 utts | 91 (48M/43F) | 6 basic emotions | Crowd-acted | ✅ Open | English |
| **SAVEE** | ~0.8h, 480 utts | 4 (Male only) | 7 emotions | Acted | ❌ Restricted | English |
| **EmoDB (German)** | <1h, 535 utts | 10 (5M/5F) | 7 emotions | Acted | ❌ Restricted | German |
| **TESS** | ~0.5h, 2,800 utts | 2 (Female only) | 7 emotions | Acted | ✅ Open | English |
| **MSP-Podcast** | 100k+ segments | 1,000+ | 8 emotions + VAD | Naturalistic | ❌ Challenge use | English |
| **CMU-MOSEI** | 65h, 23,453 utts | 1,000+ | Sentiment + 6 emotions | In-the-wild YouTube | ✅ Open | English |
| **CMU-MOSI** | 2.6h, 2,199 utts | 98 | Sentiment [-3,3] | Opinion videos | ✅ Open | English |
| **MELD** | ~13h, 1,400 utts | Multiple | 7 emotions | TV show dialogues | ✅ Open | English |

### 1.2 Current SOTA Performance (2024-2025)

**IEMOCAP (4-class: happy, sad, angry, neutral):**
- SOTA (2025): 79.14% Unweighted Accuracy (UA) - Energy-aware multimodal learning approach
- Previous SOTA: 76.04% (Kang et al., LAM baseline)
- Text-only baselines typically fail (prosody essential)

**RAVDESS:**
- SOTA (2025): 93.40% Weighted Accuracy (WA) / 92.28% UA
- CNN with raw audio: 90.42% accuracy
- Transformer-based: 87.99% (multimodal)

**CREMA-D:**
- SOTA (2025): 77.70% accuracy (5-model ensemble with DeepSpecCNN)
- Wav2Vec2 + fine-tuning: ~67% UAR
- Cross-corpus generalization challenging (~47% when tested on RAVDESS)

**MELD:**
- SOTA (2025): 94.0% accuracy (1D-CNN with multi-feature fusion)
- Multimodal approaches outperform audio-only

### 1.3 Key Metrics

- **Weighted Accuracy (WA)** / **Accuracy**: Accounts for class imbalance
- **Unweighted Accuracy (UA)** / **Macro-F1**: Equal weight per class; crucial for imbalanced datasets
- **Concordance Correlation Coefficient (CCC)**: For dimensional emotions (valence, arousal, dominance)
- **F1-score**: Per-class performance

### 1.4 Known Limitations

**Acted vs. Natural Speech:**
- RAVDESS, EmoDB, TESS: Clean studio recordings, exaggerated prosody
- IEMOCAP: Mix of scripted (acted) and improvised (more natural)
- MSP-Podcast: Naturalistic speech from podcasts (current gold standard)
- CMU-MOSEI: "In-the-wild" YouTube videos with noise

**Label Quality Issues:**
- Crowd-sourced labels (CREMA-D) show variability
- Expert annotations (EmoNet-Voice): Higher reliability but expensive
- Fine-grained emotions (>6 categories) have low inter-annotator agreement

**Projection vs Extension Relevance: 4/5** ⭐⭐⭐⭐
- Emotion recognition is THE primary benchmark for prosody understanding
- Text-only models achieve ~30-40% on audio-only emotion tasks (random ~16%)
- Strong signal for whether architecture preserves paralinguistic information

---

## 2. Sarcasm, Irony & Contradiction Detection

### 2.1 Datasets Overview

| Dataset | Size | Modalities | Language | Key Feature |
|---------|------|------------|----------|-------------|
| **MUStARD** | 690 samples | Text + Audio + Video | English | TV show clips (Friends, Big Bang Theory) |
| **MUStARD++** | ~1,000 samples | Text + Audio + Video | English | Adds emotion labels, sarcasm types |
| **MUStARD++ Balanced** | Extended | Text + Audio + Video | English | Addresses class imbalance |
| **SEEmoji MUStARD** | 690 samples | + Emoji labels | English | Emoji sentiment/emotion annotations |
| **MCSD 1.0** | ~500 samples | Text + Audio + Video | Chinese | Mandarin sarcasm |
| **MuSaG** | ~200 samples | Text + Audio + Video | German | German TV shows |

### 2.2 SOTA Performance

**MUStARD/MUStARD++:**
- Best multimodal (Text+Audio+Video): 73.97% F1 (ConAttSD with contrastive attention)
- Text-only: ~64% F1
- Audio-only: ~58% F1
- Video-only: ~68% F1 (human performance)
- **Key finding**: Humans rely most on audio cues (87.93% F1) for sarcasm detection

**Recent Benchmarking (2024):**
- ViFi-CLIP + Wav2Vec2.0 + BART: 73.6% macro-F1 (with context)
- SOTA encoders improved performance by 3.4-3.9% over classical MFCC features

### 2.3 Social Deafness / Contradiction Detection

**MCR-Bench (Modal Conflict Resolution):**
- **Purpose**: Evaluates text bias when audio and text contradict
- **Finding**: LALMs show severe text bias—when audio says "happy" but text says "sad", models trust text
- **Performance drop**: Up to 98% accuracy degradation (87.8% → 1.7% on Qwen-Audio-Chat)
- **Relevance**: Critical for crisis support where tone may contradict words ("I'm fine" said sadly)

**Cross-Modal NLI:**
- Speech-text entailment/contradiction detection
- F1 scores: 0.70-0.77 for speech-speech and speech-text tasks
- Shows promise for detecting modality mismatches

### 2.4 Sarcasm Types (in MUStARD++)

1. **Propositional**: Literal meaning opposite of intended
2. **Embedded**: Sarcasm embedded within larger utterance
3. **Like-prefixed**: "Like" signals sarcastic intent
4. **Illocutionary**: Speech act itself is sarcastic

**Projection vs Extension Relevance: 5/5** ⭐⭐⭐⭐⭐
- Sarcasm is the canonical test for prosody-text integration
- Text-only models fail catastrophically (~60% vs 88% human)
- Requires understanding incongruity between literal text and prosodic cues
- Perfect benchmark for testing "residual embedding" hypothesis

---

## 3. Paralinguistic Challenge Tasks (ComParE)

### 3.1 ComParE Challenge Series (2009-2024)

The INTERSPEECH/ACM Multimedia Computational Paralinguistics Challenge (ComParE) is the premier benchmark series for paralinguistic tasks.

| Year | Tasks | Metric | Baseline Performance |
|------|-------|--------|---------------------|
| **2025 (INTERSPEECH)** | Speech Emotion in Naturalistic Conditions | Macro-F1 | 0.329 (baseline), 0.432 (SOTA) |
| **2023** | Emotion Share (regression), Requests/Complaints | UAR/ρ | 67.2% UAR (Requests), ρ=0.514 (Emotion Share) |
| **2022** | Vocalisations, Stuttering, Activity, Mosquitoes | UAR | 44.0% (Vocalisations), 62.1% (Stuttering) |
| **2021** | COVID-19 Cough/Speech, Escalation, Primates | AUC/UAR | 76.2% (COVID Cough), 63.9% (Escalation) |
| **2020** | Breathing, Elderly Emotion, Masks | CC/UAR | ρ=0.778 (Breathing), 63.8% (Elderly Emotion) |
| **2019** | Styrian Dialects, Continuous Sleepiness, Baby Sounds, Orca | UAR/AUC | 50.6% (Dialects), 63.9% (Baby Sounds) |
| **2018** | Atypical Affect, Self-Assessed Affect, Crying, Heart Beats | UAR | 45.0% (Atypical), 68.4% (Self-Assessed) |
| **2017** | Addressee, Cold, Snoring | UAR | 70.2% (Addressee), 71.0% (Cold) |
| **2016** | Deception, Sincerity, Native Language | UAR | 72.1% (Deception), 65.4% (Sincerity) |
| **2015** | Nativeness, Parkinson's, Eating Condition | AUC/UAR | 43.3% (Nativeness), 54.0% (Parkinson's) |

### 3.2 Crisis-Relevant Tasks

**Deception Detection (2016):**
- Task: Classify deceptive vs. truthful speech
- Baseline: 72.1% UAR using ComParE features
- Key insight: Pitch increases significantly in deceptive speech

**Escalation Detection (2021):**
- Task: Detect conflict escalation in conversations
- 3-class: low, medium, high escalation
- 63.9% UAR—challenging but critical for crisis intervention

**Requests/Complaints (2023):**
- Customer service call analysis
- Requests: 67.2% UAR
- Complaints: 52.9% UAR (near chance—very difficult)

### 3.3 Key Features Used

- **ComParE 2016 feature set**: 6,373 acoustic features (openSMILE)
- **eGeMAPS**: 88 low-level descriptors (simpler, faster)
- **Deep features**: Wav2Vec2.0, HuBERT fine-tuned representations

### 3.4 Recent Trends (ComParE 2025)

- 93 teams participated in SER in Naturalistic Conditions
- Top performers use WavLM (70%) + Whisper (47%) + RoBERTa
- Ensemble methods essential (95% of top teams used ensembling)
- Multi-modal (audio + text) consistently outperforms audio-only

**Projection vs Extension Relevance: 4/5** ⭐⭐⭐⭐
- Deception and escalation tasks directly probe crisis-relevant prosody
- Clinical and counseling applications
- Challenging naturalistic conditions test robustness

---

## 4. Speaker and Voice Characteristics

### 4.1 Datasets

| Dataset | Size | Task | Metrics | License |
|---------|------|------|---------|---------|
| **VoxCeleb1** | 150k utts, 1,251 speakers | Speaker Verification | EER, minDCF | ✅ Open |
| **VoxCeleb2** | 1M+ utts, 6,112 speakers | Speaker Verification | EER, minDCF | ✅ Open |
| **VoxCeleb Enrichment** | VoxCeleb2 + metadata | Age/Gender/Accent | F1, MAE | ✅ Open |
| **VoxAging** | 293 speakers, 17 years | Speaker Aging | EER over time | ✅ Open |
| **CommonAccent** | Subset | Accent Classification | Accuracy | ✅ Open |
| **GLOBE** | 11 datasets combined | Age/Gender/Accent | Various | ✅ Open |
| **Vox-Profile** | 15+ datasets | Multi-trait benchmark | F1, CCC | ✅ Open |

### 4.2 SOTA Performance

**VoxCeleb1 Verification (VoxSRC 2023 winners):**
- Track 1 winner: EER 0.58%, minDCF 0.028
- Track 2 winner: EER 0.47%, minDCF 0.020
- Hard positive pairs (large age gap): Performance degrades significantly

**Age/Gender Recognition (VoxCeleb Enrichment):**
- Gender: F1 = 0.9829 (i-vector + logistic regression)
- Age regression: MAE = 9.44 years (challenging)
- Ridge regression performs best for age

**Vox-Profile Benchmark (2025):**
- 15+ datasets combined for multi-trait evaluation
- Static traits: Sex, Age, Accent
- Dynamic traits: Emotion, Voice Quality, Speech Flow, Expressiveness
- Models: WavLM-large + LoRA adapters

### 4.3 SVeritas Benchmark (2025)

Comprehensive speaker verification evaluation under stress:
- Natural variations: duration, spontaneity
- Background: noise, reverberation, channel mismatch
- Physical: age, health conditions
- Adversarial: spoofing attacks

**Key finding**: Models suffer substantial degradation in cross-language trials, age mismatches, and codec compression.

**Projection vs Extension Relevance: 3/5** ⭐⭐⭐
- Timbre preservation is important for speaker identity
- Less directly related to crisis/prosody focus
- Useful for verifying voice conversion quality

---

## 5. Prosody-Specific Benchmarks

### 5.1 Structural Prosody

**ProsAudit (INTERSPEECH 2023):**
- **Protosyntax task**: Identify strong vs. weak prosodic boundaries
- **Lexical task**: Distinguish pauses between words vs. within words
- Human performance: 80.5% (protosyntax), ~70% (lexical)
- SSL models perform above chance but below human level
- Non-native models perform significantly worse on lexical task

### 5.2 Mandarin Speech Prosody Benchmark (MSPB, 2025)

Eight linguistically-grounded tasks for evaluating Speech LLMs:

| Task | Description | Human | GPT-4o | Gap |
|------|-------------|-------|--------|-----|
| Intonation | 2-alternative forced choice | ~95% | ~85% | ~10% |
| Prosodic Disambiguation | "chocolate milk" vs "chocolate, milk" | ~95% | ~75% | ~20% |
| Prosodic Focus Marking | Stress for emphasis | ~90% | ~60% | ~30% |
| Focus Operator | "Only JOHN came" vs "John ONLY came" | ~95% | ~80% | ~15% |
| Scalar Meaning | "some" vs "some but not all" | ~90% | ~65% | ~25% |
| Irony (with context) | Sarcasm understanding | ~95% | ~85% | ~10% |
| Emotional Prosody (w/ context) | Emotion + prosody | ~95% | ~80% | ~15% |
| Emotional Prosody (no context) | Prosody only | ~90% | ~55% | ~35% |

**Key finding**: Speech LLMs rely more on context than subtle prosodic variations. Performance gap is largest on prosody-only tasks.

### 5.3 Emphasis and Stress

**CAST (Context-Conditioned Stress in TTS, 2025):**
- Evaluates whether TTS systems can infer and realize context-appropriate stress
- Text-only LLMs reliably recover intended stress from context (~85%)
- Current TTS systems fail to realize this in speech (~40-50%)
- **WHISTRESS**: Automatic stress detector based on Whisper

**EmphAssess (ACL 2024):**
- Evaluates emphasis transfer in speech-to-speech models
- EmphaClass: New model for frame/word-level emphasis classification

### 5.4 Prosody Evaluation Metrics

**DS-WED (Discretized Speech Weighted Edit Distance, 2025):**
- Measures prosody diversity via semantic token sequences
- Pearson correlation with PMOS: r=0.77 (vs. 0.30 for log F0 RMSE, 0.66 for MCD)
- Outperforms traditional acoustic metrics

**TTScore (2025):**
- TTScore-int: Intelligibility via content token prediction
- TTScore-pro: Prosody via prosody token prediction
- Reference-free evaluation framework

**Projection vs Extension Relevance: 5/5** ⭐⭐⭐⭐⭐
- Direct measurement of prosody understanding
- MSPB shows 35% gap on prosody-only tasks
- Perfect for testing architecture's prosodic preservation

---

## 6. Non-Verbal Vocalizations

### 6.1 Datasets

| Dataset | Size | NV Types | Language | Collection |
|---------|------|----------|----------|------------|
| **VocalSound** | 21,024 samples | 6 (laugh, sigh, cough, throat, sneeze, sniff) | Any | Crowdsourced (AMT) |
| **EmoGator** | 32,130 samples | 30 emotion categories via vocal bursts | English | Lab recorded |
| **NonverbalTTS (NVTTS)** | 17 hours | 10 types (laugh, sigh, cough, etc.) + 8 emotions | English | VoxCeleb + Expresso |
| **NonVerbalSpeech-38K** | 38,718 samples, 131h | 10 types (laugh, cough, breath, etc.) | EN, ZH | In-the-wild media |
| **MNV-17** | 7.55 hours | 17 NV categories | Mandarin | Performative/scripted |
| **Emilia-NV** | 48k human + 174k auto | 18 paralinguistic categories | Mandarin | Auto + human validation |
| **NVBench** | 4,500 samples | 45-type taxonomy | EN, ZH | Curated bilingual |

### 6.2 Key Findings

**VocalSound:**
- 3,365 unique subjects from 60 countries
- Adding VocalSound to training improves real-world recognition by 41.9%
- Contains metadata: age, gender, native language, health condition

**NonVerbalSpeech-38K:**
- Scalable automatic annotation framework
- Two alignment methods: Timestamp-Based Ordering (TBO) and Temporal-Semantic Alignment (TSA)
- NV events rarely overlap with semantic content (<5%)

**MNV-17:**
- Most extensive NV categories among public datasets
- Scripted approach ensures clear articulation
- Joint ASR + NV classification: Qwen2.5-omni achieves 3.60% CER

### 6.3 Recognition Performance

- **Laughter detection**: F1 ~0.69 in controlled (Switchboard), drops to ~0.53 in noisy (AudioSet)
- **Breathing**: Most common NV type (~40% of occurrences)
- **Crying/Sobbing**: Most challenging to detect and synthesize
- **Cross-lingual**: Models trained on English generalize reasonably to Chinese

**Projection vs Extension Relevance: 4/5** ⭐⭐⭐⭐
- Critical for expressive speech synthesis
- Tests acoustic richness beyond text content
- Laughter and crying are crisis-relevant signals

---

## 7. Multi-Modal Contradiction Benchmarks

### 7.1 Audio-Text Contradiction

**MCR-Bench (2025):**
- 3,000 samples across 3 audio-centric tasks
- Adversarial (contradictory), faithful, and irrelevant text descriptions
- Tasks: Audio QA, Speech Emotion Recognition, Sound Event Detection
- **Text Bias Index (TBI)**: Quantifies preference for text over audio
- **Finding**: All tested LALMs show >95% TBI—severe text bias when modalities conflict

### 7.2 Vision-Language Contradiction

**Clash (2025):**
- Image-text contradiction detection
- Object-level and attribute-level contradictions
- ~15k training samples + human-verified diagnostic set
- Open-source models achieve near-zero performance without fine-tuning

### 7.3 Sarcasm as Contradiction

**MUStARD Findings:**
- Sarcasm relies on incongruity between modalities
- Text says X, audio/visual convey "not X"
- Models that explicitly model contrastive features achieve 73.97% F1
- Text-only models miss the point entirely

### 7.4 Multilingual NLI

- Speech-speech and speech-text entailment/contradiction
- F1 scores 0.70-0.77 for contradiction detection
- Better than similarity-based approaches for detecting mismatches

**Projection vs Extension Relevance: 5/5** ⭐⭐⭐⭐⭐
- Directly tests whether architecture can detect text-prosody mismatch
- Critical for crisis support (detecting when words say "fine" but tone says "distress")
- MCR-Bench shows catastrophic failure of text-aligned models

---

## 8. Evaluation Methodology

### 8.1 LLM-as-Judge for Prosody

**AudioJudge (EACL 2026):**
- Systematic study of Large Audio Models (LAMs) as judges
- Tasks: pronunciation, speaking rate, speaker ID, speech quality
- **Strategies**: Audio concatenation + in-context learning significantly improves performance
- **Multi-aspect ensemble**: 3 specialized judges (lexical, paralinguistic, speech quality)
- **Spearman correlation**: Up to 0.91 with human preferences
- **Biases**: Verbosity bias, positional bias must be mitigated

**TRACE (2026):**
- Textual Reasoning over Audio Cues for Evaluation
- LLM judges reason over audio cues via textual blueprint
- Separates evaluation into: Content (C), Voice Quality (VQ), Paralinguistics (P)
- Higher agreement with humans than ALMs, more cost-effective

### 8.2 Speech Captioning

**AffectSpeech (2025):**
- 252,999 utterances with fine-grained emotion + prosody descriptions
- Human-in-the-loop verification with LLM adjudication
- Metrics: Emotion/prosody accuracy, lexical diversity, structural consistency
- Human evaluation: 16 subjects rating emotion, prosody, comprehension

### 8.3 Objective Metrics

**Prosody Metrics:**
- F0 RMSE: Reference-dependent, alignment-sensitive
- MCD (Mel Cepstral Distortion): Measures spectral difference
- DS-WED: Semantic token-based, reference-free
- PMOS (Prosody Mean Opinion Score): Human ratings of prosody diversity

**Similarity Metrics:**
- Emotion Embedding Cosine Similarity (EECS): Prone to linguistic/speaker interference
- **Caution**: EMO-SIM is unreliable—degrades below chance with distractors

### 8.4 Human Evaluation Protocols

**A/B Testing:**
- Compare two speech samples directly
- Randomized ordering to mitigate bias
- Trap questions to ensure attention

**AXY Discrimination:**
- Reference (A) + two candidates (X, Y)
- Rate which candidate is closer to reference prosody
- 7-point scale: -3 (X much closer) to +3 (Y much closer)

**MOS (Mean Opinion Score):**
- 1-5 scale for naturalness, quality, prosody
- Minimum 4 raters per sample
- 95% confidence intervals reported

**Projection vs Extension Relevance: 4/5** ⭐⭐⭐⭐
- Standardizes evaluation across architectures
- DS-WED and TTScore provide objective prosody measures
- Human evaluation remains gold standard

---

## 9. Existing Speech Language Model Evaluations

### 9.1 Major Speech LMs and Their Evaluations

| Model | Architecture | Key Benchmarks | Prosody Performance |
|-------|-------------|------------------|---------------------|
| **Ichigo** | Early-fusion, discrete tokens | AudioBench (OpenHermes, ALPACA), VoiceBench | 67.8 OpenHermes, 67.2 ALPACA—outperforms Qwen2-Audio |
| **Qwen2-Audio** | Multi-task, frozen LLM | 13 benchmarks (SER, ASR, S2TT, AIR-Bench) | SER on MELD: 53.5% ACC (lags WavLM) |
| **SALMONN** | Dual auditory encoders + LLM | SALMon, AudioBench | Speech quality assessment: competitive with WavLM |
| **SpeechGPT** | Discrete unit-based | ASR, emotion, speaker | Limited prosody-specific evaluation |
| **SpiritLM** | Expressive + discrete | SALMon | Sentiment consistency: 73.5% |
| **TWIST** | Tokenized speech | SALMon | Sentiment consistency: 61.5% |
| **Gemini-1.5-Pro** | Commercial multimodal | Various | Strong baseline for comparison |
| **GPT-4o** | Multimodal native | AudioBench | Strong on context-rich tasks |

### 9.2 AudioBench Results (2024-2025)

| Model | OpenHermes-Audio | ALPACA-Audio | Latency |
|-------|------------------|--------------|---------|
| Whisper + Llama-3 8B | 63.0 | 70.8 | ~45ms |
| SALMONN | 19.2 | 12.4 | N/A |
| Qwen2-Audio | 44.8 | 52.0 | ~317ms |
| **Ichigo v0.3** | **67.8** | **67.2** | **~110ms** |

### 9.3 SALMon Benchmark Results (ICASSP 2025)

| Method | Sentiment Consistency | Speaker Consistency | Gender Consistency |
|--------|----------------------|---------------------|-------------------|
| Human Baseline | 97.2 | 91.2 | 98.6 |
| SpiritLM 7B (Expr.) | 73.5 | 81.0 | 85.0 |
| TWIST 7B | 61.5 | 71.0 | 70.0 |
| pGSLM | 40.5 | 83.0 | 88.5 |
| Flow-SLM-270M | 61.5 | 75.5 | 78.0 |
| CAST-1B | 81.8 | 90.0 | 90.0 |

**SALMon Tests:**
- **Sentiment Consistency**: Does model recognize consistent emotion across samples?
- **Speaker Consistency**: Does model identify same speaker across utterances?
- **Gender Consistency**: Does model maintain gender recognition?
- **Background/Room Consistency**: Does model ignore/accommodate acoustic environment?
- **Alignment**: Does audio match described content?

### 9.4 SUPERB Benchmark Family

**SUPERB (2021):**
- 15 tasks: ASR, emotion, speaker, intent, slot filling, etc.
- Emotion Recognition (ER) on IEMOCAP
- SSL representations + lightweight heads

**SUPERB-prosody (2022):**
- 3 prosody-related downstream tasks:
  1. Sentiment Analysis (SA)
  2. Sarcasm Detection (SarD)
  3. Persuasiveness Prediction (PP)
- 2 pseudo tasks:
  1. Prosody Reconstruction (pitch, energy)
  2. Future Value Prediction
- Finding: 13 of 15 SSL models outperform baseline on all prosody tasks

**TS-SUPERB (2025):**
- Target-speaker tasks: TS-ASR, TSE, PSE, PVAD
- Speaker embeddings as conditioning
- Tests whether SSL model performance on single-speaker tasks predicts multi-speaker performance

### 9.5 Key Observations

1. **Context over prosody**: Speech LLMs perform better on context-rich tasks than prosody-only tasks (MSPB findings)
2. **Latency vs quality**: Trade-off between real-time performance and prosody quality
3. **Multimodal fusion**: Audio + text consistently outperforms audio-only on emotion/sentiment
4. **Text bias**: LALMs show strong bias toward textual input (MCR-Bench)

**Projection vs Extension Relevance: 4/5** ⭐⭐⭐⭐
- Provides baseline comparisons for new architectures
- Shows current SOTA limitations (text bias, context reliance)
- SALMon specifically tests acoustic awareness

---

## 10. Clinical/Crisis-Relevant Benchmarks

### 10.1 Suicide Risk Detection

**SpeechWellness Challenge (SW1, INTERSPEECH 2025):**
- **Dataset**: 600 adolescents (10-18 years), 300 at-risk, 300 control
- **Tasks**: 
  1. Emotional Regulation (ER): Open-ended response
  2. Passage Reading (PR): Read poetry
  3. Expression Description (ED): Describe facial expression
- **Privacy**: Timbre modification applied, prosody preserved
- **Metric**: Binary classification (current suicide risk or not)

**EMASS Dataset:**
- Ecological Measurement of Affect, Speech, and Suicide
- Natural phone conversations + EMA self-reports
- Participants: Healthy controls, psychiatric controls, suicidal ideation (SI), suicide attempts (SA)
- Finding: Emotion predictions from speech can distinguish healthy from SI subjects

**Voice Biomarkers (Australian Telehealth Study):**
- 281 helpline calls (77 imminent risk, 204 low risk)
- 36 voice biomarkers → 12 primary markers via Lasso
- Classification: 99.85% of speech frames correctly classified (AUROC=1.0)
- **Key biomarkers**: Timing patterns (95% median accuracy), power spectral density (90.3%), MFCC (80%)

### 10.2 Depression Detection

**Methodological Study (2025):**
- 52-patient dataset from chronic pain clinic
- Acoustic features + SUDs (self-report) achieve 92.4% accuracy
- Acoustic features alone: 15% lift over baseline
- SUDs alone: 86% accuracy

**Meta-Analysis (2022):**
- 21 studies, 1,734 participants
- Depression detection from speech: ~80-85% accuracy typical
- Prosodic features (timing, pitch variation) most predictive

### 10.3 Distress Detection

**Voice Biomarkers for Psychological Distress:**
- 120 telephone counseling calls
- 7 vocal characteristics predict high vs. low distress
- AUROC: 97.39%, AUCPR: 97.52%
- **Key features**: Spectral slope, amplitude, formant frequencies

**Acoustic Analysis of Suicidality (2022):**
- 348 recordings from 104 mood disorder patients
- Between-person model: 69% accuracy (detect high risk)
- Within-person model: 79% accuracy (detect worsening)
- Longitudinal tracking shows promise

### 10.4 Crisis Helpline Applications

**Findings:**
- Timing patterns of speech most predictive of imminent risk
- Lower formant frequencies (F1, F2) in high-risk speech
- Jitter (voice roughness) increases with suicide risk
- Gender-specific models improve performance

### 10.5 Clinical Speech Characteristics

| Condition | Speech Markers |
|-----------|---------------|
| **Suicide Risk** | Reduced amplitude ("hushed tones"), increased spectral slope, slower speech rate, reduced pitch variation |
| **Depression** | Monotonic prosody, slower rate, longer pauses, lower intensity |
| **Anxiety** | Faster rate, higher pitch, more pauses (disfluency) |
| **Psychosis** | Disorganized prosody, unusual stress patterns |

**Projection vs Extension Relevance: 5/5** ⭐⭐⭐⭐⭐
- Ultimate test case for "residual embedding" hypothesis
- Crisis support requires understanding prosody beyond text
- High stakes: false negatives are costly
- Privacy concerns: timbre modification necessary

---

## 11. Recommended Minimal Benchmark Suite

Based on the comprehensive review above, we recommend an **8-benchmark suite** that maximally tests the "residual embedding extension" vs "text-aligned projection" hypothesis with minimal effort:

### Tier 1: Essential (Must Run)

#### 1. **MUStARD/MUStARD++ (Sarcasm Detection)**
- **Why**: Canonical test for text-prosody incongruity detection
- **Metric**: Macro-F1 on sarcasm classification
- **Setup**: Test audio-only, text-only, and audio+text
- **Hypothesis test**: Extension architecture should show >10% improvement on audio-only; Projection architecture may rely on text
- **Relevance**: 5/5
- **Effort**: Low (public dataset, 690 samples)
- **Expected SOTA**: 73-74% F1 (multimodal)

#### 2. **MCR-Bench (Modal Conflict Resolution)**
- **Why**: Directly measures text bias when audio-text contradict
- **Metric**: Performance drop under adversarial text, Text Bias Index (TBI)
- **Setup**: Evaluate with and without contradictory text prompts
- **Hypothesis test**: Extension should maintain performance under adversarial text; Projection should show severe degradation
- **Relevance**: 5/5
- **Effort**: Medium (need to implement adversarial prompts)
- **Expected**: 98% degradation in text-aligned models

#### 3. **MSP-Podcast SER (INTERSPEECH 2025 Challenge)**
- **Why**: Naturalistic emotion recognition, gold standard for SER
- **Metric**: Macro-F1 (8 classes), CCC (VAD dimensions)
- **Setup**: Use provided train/dev splits
- **Hypothesis test**: Extension should excel on spontaneous speech where prosody is subtle
- **Relevance**: 4/5
- **Effort**: Medium (larger dataset, requires registration)
- **Expected SOTA**: Macro-F1 ~0.43-0.45

#### 4. **MSPB (Mandarin Speech Prosody Benchmark)**
- **Why**: Linguistically-grounded prosody evaluation
- **Metric**: Accuracy on 8 prosody tasks, especially "Prosodic Focus Marking" and "Emotional Prosody (no context)"
- **Setup**: Test on English or adapt to other languages
- **Hypothesis test**: Extension should close the 35% gap on prosody-only tasks
- **Relevance**: 5/5
- **Effort**: Medium (may need language adaptation)
- **Expected**: Human ~90%, Current models ~55-60% on prosody-only

### Tier 2: Important (Should Run)

#### 5. **SpeechWellness Challenge (Suicide Risk Detection)**
- **Why**: Crisis-relevant, high-stakes application
- **Metric**: Accuracy on binary risk classification
- **Setup**: 3-task evaluation (ER, PR, ED)
- **Hypothesis test**: Extension should better detect distress signals in prosody
- **Relevance**: 5/5
- **Effort**: Medium (600 samples, may need registration)
- **Expected**: Baseline to be established (2025 challenge)

#### 6. **SUPERB-prosody (Sarcasm + Prosody Reconstruction)**
- **Why**: Standardized prosody evaluation framework
- **Metric**: Sarcasm detection F1, Pitch/Energy reconstruction error
- **Setup**: Use JSALT-2022-SSL implementation
- **Hypothesis test**: Extension should achieve lower reconstruction error
- **Relevance**: 4/5
- **Effort**: Low (open-source implementation)
- **Expected SOTA**: Sarcasm ~73% F1

#### 7. **NonVerbalSpeech-38K (Laughter/Crying Detection)**
- **Why**: Tests acoustic richness beyond text
- **Metric**: F1 on laughter, crying, sigh detection
- **Setup**: Evaluate on held-out test set
- **Hypothesis test**: Extension should better preserve non-verbal vocalization cues
- **Relevance**: 4/5
- **Effort**: Low (public dataset)
- **Expected**: Laughter F1 ~0.70 (controlled), ~0.55 (noisy)

### Tier 3: Supplementary

#### 8. **Vox-Profile (Speaker Trait Benchmark)**
- **Why**: Tests timbre preservation
- **Metric**: F1 on age/sex/accent classification, emotion expressiveness
- **Setup**: Use provided wavlm-large adapters
- **Hypothesis test**: Extension should maintain speaker characteristics
- **Relevance**: 3/5
- **Effort**: Medium (15+ datasets integrated)

### Implementation Priority

| Priority | Benchmark | Rationale |
|----------|-----------|-----------|
| **1** | MUStARD++ | Fast, definitive test of prosody-text integration |
| **2** | MCR-Bench | Critical for crisis support validation |
| **3** | MSP-Podcast SER | Industry standard, naturalistic data |
| **4** | MSPB | Pure prosody understanding test |
| **5** | SpeechWellness | Crisis-relevant validation |
| **6** | SUPERB-prosody | Standardized comparison |
| **7** | NonVerbalSpeech-38K | Expressive synthesis validation |
| **8** | Vox-Profile | Timbre preservation check |

### Expected Outcomes

**If "Residual Embedding Extension" is superior:**
- >10% improvement on MUStARD audio-only
- <30% degradation on MCR-Bench under adversarial text
- Closes 50%+ of prosody-only gap on MSPB
- Maintains performance on SpeechWellness

**If "Text-Aligned Projection" is sufficient:**
- Minimal improvement on audio-only tasks
- Severe degradation on MCR-Bench (>80%)
- Large remaining gap on prosody-only tasks
- Comparable performance on text-rich tasks

---

## Appendix A: Key Papers and Citations

### Benchmark Papers

1. **MUStARD**: Castro et al., "Towards Multimodal Sarcasm Detection," ACL 2019
2. **MUStARD++**: Ray et al., "Sarcasm in Sight and Sound," 2022
3. **MCR-Bench**: Wang et al., "When Audio and Text Disagree," 2025
4. **ComParE**: Schuller et al., Annual INTERSPEECH/ACM Multimedia Challenges
5. **MSPB**: Wang et al., "Mandarin Speech Prosody Benchmark," INTERSPEECH 2025
6. **SALMon**: Maimon et al., "Suite for Acoustic Language Model Evaluation," ICASSP 2025
7. **SUPERB**: Yang et al., "Speech processing Universal PERformance Benchmark," 2021
8. **SUPERB-prosody**: de Seyssel et al., "On the Utility of Self-Supervised Models for Prosody-Related Tasks," 2022
9. **SpeechWellness**: Wu et al., "Detecting Suicide Risk Among Adolescents," INTERSPEECH 2025
10. **NonVerbalSpeech-38K**: Ye et al., "Large-Scale Automatic Annotation of Non-Verbal Vocalizations," 2025

### Methodology Papers

1. **AudioJudge**: Chiang et al., "Understanding What Works in Large Audio Model Based Speech Evaluation," EACL 2026
2. **TRACE**: Anonymous, "Hearing Between the Lines: Unlocking the Reasoning Power of LLMs for Speech Evaluation," 2026
3. **DS-WED**: Talman et al., "ProsodyEval: Prosodic Prominence Benchmark," 2025

### Clinical Applications

1. **Voice Biomarkers for Suicide Risk**: Kandsberger et al., 2022
2. **EMASS Dataset**: Cummins et al., "Ecological Measurement of Affect, Speech, and Suicide"
3. **SpeechWellness**: Wu et al., INTERSPEECH 2025

---

## Appendix B: Dataset Access and Licensing

### Open Datasets (✅)
- RAVDESS, CREMA-D, TESS
- VoxCeleb1/2, Vox-Profile
- MUStARD/MUStARD++
- CMU-MOSEI, CMU-MOSI
- VocalSound
- NonVerbalSpeech-38K, NVTTS
- SUPERB, SUPERB-prosody
- ComParE 2016-2023 (via challenge registration)

### Restricted (❌ - Academic/Registration Required)
- IEMOCAP (email USC)
- SAVEE (commercial license)
- EmoDB (academic agreement)
- MSP-Podcast (challenge registration)
- SpeechWellness (challenge registration)

---

## Summary

This research document provides a comprehensive foundation for benchmarking prosody and timbre performance in speech/audio models. The **8-benchmark recommended suite** specifically targets the hypothesis that a "residual embedding extension" architecture will outperform "text-aligned projection" on tasks requiring prosodic understanding—particularly sarcasm detection, modal conflict resolution, naturalistic emotion recognition, and crisis-relevant distress detection.

The key insight across all reviewed benchmarks is that **current speech LMs exhibit strong text bias and struggle with prosody-only tasks**. The 35% performance gap on MSPB's "emotional prosody without context" task and the 98% degradation on MCR-Bench under adversarial text demonstrate the critical need for architectures that preserve rather than project acoustic information.

**Next Steps:**
1. Implement Tier 1 benchmarks (MUStARD++, MCR-Bench, MSP-Podcast, MSPB)
2. Evaluate both architectures on identical data splits
3. Analyze failure modes—where does each architecture fail and why?
4. Publish results with statistical significance testing

---

*Document Version: 1.0*
*Last Updated: May 2026*
*Research Areas: 10 comprehensive categories*
*Benchmarks Reviewed: 50+ datasets*
*Recommended Suite: 8 prioritized benchmarks*