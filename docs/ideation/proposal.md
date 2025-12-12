# **Title:** The Semantic Bridge: Disentangling Content and Prosody in Neural Audio Codecs via Causal LLM Distillation

## **1. Executive Summary**

Current neural audio codecs (e.g., EnCodec, Mimi) achieve high compression but suffer from "entanglement"—the semantic meaning (text) is inextricably mixed with acoustic details (speaker identity, emotion). This limits their utility for downstream editing and zero-shot voice transfer.

This research proposes the **"Semantic Bridge" Mimi**, a modified architecture that enforces a strict hierarchical disentanglement. By locking the first codebook to the **Causal Hidden States** of a Large Language Model (Qwen2.5) and the second to a self-supervised speech model (WavLM), we create a codec where "Text," "Prosody," and "Timbre" are explicitly separated layers.

## **2. Core Objectives**

* **Primary Goal:** Train a 9-Codebook Mimi model where **Codebook 0** represents pure semantic content (aligned with LLM states) and **Codebook 1** represents prosody, independent of speaker timbre.
* **Target Metrics:**
  * **Frame Rate:** 12.5 Hz (80ms per frame).
  * **Bitrate:** \~1.2 kbps.
  * **Intelligibility:** WER \< 5% on reconstructed audio using only Codebooks 0+1.
* **Capabilities:** Zero-shot Voice Swapping (mixing Codebook 0 from Source A with Codebooks 1-8 from Source B) and text-based audio editing.

## **3. Methodology**

We employ a **"Teacher-Student Distillation"** strategy:

1. **The Semantic Anchor:** Codebook 0 is forced to predict the **Layer 27 Hidden State** of Qwen2.5-1.5B. This injects causal linguistic context (meaning) into the audio token.
2. **The Prosody Anchor:** Codebook 1 is forced to predict the residual features of WavLM-Large.
3. **The Acoustic Residuals:** Codebooks 2-8 utilize standard GAN/Mel-spectrogram losses to reconstruct high-fidelity timbre and environmental noise.

## **4. Execution Roadmap**

* **Phase 1: Data Factory (Weeks 1-2):** Build the $(Audio, Qwen_{State}, WavLM, Duration)$ tuples.
* **Phase 2: Model Engineering (Week 3):** Implement the 9-Codebook Bridge Quantizer and Weight Surgery.
* **Phase 3: Curriculum Training (Weeks 4-6):** Use "Acoustic Dropout" to force dependency on the semantic head.
* **Phase 4: Validation (Week 7):** Verify disentanglement via "Frankenstein" voice swapping tests.
