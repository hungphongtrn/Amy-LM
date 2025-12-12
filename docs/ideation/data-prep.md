# Offline Data Preparation

## **Objective**

Create a massive dataset of "Bridge Tuples" rigidly aligned to Mimi's **12.5 Hz** clock.

## **Specifications**

* **Audio Standard:** 24,000 Hz, Mono.
* **Frame Rate:** 12.5 Hz (Duration: 80ms).
* **Stride:** 1,920 samples $(24000 \times 0.08)$.
* **Text Model:** Qwen/Qwen3-1.7B (Hidden Dim: 2048).
* **Speech Model:** WavLM-Large (Feature Dim: 1024).

## **Pipeline Workflow**

### **Step 1: Alignment (The "Time" Truth)**

**Tool:** NVIDIA NeMo (`parakeet-ctc-1.1b` + `nemo_forced_aligner`).

1. **ASR & Force Align:** Generate `.ctm` files containing word/token timestamps.
2. **Quantization (12.5 Hz):**
      * Map exact timestamps to 80ms grid indices.
      * *Logic:* If a frame overlaps multiple tokens, assign the token at the frame's temporal center (40ms mark).
      * *Result:* An integer array `[TokenID_A, TokenID_A, TokenID_B, ..., SILENCE]`.

### **Step 2: Semantic Extraction (The "Content" Truth)**

**Tool:** HuggingFace Transformers (`Qwen/Qwen3-1.7B`).

* **Input:** The raw text transcript.
* **Extraction:**
  * Run forward pass.
  * Extract **Hidden States from Layer 27** (Index `-2`).
  * *Reason:* Layer 28 is too specialized for next-token prediction; Layer 27 captures broader semantic context.
* **Output:** `.npy` file of shape `[N_Tokens, 2048]`, `float16`.

### **Step 3: Acoustic Extraction (The "Prosody" Truth)**

**Tool:** `microsoft/wavlm-large`.

* **Extraction:** Extract features from the final transformer layer (or weighted sum).
* **Downsampling:** WavLM natively outputs \~50Hz.
  * *Action:* Apply **Average Pooling** with **Kernel Size 8** and **Stride 4** (Overlapping).
  * *Note:* This matches Kyutai's configuration to smooth the 50Hz WavLM features before distillation.
* **Output:** `.npy` file of shape `[N_Frames, 1024]`, `float16`.

### **Step 4: Packaging (WebDataset)**

Shard data into `.tar` files for high-performance streaming.

* **Format:**
  * `sample_001.wav` (Raw Audio)
  * `sample_001.qwen.npy` (Hidden States: `[T, 2048]`)
  * `sample_001.wavlm.npy` (Pooled Features: `[F, 1024]`)
  * `sample_001.dur.npy` (RLE Durations: `[T]`)
