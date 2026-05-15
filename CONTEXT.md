# Amy-LM

Speech Language Model that understands prosody and timbre natively, solving "social deafness" — the failure to process emotional tone and subtext in spoken language. Built on Amy Codec (upstream neural audio codec) and Amy LM (downstream understanding model).

## Language

### Models

**Amy Codec**:
A neural audio codec that factorizes speech into 9 discrete codebooks (1 semantic + 1 prosody + 7 acoustic) at 12.5 Hz frame rate.
_Avoid_: Codec, encoder (ambiguous — could mean any audio encoder)

**Amy LM**:
The downstream Speech Language Model that consumes factorized speech codes and produces text understanding with prosody/timbre awareness.
_Avoid_: LLM, model (too generic in this context)

**FACodec**:
A third-party factorized neural audio codec (Microsoft, arXiv:2403.03100). Produces 1 prosody codebook (vocab=1024, 80 Hz), 2 content codebooks, 3 acoustic detail codebooks, and 1 global timbre vector. Used as a substitute for Amy Codec during pilot validation.
_Avoid_: FAcodec, FA codec

**MOSS-Audio**:
An open-source audio understanding model (OpenMOSS, Apache 2.0). 4B variant used as the semantic backbone for the issue #8 pilot. Architecture: Whisper-style audio encoder → GatedMLP adapter → Qwen3 language model (hidden dim 2560). We extract sub-modules (`audio_encoder`, `audio_adapter`, `language_model`) from the loaded `MossAudioModel` rather than calling its generative `forward()`.
_Avoid_: MOSS, audio model, backbone

**Amy Classifier Head**:
The issue #8 pilot model: `AmyForProsodyClassification`. Wraps MOSS-Audio sub-modules with FACodec embedding tables and a classification head. Forward path: audio encoder → audio adapter → ResidualFusion(prosody stream) → Qwen3 language model → mean-pool frames → Linear(2560→2). No text tokens, no DeepStack, no LM head. Trained with CrossEntropyLoss for binary sarcasm classification.
_Avoid_: classification wrapper, downstream model

### Architecture Concepts

**Semantic Stream (S_t)**:
The "what was said" representation — linguistic content extracted at 12.5 Hz. In the pilot, sourced from MOSS-Audio's encoder projected into LLM embedding space.
_Avoid_: Content, text features, transcript

**Prosody Stream (p_t)**:
Discrete indices encoding pitch, rhythm, and intonation. Sourced from FACodec's prosody codebook at 80 Hz, pooled to 12.5 Hz via average pooling after embedding.
_Avoid_: Pitch, tone (overloaded), F0 features

**Acoustic Stream (a_t)**:
Discrete residual acoustic detail indices from FACodec's three acoustic codebooks at 80 Hz. Captures speaker/environment/reconstruction artifacts beyond prosody. Optional for the pilot; not to be conflated with Timbre Vector.
_Avoid_: Residual, detail, timbre indices

**FACodec Content Stream (c_t)**:
FACodec's discrete linguistic-content codebooks at 80 Hz. A competing semantic signal, distinct from Semantic Stream (which comes from MOSS-Audio). Disabled in the initial experiment; enabled only in later ablation runs to compare codec-derived vs. model-derived semantics.
_Avoid_: Content indices, FACodec semantics (ambiguous with Semantic Stream)

**Timbre Vector**:
A single global utterance-level embedding representing speaker identity. Sourced from FACodec's timbre encoder, not per-frame.
_Avoid_: Speaker embedding, voice print, speaker ID

**Residual Summation**:
The frame-level additive fusion operation: `LayerNorm(S_t + λ_p·P_t + λ_a·A_t + λ_c·C_t + λ_t·T_t)`. Each stream occupies the same embedding dimension as S_t and contributes marginal signal gated by its own learnable λ. Prosody/Acoustic/Content/Timbre are added to semantics like positional encodings are added to token embeddings.
_Avoid_: Addition, injection (may be confused with DeepStack mechanism), concat fusion

**MOSS-Audio Internal Residual Extension**:
The Amy LM pilot architecture for issue #8: MOSS-Audio still encodes audio at 12.5 Hz, then Residual Summation is applied after MOSS-Audio's audio embedding layer and before the normal transformer forward path. The rest of MOSS-Audio's forward pass remains unchanged.
_Avoid_: External wrapper, post-hoc fusion

**Stream Dimensionality Contract**:
Every FACodec stream must output the same embedding dimension as MOSS-Audio's hidden dim (D_moss) after embedding/projection. Multi-codebook streams (Content, Acoustic) sum their per-codebook embeddings. TemporalPool aligns all VQ streams from 80 Hz to MOSS frame rate after embedding.
_Avoid_: Concatenating codebooks in the embedding dimension, projecting down before fusion

**Online Semantic Encoding**:
The issue #8 training path: batches load audio and FACodec indices from the preprocessed dataset, then MOSS-Audio computes the Semantic Stream during each forward pass. Semantic frames are not precomputed to disk because they are large and MOSS-Audio encoding is considered fast enough.
_Avoid_: Precomputed semantic frames

**Stream Activation Config**:
A YAML block controlling which FACodec streams participate in fusion: `prosody`, `content`, `acoustic`, `timbre` (each boolean). Modules and forward() build only active streams. Disabled streams are excluded from both module instantiation and the Residual Summation computation, not merely gated at λ=0.
_Avoid_: Freezing gates, masking tensors at runtime

### Architecture Comparison

**Projection Architecture**:
Compressing audio features through a learned projector (e.g., GatedMLP) into a text-aligned LLM embedding space. MOSS-Audio's DeepStack uses this. Hypothesis: fundamentally lossy for prosody.
_Avoid_: Text projection, modality adapter

**Extension Architecture**:
Adding new embedding dimensions that exist explicitly for prosody/timbre, not mediated through text. Amy LM uses this via FACodec embedding tables. Hypothesis: preserves acoustic structure that text-aligned spaces lose.
_Avoid_: Embedding expansion, modality extension

### Training

**Training Loop**:
Vanilla PyTorch (no Lightning, no HF Trainer). Single optimizer, single forward/backward per step. Chosen over PyTorch Lightning because (1) no GAN dual-optimizer complexity, (2) avoids Lightning's memory overhead with the 4B Qwen3 backbone, (3) the loop is ~50 lines and easier to debug.
_Avoid_: Trainer, LightningModule

**MUStARD Formulation**:
Binary sarcasm classification. Input: raw audio waveform + FACodec prosody indices (from preprocessing). MOSS-Audio computes mel spectrograms internally. Output: 2-class logits trained with CrossEntropyLoss.
_Avoid_: multi-label, multi-class (binary classification only in the simplest training row)

**λ (Lambda)**:
A family of learnable per-stream scalar gates: λ_p (Prosody), λ_a (Acoustic), λ_c (FACodec Content), λ_t (Timbre). Each initialized at zero. Zero-init guarantees the model equals MOSS-Audio at step 0. Individual gates enable clean ablation — freeze a gate at zero to disable its stream.
_Avoid_: Alpha, weight, scale factor

**Social Deafness**:
The failure mode where a speech model correctly transcribes words but misses emotional/tonal implication (e.g., "I'm fine" spoken with distress).
_Avoid_: Prosody blindness, tone deafness

**Hypothesis Matrix**:
Multi-dimensional experiment design. 3 embedding init strategies (random, FACodec warm-start, continuous projector) × 3 training strategies (frozen, LoRA, full fine-tune) × 3 losses (classification, LM loss, combined) — evaluated across 8 benchmarks.
_Avoid_: Ablation grid, experiment table

## Relationships

- **Amy LM** uses **MOSS-Audio** as its semantic backbone and **FACodec** for optional Prosody, Content, Acoustic, and Timbre streams
- **FACodec** substitutes for **Amy Codec** during pilot validation
- **Semantic Stream**, **Prosody Stream**, **Acoustic Stream**, and **FACodec Content Stream** are fused via **Residual Summation** at 12.5 Hz; each has an independent learnable gate
- **Stream Activation Config** controls which streams are built and fused; disabled streams are excluded from both module instantiation and forward()
- **Stream Dimensionality Contract** requires all FACodec streams to output the same embedding dimension as MOSS-Audio's hidden dim; multi-codebook streams sum per-codebook embeddings
- **MOSS-Audio Internal Residual Extension** places **Residual Summation** inside MOSS-Audio after the audio embedding layer, not outside the model as a post-processing wrapper
- **Online Semantic Encoding** means issue #8 uses MOSS-Audio to compute the **Semantic Stream** per batch, while FACodec-derived indices are loaded from preprocessing output
- **Timbre Vector** is broadcast to all frames of an utterance and never passes through TemporalPool
- **Prosody Stream**, **Acoustic Stream**, and **FACodec Content Stream** each pass through **TemporalPool** (80 Hz → MOSS frame rate) after embedding
- **Projection Architecture** and **Extension Architecture** are competing hypotheses for how to represent speech in LLMs
- **Social Deafness** is the problem; **Hypothesis Matrix** is the evaluation framework

## Example dialogue

> **Dev:** "For the pilot, are we using FACodec's content codebooks for the Semantic Stream, or MOSS-Audio's encoder?"
> **Domain expert:** "MOSS-Audio. FACodec's content codebooks are a separate **FACodec Content Stream** — disabled in the first experiment. We start with Prosody-only to isolate the social-prosody hypothesis."

> **Dev:** "When we say Extension, where exactly does the injection happen?"
> **Domain expert:** "At the LLM input, before the first transformer layer — same as positional embeddings. Each stream has its own λ gate, all zero-init so at step 0 the model is literally just MOSS-Audio."

> **Dev:** "If λ_p stays near zero after training, is that a failure?"
> **Domain expert:** "Depends. If benchmarks improve, the embedding tables learned useful structure and λ_p acts as a normalizer. If nothing changes, the hypothesis is falsified — prosody signals from FACodec didn't add anything MOSS-Audio doesn't already have."

> **Dev:** "What does the #6 preprocessing pipeline store for Acoustic Stream?"
> **Domain expert:** "Three codebook indices per frame: `acoustic_codebooks_idx` with shape `[3, T80]`. That's the raw VQ output. Embedding and pooling happen in the model, not in preprocessing."

> **Dev:** "Is Timbre Vector the same thing as the old `timbre_codebooks_idx` field?"
> **Domain expert:** "No. `timbre_codebooks_idx` was a mistake — it stored averaged residual acoustic VQ indices under the wrong name. Timbre Vector is a separate continuous embedding from FACodec's `spk_embs`, utterance-level, float32."

## Flagged ambiguities

- "Encoder" was used for both the audio encoder (speech → features) and the text encoder (tokens → embeddings), and for the neural audio codec — resolved: use "audio encoder" or "speech encoder" for the former, "LLM backbone" or "token embedder" for the latter, and "Amy Codec" / "FACodec" for the codec.
- "Injection" blurred the line between DeepStack's mid-layer summation and input-level residual summation — resolved: Residual Summation is the canonical term for the Amy LM approach.
- "timbre_codebooks_idx" stored averaged residual acoustic VQ indices under the wrong name — resolved: renamed to **Acoustic Stream** (a_t). The true **Timbre Vector** is a separate continuous utterance-level embedding from FACodec `spk_embs`.
