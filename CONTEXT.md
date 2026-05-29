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
An open-source audio understanding model (OpenMOSS, Apache 2.0). 4B variant used as the semantic backbone. Architecture: Whisper-style audio encoder → GatedMLP adapter → Qwen3 language model (hidden dim 2560). Source vendored as `src/models/moss_audio_model.py` (copied from `vendor/MOSS-Audio/src/`) to enable Liger kernel patching and flash attention config without vendor path dependencies. Composed as `self.moss` in **AmyMossLM**, not inherited.
_Avoid_: MOSS, audio model, backbone

**Amy Classifier Head**:
The issue #8 pilot model: `AmyForProsodyClassification`. Composes (HAS-A) an `AmyMossLM` as `self.amy_moss` for audio encoding and FACodec enrichment, with a classification head on top. Forward path: `amy_moss.encode_enriched_audio_embeds()` → Qwen3 language model → mean-pool frames → Linear(2560→2). No text tokens, no DeepStack, no LM head. Trained with CrossEntropyLoss for binary sarcasm classification. FACodec modules owned by `self.amy_moss` (no duplicate modules).
_Avoid_: classification wrapper, downstream model

**AmyMossLM**:
The HF-compatible Speech Language Model (issues #14, #26). A standalone `PreTrainedModel` + `GenerationMixin` that **composes** (HAS-A) a `MossAudioModel` as `self.moss`, adding `ProsodyEmbedding`, `TimbreProjection`, `TemporalPool`, and `ResidualFusion` as sibling attributes. Forward delegates backbone work to `self.moss` methods (`get_audio_features`, `audio_adapter`, `language_model`, `lm_head`, `_register_llm_deepstack_hooks`) while injecting FACodec enrichment between `audio_adapter` and `masked_scatter_`. Supports `save_pretrained`/`from_pretrained`/`generate()` and HF `DPOTrainer` integration. Base checkpoint bootstrapped via `prepare_base_checkpoint()`: loads MossAudio weights, wraps them in `self.moss` (getting `moss.*` key prefix naturally via PyTorch child naming), saves full model to `hungphongtrn/amy-moss-lm-base` on HF Hub. Trainable components: FACodec embedding projector, timbre projection, λ gates. Backbone frozen by default — LoRA applied via `target_modules` regex scoped to `moss.*` prefix.
_Avoid_: AmyLM (replaced by AmyMossLM), wrapper model, `__class__` mutation, `_upgrade_from_moss`

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
The frame-level additive fusion operation: `LayerNorm(S_t + Σ λ_i·LayerNorm_i(stream_i))`. Each stream is normalized via its own per-stream LayerNorm before gate scaling so contributions have unit magnitude regardless of raw embedding scale. Each stream occupies the same embedding dimension as S_t and contributes marginal signal gated by its own learnable λ. Prosody/Acoustic/Content/Timbre are added to semantics like positional encodings are added to token embeddings.
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
A configuration block (currently a Python dict in `AmyForProsodyClassification`, planned as a `AmyLMConfig` field for AmyLM) controlling which FACodec streams participate in fusion: `prosody`, `content`, `acoustic`, `timbre` (each boolean). Modules and forward() build only active streams. Disabled streams are excluded from both module instantiation and the Residual Summation computation, not merely gated at λ=0.
_Avoid_: Freezing gates, masking tensors at runtime

### Architecture Comparison

**Projection Architecture**:
Compressing audio features through a learned projector (e.g., GatedMLP) into a text-aligned LLM embedding space. MOSS-Audio's DeepStack uses this. Hypothesis: fundamentally lossy for prosody.
_Avoid_: Text projection, modality adapter

**Extension Architecture**:
Adding new embedding dimensions that exist explicitly for prosody/timbre, not mediated through text. Amy LM uses this via FACodec embedding tables. Hypothesis: preserves acoustic structure that text-aligned spaces lose.
_Avoid_: Embedding expansion, modality extension

### Training

**Classification Training Loop**:
Vanilla PyTorch (no Lightning, no HF Trainer). Single optimizer, single forward/backward per step. Used for issue #8 classification probe (`AmyForProsodyClassification`). Chosen over PyTorch Lightning because (1) no GAN dual-optimizer complexity, (2) avoids Lightning's memory overhead with the 4B Qwen3 backbone, (3) the loop is ~50 lines and easier to debug.
_Avoid_: Trainer, LightningModule

**DPO (Direct Preference Optimization)**:
Generative training formulation (issue #14). Trains AmyLM to assign higher log-probability to a context-aware "chosen" response than a literal-interpretation "rejected" response, given the same speech input. Uses HF `DPOTrainer` with a frozen reference model. DPO loss: `-log(sigmoid(β * (log_p_chosen - log_p_rejected)))`. Default β=0.1. Trainable parameters: FACodec embedding projector, timbre projection, λ gates. Backbone frozen.
_Avoid_: contrastive loss, preference loss (ambiguous — use DPO specifically)

**MUStARD Formulation**:
Binary sarcasm classification (issue #8 probe). Input: raw audio waveform + FACodec prosody indices (from preprocessing). MOSS-Audio computes mel spectrograms internally. Output: 2-class logits trained with CrossEntropyLoss.
_Avoid_: multi-label, multi-class (binary classification only in the simplest training row)

### Data

**Preference Pair**:
A DPO training sample: `(audio, chosen_response, rejected_response)` where:
- `chosen` = context-aware response accounting for prosody, emotion, and speaker identity
- `rejected` = literal-interpretation response from a bare transcript with no paralinguistic context
Generated by an external LLM (DeepSeek V4 Flash) via rich vs. stripped prompt templates, then filtered by cosine similarity between the two responses. The speech signal is the sole disambiguation source — AmyLM receives only the audio and a uniform system prompt; no sample-specific text cues.
_Avoid_: good/bad pair, contrastive sample, ranked pair

**NVTTS**:
The NonverbalTTS corpus (`deepvk/NonverbalTTS`, 17h, English, Apache 2.0). Rich speech from VoxCeleb + Expresso with word-level inline paralinguistic vocalization annotations (10 NV types expressed as emoji tags in the `Result` column) and 8 emotion categories. Used as the source for preference pair construction. NV tags are mapped from emoji to text labels (`[Laughter]`, `[Breathing]`, etc.) before use in LLM prompts. Split structure: train (3641), dev (46), test (359) — 4046 total samples, audio at 48kHz decoded via torchcodec.
_Avoid_: NonverbalTTS (use NVTTS), NV dataset

**NVTTS-FACodec Dataset**:
The output of the preprocessing pipeline (issue #21): NVTTS audio encoded through FACodec into all 4 streams (prosody, content, acoustic, timbre) and joined with preference pair annotations. Schema: `index`, `audio`, `prosody_codebooks_idx`, `content_codebooks_idx`, `acoustic_codebooks_idx`, `timbre_vector`, `chosen`, `rejected`, `cosine_similarity`. Single FACodec forward pass produces all streams — storing everything enables downstream experiments to cherry-pick streams (e.g., prosody+timbre for DPO) without re-encoding. Preserves NVTTS train/dev/test splits. Pushed to HF Hub with `datasets.push_to_hub()`.
_Avoid_: NVTTS preference dataset (ambiguous — the preference pairs alone, without FACodec encoding)

**LLM-Generated Preference Pair**:
Dataset construction strategy: an external LLM generates candidate chosen/rejected responses from a rich paralinguistic prompt vs. a stripped literal prompt. Pairs are embedded and filtered by cosine similarity to retain only pairs where paralinguistic context produced meaningfully different responses. Three-phase pipeline: (1) speaker context enrichment, (2) LLM pair generation, (3) embedding + cosine filter.
_Avoid_: synthetic data (the audio is real, only the responses are LLM-generated), rule-based negative mining

### DPO Training (Issue #24)

**DPOCollator**:
Custom HF data collator for AmyLM DPO training. Converts raw `nvtts_facodec` rows into DPO batch dicts: extracts mel spectrograms from raw audio via MOSS-Audio's `_extract_mel()`, tokenizes system prompt + `<audio>` placeholder + chosen/rejected responses, computes `audio_input_mask` from `<|AUDIO|>` token positions, pads `prosody_indices` and `timbre_vector` across batch. Outputs standard DPO fields (`chosen_input_ids`, `chosen_labels`, `rejected_input_ids`, `rejected_labels`) plus extra fields (`audio_data`, `audio_data_seqlens`, `audio_input_mask`, `prosody_indices`, `timbre_vector`). Prompt tokens masked with -100 in labels; only response tokens contribute to log-prob computation.
_Avoid_: DPO data loader, preference collator

**AmyDPOTrainer**:
Subclass of `trl.DPOTrainer` for AmyLM. Overrides `concatenated_forward()` to duplicate non-text inputs (`audio_data`, `audo_data_seqlens`, `audio_input_mask`, `prosody_indices`, `timbre_vector`) along the batch dimension (B → 2*B) before calling `model()`. Same audio/prosody/timbre for chosen and rejected since they share the speech input. Policy model trained with LoRA on a bf16 MOSS-Audio backbone + fp32 FACodec modules; reference model is a frozen AmyLM snapshot at initialization (zero λ → equivalent to MOSS-Audio). Gradients flow only through FACodec modules and LoRA adapters; audio encoder, audio adapter, and LM head frozen.
_Avoid_: preference trainer, contrastive trainer

**Concatenated Forward (AmyLM DPO)**:
The batch-doubling strategy for DPO with speech: standard `DPOTrainer.concatenated_inputs()` stacks chosen and rejected `input_ids`/`attention_mask`/`labels` [B, S] → [2*B, S] along the batch dimension. `AmyDPOTrainer` does the same for non-text inputs: `audio_data` [B, 128, T_mel] → [2*B, 128, T_mel], `prosody_indices` [B, 1, T80] → [2*B, 1, T80], `timbre_vector` [B, 256] → [2*B, 256], `audio_input_mask` [B, S] → [2*B, S]. After `model()`, logits are split back into `chosen_logits` (indices 0:B) and `rejected_logits` (B:2B) for DPO loss computation.
_Avoid_: double pass, separate forward

**DPO Reference Model**:
A frozen copy of AmyLM at training step 0 (same architecture, same random weights, zero λ gates). Since λ gates are zero-initialized and FACodec modules are randomly initialized, the reference model is functionally equivalent to MOSS-Audio at step 0. DPO reward = β × (log(policy_logp_chosen/reference_logp_chosen) − log(policy_logp_rejected/reference_logp_rejected)). The reward isolates the FACodec module's contribution — how much the learned prosody/timbre enrichment improves chosen vs. rejected preference over the MOSS-Audio baseline.
_Avoid_: base model, pre-trained reference

**DPO System Prompt**:
The uniform prompt used for all DPO training samples: "You are a helpful assistant. Listen carefully to the speaker's tone and respond appropriately to the following speech: `<audo>`". The `<audio>` placeholder (regex `<|audio_bos|>(?:<|AUDIO|>)+<|audio_eos|>`) is expanded to N × `<|AUDIO|>` tokens (12.5 per second of audio) by MOSS-Audio's processor. No sample-specific text cues — the model must derive prosody and timbre from the speech signal alone.
_Avoid_: instruction prompt, task prompt

**LoRA (DPO)**:
bf16 MOSS-Audio backbone with LoRA adapters on all Qwen3 linear layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `up_proj`, `down_proj`, `gate_proj`). LoRA rank, alpha, and dropout are configurable hyperparameters. FACodec modules (prosody_embedding, timbre_projection, λ gates) train in fp32. Uses bf16 mixed precision with gradient checkpointing on Qwen3 backbone. Optimizer states may use `paged_adamw_8bit`; the backbone itself is not 4-bit quantized.
_Avoid_: full fine-tune, QLoRA, 4-bit backbone

**NVTTS-FACodec DPO Split**:
Training data pipeline: load `hungphongtrn/nvtts_facodec` → filter by `cosine_similarity < threshold` (default 0.85) → use existing NVTTS splits (train: 3641, dev: 46, test: 359). Content and acoustic streams present in dataset but excluded from DPO collation. Threshold is a configurable hyperparameter; the full dataset is kept intact, split applied post-filter.
_Avoid_: random split, hard train/val boundary

### Evaluation

**LLM-as-Judge**:
Evaluation protocol comparing AmyLM's response quality before vs. after DPO training. 50 held-out test samples (lowest cosine similarity — hardest cases). For each sample, the pre-training and post-training models generate responses given `<audio>` + uniform system prompt. An external LLM (DeepSeek V4 Flash) receives both responses plus the ground-truth paralinguistic context and judges which is more emotionally appropriate and speaker-aware. Metric: win rate = post-training wins / non-tie comparisons.
_Avoid_: human eval (it's LLM-judged, not human-judged), automated metrics

**λ (Lambda)**:
A family of learnable per-stream scalar gates: λ_p (Prosody), λ_a (Acoustic), λ_c (FACodec Content), λ_t (Timbre). Initialized at 1.0 in `ResidualFusion` so FACodec modules receive gradient from epoch 1 (zero-init deadlocks FACodec training because `dL/d(FACodec_weights) ∝ λ`). For the classification proof (#31), λ is kept at unit magnitude (effective no-gate additive fusion with per-stream LayerNorm). Individual gates enable clean ablation — freeze a gate at zero to disable its stream.
_Avoid_: Alpha, weight, scale factor

**FACodec Shuffle Control**:
Negative control for the claim "FACodec P+T streams contain useful classification signal." Each audio sample is paired with prosody/timbre indices from a different training sample in the same split (derangement). Same model architecture, same parameter count, same LayerNorms, same λ. Only the alignment between speech and FACodec features is broken. If Amy true > Amy shuffled, the aligned FACodec signal is useful. If Amy true ≈ Amy shuffled > Baseline, the gain is from extra capacity/regularization, not FACodec signal.
_Avoid_: Cross-sample noise, mismatched features

**Social Deafness**:
The failure mode where a speech model correctly transcribes words but misses emotional/tonal implication (e.g., "I'm fine" spoken with distress).
_Avoid_: Prosody blindness, tone deafness

**Hypothesis Matrix**:
Multi-dimensional experiment design. 3 embedding init strategies (random, FACodec warm-start, continuous projector) × 3 training strategies (frozen, LoRA, full fine-tune) × 3 losses (classification, LM loss, combined) — evaluated across 8 benchmarks.
_Avoid_: Ablation grid, experiment table

## Relationships

- **Amy LM** uses **MOSS-Audio** as its semantic backbone and **FACodec** for optional Prosody, Content, Acoustic, and Timbre streams
- **AmyMossLM** (HF model class) **composes** **MossAudioModel** as `self.moss`, adding FACodec enrichment modules as sibling attributes — no inheritance, no `__class__` mutation
- **AmyMossLM** lives alongside **MossAudioModel** source in `src/models/` (no vendor path dependency at runtime); Liger kernel applied globally via `apply_liger_kernel_to_qwen3()` before model construction; flash attention configured via `language_config._attn_implementation`
- **Amy Classifier Head** is the issue #8 classification probe; **AmyMossLM** is the issue #14/26 generative model — distinct models with different forward paths
- **FACodec** substitutes for **Amy Codec** during pilot validation
- **Semantic Stream**, **Prosody Stream**, **Acoustic Stream**, and **FACodec Content Stream** are fused via **Residual Summation** at 12.5 Hz; each has an independent learnable gate
- **Stream Activation Config** controls which streams are built and fused; disabled streams are excluded from both module instantiation and forward()
- **Stream Dimensionality Contract** requires all FACodec streams to output the same embedding dimension as MOSS-Audio's hidden dim; multi-codebook streams sum per-codebook embeddings
- **MOSS-Audio Internal Residual Extension** places **Residual Summation** inside MOSS-Audio after the audio embedding layer, not outside the model as a post-processing wrapper
- **Online Semantic Encoding** means issue #8 uses MOSS-Audio to compute the **Semantic Stream** per batch, while FACodec-derived indices are loaded from preprocessing output
- **Timbre Vector** is broadcast to all frames of an utterance and never passes through TemporalPool
- **Prosody Stream**, **Acoustic Stream**, and **FACodec Content Stream** each pass through **TemporalPool** (80 Hz → MOSS frame rate) after embedding
- **Projection Architecture** and **Extension Architecture** are competing hypotheses for how to represent speech in LLMs
- **FACodec Shuffle Control** is the negative control for the aligned FACodec signal claim; it uses the same model architecture with per-sample prosody/timbre derangement within each split
- **Preference Pairs** are constructed by an external LLM from **NVTTS** speech, then encoded through the **FACodec** preprocessing pipeline to produce the **NVTTS-FACodec Dataset** before **DPO** training
- **DPO** trains **AmyMossLM** via **AmyDPOTrainer** (subclass of `trl.DPOTrainer`), computing per-token log-probabilities of chosen vs. rejected responses on Qwen3's full vocabulary
- **DPOCollator** produces batches from **NVTTS-FACodec Dataset** rows; **AmyDPOTrainer** uses **Concatenated Forward (AmyLM DPO)** to pass shared audio/prosody/timbre through duplicated batch dimension
- **DPO Reference Model** is a frozen **AmyMossLM** snapshot at step 0; λ=0 makes it functionally equivalent to **MOSS-Audio** at training start
- **LoRA (DPO)** freezes the bf16 **MOSS-Audio** backbone and trains LoRA adapters + fp32 FACodec modules; LoRA `target_modules` uses regex scoped to `moss.*` prefix to avoid touching FACodec linear layers
- **NVTTS-FACodec DPO Split** applies cosine similarity filter to the full **NVTTS-FACodec Dataset** while preserving original train/dev/test splits
- **DPO System Prompt** is uniform across all samples; the model receives no sample-specific text cues
- **LLM-as-Judge** evaluates **Social Deafness** improvement by comparing pre/post **DPO** response appropriateness on held-out samples
- **AmyMossLM Base Checkpoint** (`hungphongtrn/amy-moss-lm-base`): One-time bootstrap through `AmyMossLM.prepare_base_checkpoint()` — loads MossAudio model weights, wraps in `self.moss` composition, zero-inits FACodec modules, saves full safetensors + processor to HF Hub. Subsequent training uses `AmyMossLM.from_pretrained("hungphongtrn/amy-moss-lm-base")`.

## Example dialogue

> **Dev:** "For the pilot, are we using FACodec's content codebooks for the Semantic Stream, or MOSS-Audio's encoder?"
> **Domain expert:** "MOSS-Audio. FACodec's content codebooks are a separate **FACodec Content Stream** — disabled in the first experiment. We start with Prosody-only to isolate the social-prosody hypothesis."

> **Dev:** "When we say Extension, where exactly does the injection happen?"
> **Domain expert:** "At the LLM input, before the first transformer layer — same as positional embeddings. Per-stream LayerNorm normalizes each FACodec stream before gated summation."

> **Dev:** "If λ_p stays near zero after training, is that a failure?"
> **Domain expert:** "Depends. If benchmarks improve, the embedding tables learned useful structure and λ_p acts as a normalizer. If nothing changes, the hypothesis is falsified — prosody signals from FACodec didn't add anything MOSS-Audio doesn't already have."

> **Dev:** "What does the #6 preprocessing pipeline store for Acoustic Stream?"
> **Domain expert:** "Three codebook indices per frame: `acoustic_codebooks_idx` with shape `[3, T80]`. That's the raw VQ output. Embedding and pooling happen in the model, not in preprocessing."

> **Dev:** "Is Timbre Vector the same thing as the old `timbre_codebooks_idx` field?"
> **Domain expert:** "No. `timbre_codebooks_idx` was a mistake — it stored averaged residual acoustic VQ indices under the wrong name. Timbre Vector is a separate continuous embedding from FACodec's `spk_embs`, utterance-level, float32."

> **Dev:** "Does AmyLM see the emotion label or speaker name during DPO training?"
> **Domain expert:** "No. The prompt is just a uniform system prompt + `<audio>` placeholder. The model must derive prosody and timbre understanding exclusively from the speech signal. The rich context was only used by the external LLM to *generate* the chosen/rejected pairs — it's not in the training data."

> **Dev:** "What's the difference between AmyForProsodyClassification and AmyLM?"
> **Domain expert:** "The classifier is a probe — it extracts MOSS-Audio sub-modules, wraps them, mean-pools the output, and classifies. AmyLM is a proper HF model inheriting MossAudioModel directly, replacing the classification head with Qwen3's full generative forward path. Classification was Step 1; AmyLM + DPO is Step 2."

## Flagged ambiguities

- "Encoder" was used for both the audio encoder (speech → features) and the text encoder (tokens → embeddings), and for the neural audio codec — resolved: use "audio encoder" or "speech encoder" for the former, "LLM backbone" or "token embedder" for the latter, and "Amy Codec" / "FACodec" for the codec.
- "Injection" blurred the line between DeepStack's mid-layer summation and input-level residual summation — resolved: Residual Summation is the canonical term for the Amy LM approach.
- "timbre_codebooks_idx" stored averaged residual acoustic VQ indices under the wrong name — resolved: renamed to **Acoustic Stream** (a_t). The true **Timbre Vector** is a separate continuous utterance-level embedding from FACodec `spk_embs`.
- NVTTS paralinguistic tags are emoji symbols (🌬️, 🤣, 😷) in the `Result` column, not text labels like `[Breathing]` — resolved: NV Tag Emoji Mapping converts emojis to `[Tag]` text labels before use in LLM prompts.
