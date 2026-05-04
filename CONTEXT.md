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
An open-source audio understanding model (OpenMOSS, Apache 2.0). 4B variant used as the semantic backbone for Amy LM. Produces continuous encoder features at 12.5 Hz with DeepStack cross-layer injection.
_Avoid_: MOSS, audio model, backbone

### Architecture Concepts

**Semantic Stream (S_t)**:
The "what was said" representation — linguistic content extracted at 12.5 Hz. In the pilot, sourced from MOSS-Audio's encoder projected into LLM embedding space.
_Avoid_: Content, text features, transcript

**Prosody Stream (p_t)**:
Discrete indices encoding pitch, rhythm, and intonation. Sourced from FACodec's prosody codebook at 80 Hz, pooled to 12.5 Hz via average pooling after embedding.
_Avoid_: Pitch, tone (overloaded), F0 features

**Timbre Vector**:
A single global utterance-level embedding representing speaker identity. Sourced from FACodec's timbre encoder, not per-frame.
_Avoid_: Speaker embedding, voice print, speaker ID

**Residual Summation**:
The frame-level fusion operation: `Projector(S_t) + λ · (E_prosody[p_t] + h_timbre)`. Prosody/timbre are added to semantics like positional encodings are added to token embeddings.
_Avoid_: Addition, injection (may be confused with DeepStack mechanism)

### Architecture Comparison

**Projection Architecture**:
Compressing audio features through a learned projector (e.g., GatedMLP) into a text-aligned LLM embedding space. MOSS-Audio's DeepStack uses this. Hypothesis: fundamentally lossy for prosody.
_Avoid_: Text projection, modality adapter

**Extension Architecture**:
Adding new embedding dimensions that exist explicitly for prosody/timbre, not mediated through text. Amy LM uses this via FACodec embedding tables. Hypothesis: preserves acoustic structure that text-aligned spaces lose.
_Avoid_: Embedding expansion, modality extension

### Training

**λ (Lambda)**:
A learnable scalar gate initialized at zero. Controls how much prosody/timbre signal enters the fused representation. Zero-init guarantees the model equals MOSS-Audio at step 0.
_Avoid_: Alpha, weight, scale factor

**Social Deafness**:
The failure mode where a speech model correctly transcribes words but misses emotional/tonal implication (e.g., "I'm fine" spoken with distress).
_Avoid_: Prosody blindness, tone deafness

**Hypothesis Matrix**:
Multi-dimensional experiment design. 3 embedding init strategies (random, FACodec warm-start, continuous projector) × 3 training strategies (frozen, LoRA, full fine-tune) × 3 losses (classification, LM loss, combined) — evaluated across 8 benchmarks.
_Avoid_: Ablation grid, experiment table

## Relationships

- **Amy LM** uses **MOSS-Audio** as its semantic backbone and optionally **FACodec** for prosody/timbre
- **FACodec** substitutes for **Amy Codec** during pilot validation
- **Semantic Stream** and **Prosody Stream** are fused via **Residual Summation** at 12.5 Hz
- **Timbre Vector** is broadcast to all frames of an utterance
- **λ** gates the contribution of Prosody Stream + Timbre Vector into the fused representation
- **Projection Architecture** and **Extension Architecture** are competing hypotheses for how to represent speech in LLMs
- **Social Deafness** is the problem; **Hypothesis Matrix** is the evaluation framework

## Example dialogue

> **Dev:** "For the pilot, are we using FACodec's content codebooks for the Semantic Stream, or MOSS-Audio's encoder?"
> **Domain expert:** "MOSS-Audio. FACodec only provides Prosody Stream and Timbre Vector — the Semantic Stream stays with MOSS-Audio. That way we test whether Extension improves an existing understanding model."

> **Dev:** "When we say Extension, where exactly does the injection happen?"
> **Domain expert:** "At the LLM input, before the first transformer layer — same as positional embeddings. λ is zero-init so at step 0 the model is literally just MOSS-Audio."

> **Dev:** "If λ stays near zero after training, is that a failure?"
> **Domain expert:** "Depends. If benchmarks improve, the embedding tables learned useful structure and λ acts as a normalizer. If nothing changes, the hypothesis is falsified — prosody/timbre signals from FACodec didn't add anything MOSS-Audio doesn't already have."

## Flagged ambiguities

- "Encoder" was used to mean both the neural audio codec (Amy Codec) and the speech feature extractor (FACodec encoder, MOSS-Audio encoder) — resolved: use specific model names.
- "Injection" blurred the line between DeepStack's mid-layer summation and input-level residual summation — resolved: Residual Summation is the canonical term for the Amy LM approach.