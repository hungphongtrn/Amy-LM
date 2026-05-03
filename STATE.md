# Amy-LM Project State

**Project**: The Semantic Bridge - Disentangled Neural Audio Codec
**Last Updated**: 2026-05-01
**Current Phase**: Adversarial Fine-tuning (Epoch 10)

---

## Architecture Overview

A 9-codebook modification of [Mimi](https://github.com/kyutai-labs/mimi) audio codec:

| Codebook | Content                            | Alignment Target                    | Status                           |
| -------- | ---------------------------------- | ----------------------------------- | -------------------------------- |
| 0        | Semantic content                   | Qwen2.5-1.5B Layer 27 hidden states | Working well (~91.4% similarity) |
| 1        | Prosody/pitch/rhythm               | WavLM features                      | Struggling (~30.7% similarity)   |
| 2-8      | Acoustic texture (timbre, speaker) | Reconstruction                      | Baseline stable                  |

**Key Innovation**: Enables zero-shot voice cloning and text-based audio editing by mixing codebooks from different sources.

---

## Training Status (Epoch 10 - Dec 25, 2024)

### Configuration

- `adversarial_only=True` - multi-scale mel spectrogram loss disabled
- Generator weight surgery applied (bypass linear projections for CB 0)
- Discriminator training enabled

### Key Metrics

| Metric                  | Value          | Target  | Assessment                |
| ----------------------- | -------------- | ------- | ------------------------- |
| LLM cosine similarity   | ~0.914         | > 0.85  | Strong semantic alignment |
| WavLM cosine similarity | ~0.307         | > 0.70  | Poor prosody separation   |
| Generator loss          | 2.7-4.6        | < 3.0   | Acceptable but high       |
| Adversarial loss %      | ~65% of total  | ~30-40% | Over-dominant             |
| WavLM loss %            | ~69% distance  | < 10%   | Needs improvement         |
| LLM loss %              | ~8.6% distance | < 10%   | On target                 |
| Discriminator loss      | ~1.7-1.8       | < 2.0   | Healthy balance           |

### Critical Issues Identified

1. **WavLM distillation is failing**: ~0.307 cosine similarity is far below target. The model cannot map prosody features effectively.
2. **Adversarial loss dominance**: At ~65% of total generator loss, the GAN objective is overwhelming distillation losses.
3. **Linear projection limitation**: Current simple linear mappings may not capture complex WavLM-to-prosody relationships.

---

## Recommended Actions

Based on training analysis:

1. **Re-enable multi-scale mel spectrogram loss** - Currently disabled via `adversarial_only=True`, but provides crucial gradient signal
2. **Add MLPs to WavLM projections** - Replace linear layers with deeper networks to better map rich WavLM features to compact prosody codebook
3. **Balance loss weights** - Reduce adversarial loss weight or increase distillation weights
4. **Increase WavLM loss weight** - From 0.5 to 1.0 or higher to prioritize prosody capture
5. **Implement qualitative monitoring** - Add W&B audio samples for subjective verification

---

## Benchmark Development

**Recently Completed**: Prosody Validation Benchmark

A comprehensive pipeline for testing whether speech models understand prosody when text is neutralized:

| Step   | Description                                   | Input                                      | Output                         |
| ------ | --------------------------------------------- | ------------------------------------------ | ------------------------------ |
| Step 0 | Merge and sample data                         | `data/(2000 samples) merged_output.xlsx` | `data/step0_sampled_200.csv` |
| Step 1 | Rewrite text to neutral                       | `data/step0_sampled_200.csv`             | `data/step1_rewritten.csv`   |
| Step 2 | Generate prosody-guided speech with Qwen3-TTS | `data/step1_rewritten.csv`               | `outputs/audio/*.wav`        |
| Step 3 | Text-only baseline (GPT-4o-mini)              | `data/step1_rewritten.csv`               | `step3_text_responses.jsonl` |
| Step 4 | ASR pipeline (Whisper → GPT-4o-mini)         | `outputs/audio/*.wav`                    | `step4_asr_responses.jsonl`  |
| Step 5 | End-to-end audio (Gemini 2.5 Flash)           | `outputs/audio/*.wav`                    | `step5_e2e_responses.jsonl`  |
| Step 6 | Evaluation with GPT-4o                        | All responses                              | `evaluation_summary.json`    |

**Status**: 200 audio samples generated, pipeline complete, evaluations run

---

## Target Specifications

| Specification                 | Target              |
| ----------------------------- | ------------------- |
| Frame rate                    | 12.5 Hz             |
| Bitrate                       | ~1.2 kbps           |
| WER (Codebooks 0+1 only)      | < 5%                |
| WavLM similarity (Codebook 1) | > 70%               |
| Inference speed               | Real-time on 1x A10 |

---

## Key Files

| File                                              | Description                                          |
| ------------------------------------------------- | ---------------------------------------------------- |
| `docs/ideation/proposal.md`                     | Research proposal with methodology                   |
| `docs/ideation/amy.md`                          | Architecture details and CausalBridgeQuantizer class |
| `docs/training_records/20_12_25_14_40.md`       | Epoch 10 training log with full metrics              |
| `src/models/mimi/quantization/causal_bridge.py` | BridgeQuantizer implementation                       |
| `benchmark/prosody_validation/`                 | Complete evaluation pipeline                         |
| `train.py`                                      | Main training script                                 |

---

## Next Steps

1. Implement MLPs for WavLM projections
2. Adjust loss weight balance
3. Re-enable multi-scale mel spectrogram loss
4. Continue adversarial fine-tuning
5. Monitor qualitative audio samples via W&B

---

## Open Questions

- Will deeper networks (MLPs) resolve WavLM distillation issues?
- How to prevent adversarial loss from overwhelming distillation?
- What is the minimum viable prosody similarity threshold?
- Can voice cloning quality be assessed without human evaluation?

---

*This state file reflects the project as of the latest training record (Dec 25, 2024) and recent benchmark development.*
