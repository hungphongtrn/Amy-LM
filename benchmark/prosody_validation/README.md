# Prosody Validation Benchmark Pipeline

A comprehensive benchmark for testing whether speech language models (SLMs) can understand prosody (tone, pitch, emotion) when text is neutralized.

## Overview

This pipeline evaluates how well different approaches can understand the intended meaning and emotion behind text that has been rewritten to be semantically neutral. It compares:

1. **Text Baseline**: Text-only model responding to neutral text
2. **ASR Pipeline**: Transcribe audio with Whisper, then analyze with LLM
3. **End-to-End Audio**: Multimodal LLM processing audio directly

## Directory Structure

```
benchmark/prosody_validation/
├── README.md                    # This file
├── pyproject.toml               # Project configuration (uv)
├── uv.lock                      # Lockfile for reproducible builds
├── .python-version              # Python version specification
├── config.yaml                  # Configuration template
├── src/                         # Shared components
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── openrouter_client.py    # Async OpenRouter API client
│   ├── utils.py                # Utility functions
│   ├── qwen3_tts_wrapper.py    # Qwen3-TTS integration
│   └── whisper_wrapper.py       # Local Whisper integration
├── steps/                       # Pipeline step scripts
│   ├── step0_merge_sample.py
│   ├── step1_rewrite_text.py
│   ├── step2_generate_speech.py
│   ├── step3_text_baseline.py
│   ├── step4_asr_pipeline.py
│   └── step5_e2e_audio.py
├── data/                        # Generated data files
└── outputs/                     # Generated outputs
    ├── audio/                   # Generated .wav files
    └── responses/               # JSONL results
```

## Prerequisites

### 1. Python Environment (using uv)

This project uses [uv](https://docs.astral.sh/uv/) for fast Python package management.

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Navigate to project
cd benchmark/prosody_validation

# Create virtual environment and install dependencies
uv sync

# Or with dev dependencies
uv sync --all-extras
```

### 2. API Keys

Set your OpenRouter API key in the environment:

```bash
export OPENROUTER_API_KEY="your-api-key-here"
```

Or create a `.env` file:

```bash
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

### 3. Qwen3-TTS Setup

Qwen3-TTS should be cloned and set up on your GPU machine:

```bash
git clone https://github.com/QwenLM/Qwen3-TTS.git
cd Qwen3-TTS
# Follow installation instructions in Qwen3-TTS repo
```

Update `config.yaml` with the path to your Qwen3-TTS repository.

## Pipeline Steps

### Step 0: Merge and Sample Data

Merge the annotation and dialogue datasets, filter for valid samples, and select 200 representative utterances.

```bash
cd benchmark/prosody_validation
uv run python -m steps.step0_merge_sample
```

**Input:** `data/(2000 samples) merged_output.xlsx - Annotation.csv` and `... - Dialogue.tsv`
**Output:** `data/step0_sampled_200.csv`

### Step 1: Rewrite Text to Neutral

Use LLM to rewrite utterances to be semantically neutral or positive-leaning.

```bash
cd benchmark/prosody_validation
uv run python -m steps.step1_rewrite_text --max-concurrent 50
```

**Input:** `data/step0_sampled_200.csv`
**Output:** `data/step1_rewritten.csv`

### Step 2: Generate Prosody-Guided Speech

Generate audio files using Qwen3-TTS with prosody guided by emotion/intent labels.

```bash
cd benchmark/prosody_validation
uv run python -m steps.step2_generate_speech --device cuda --batch-size 10
```

**Input:** `data/step1_rewritten.csv`
**Output:** `outputs/audio/{dialog_id}.wav` and `data/step2_audio_manifest.csv`

### Step 3: Text-Only Baseline

Send neutral text to text-only LLM model and collect responses.

```bash
cd benchmark/prosody_validation
uv run python -m steps.step3_text_baseline --model openai/gpt-4o-mini
```

**Input:** `data/step1_rewritten.csv`
**Output:** `outputs/responses/step3_text_responses.jsonl`

### Step 4: ASR Pipeline

Transcribe audio with Whisper, then send transcription to LLM for analysis.

```bash
cd benchmark/prosody_validation
uv run python -m steps.step4_asr_pipeline --llm-model openai/gpt-4o-mini
```

**Input:** `data/step2_audio_manifest.csv`
**Output:** `outputs/responses/step4_asr_responses.jsonl`

### Step 5: End-to-End Audio

Send audio directly to multimodal LLM for understanding.

```bash
cd benchmark/prosody_validation
uv run python -m steps.step5_e2e_audio --model google/gemini-2.5-flash
```

**Input:** `data/step2_audio_manifest.csv`
**Output:** `outputs/responses/step5_e2e_responses.jsonl`

## Running the Complete Pipeline

You can run all steps sequentially:

```bash
cd benchmark/prosody_validation

# Set up environment
export OPENROUTER_API_KEY="your-key"

# Run all steps
uv run python -m steps.step0_merge_sample
uv run python -m steps.step1_rewrite_text
uv run python -m steps.step2_generate_speech --device cuda
uv run python -m steps.step3_text_baseline
uv run python -m steps.step4_asr_pipeline
uv run python -m steps.step5_e2e_audio
```

Or create a shell script:

```bash
#!/bin/bash
export OPENROUTER_API_KEY="your-key"

echo "Running Step 0..."
uv run python -m steps.step0_merge_sample

echo "Running Step 1..."
uv run python -m steps.step1_rewrite_text

echo "Running Step 2..."
uv run python -m steps.step2_generate_speech --device cuda

echo "Running Step 3..."
uv run python -m steps.step3_text_baseline

echo "Running Step 4..."
uv run python -m steps.step4_asr_pipeline

echo "Running Step 5..."
uv run python -m steps.step5_e2e_audio

echo "Pipeline complete!"
```

## Configuration

All configuration is managed through `config.yaml`. Key settings:

```yaml
models:
  rewrite_model: "google/gemini-2.5-flash"  # For text rewriting
  text_model: "openai/gpt-4o-mini"          # For text baseline
  e2e_model: "google/gemini-2.5-flash"      # For E2E audio

batch_sizes:
  tts: 10        # Qwen3-TTS batch size
  whisper: 20    # Whisper batch size

concurrency:
  max_concurrent_llm: 100  # Max parallel LLM API calls
```

## Output Format

### Text Responses (JSONL)

Each line is a JSON object:

```json
{
  "dialog_id": "12345",
  "original_utterance": "I'm so frustrated with this!",
  "rewritten_text": "I have concerns about this.",
  "emotion": "frustrated",
  "model": "openai/gpt-4o-mini",
  "response": "I understand you have concerns. Let me help you with that.",
  "success": true,
  "error": null
}
```

### Audio Manifest (CSV)

```csv
dialog_id,audio_path,text,emotion,speech_act,intent,success
12345,outputs/audio/12345.wav,I have concerns about this.,neutral,statement,inform,true
```

## Troubleshooting

### Common Issues

1. **OpenRouter API errors**
   - Check your API key is set correctly
   - Verify you have credits available
   - Check rate limits

2. **Whisper transcription failures**
   - Ensure audio files are valid WAV format
   - Check file paths are correct
   - Verify file size is reasonable (< 10MB)

3. **Qwen3-TTS errors**
   - Ensure CUDA is available if using GPU
   - Check model repository path is correct
   - Verify sufficient GPU memory

4. **Memory issues**
   - Reduce batch sizes
   - Process in smaller chunks
   - Use CPU if GPU memory is insufficient

### Logging

Logs are written to `benchmark.log` with configurable verbosity:

```bash
uv run python -m steps.step1_rewrite_text --log-level DEBUG
```

## Adding New Steps

To add a new pipeline step:

1. Create a new file in `steps/`
2. Follow the pattern of existing steps
3. Read from previous step's output
4. Write to new output file
5. Update this README with usage instructions

## License

This benchmark is for research purposes only.
