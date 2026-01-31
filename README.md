# Amy-LM

Data preparation pipeline for Amy-LM. The pipeline is designed to run on distributed machines (A, B, C, D) to parallelize heavy tasks.

## Proactive-SAT Pipeline (Phase 1)

Run the complete Phase 1 pipeline to produce a 200-sample HuggingFace dataset:

```bash
uv run python -m proactive_sat.data_pipeline.run_pipeline
```

### LLM Neutralization

Use OpenAI for higher-quality text neutralization:

```bash
OPENAI_API_KEY=your_key_here uv run python -m proactive_sat.data_pipeline.run_pipeline --neutralizer openai
```

The `--neutralizer` option supports three modes:
- `auto` (default): Uses OpenAI if `OPENAI_API_KEY` is set, otherwise falls back to rule-based neutralization
- `openai`: Requires `OPENAI_API_KEY` environment variable
- `rule_based`: Pattern-based neutralization using regex and heuristics (no API key required)

Optional: Set `PROACTIVE_SAT_OPENAI_MODEL` to override the default model (`gpt-5-mini`).

## Installation

This project uses `uv` for fast dependency management.

First, ensure `uv` is installed:

```bash
pip install uv
```

## Distributed Data Processing

### Machine A: Alignment

**Goal**: Generate word-level timestamps using NeMo.
**Script**: `src/data/1_extract_alignment.py`

**System Dependencies**:

```bash
sudo apt-get install ffmpeg libsndfile1 clang 
```

**Python Dependencies**:

```bash
uv sync --extra alignment
```

**Run**:

```bash
uv run src/data/1_extract_alignment.py
```

---

### Machine B: WavLM Extraction

**Goal**: Extract acoustic features using WavLM.
**Script**: `src/data/2_extract_wavlm.py`

**Python Dependencies**:

```bash
uv sync --extra wavlm
```

*Note: This machine does not need NeMo.*

**Run**:

```bash
uv run src/data/2_extract_wavlm.py
```

---

### Machine C: LLM Extraction

**Goal**: Extract semantic features (hidden states) from Qwen.
**Script**: `src/data/3_extract_llm.py`

**Python Dependencies**:

```bash
uv sync --extra llm
```

*Note: Includes `sentencepiece` and `protobuf` for Qwen tokenizer support.*

**Run**:

```bash
uv run src/data/3_extract_llm.py
```

---

### Machine D: Upload/Merge

**Goal**: Merge intermediate datasets and upload the final dataset.
**Script**: `src/data/push_to_hub.py`

**Python Dependencies**:

```bash
uv sync --extra upload
```

*Note: CPU-only instance is sufficient for this step.*

**Run**:

```bash
uv run src/data/push_to_hub.py
```
