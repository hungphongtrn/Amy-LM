# Amy-LM

Data preparation pipeline for Amy-LM. The pipeline is designed to run on distributed machines (A, B, C, D) to parallelize heavy tasks.

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
