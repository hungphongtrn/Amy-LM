# Phase 2: DeepSeek Pair Generation

## Phase Goal

All ~4K NVTTS samples have chosen/rejected response pairs saved to `data/nvtts_pairs/pairs.jsonl`. Resumable from partial failures via per-sample append checkpointing.

**Issue:** #22 (B2)

**Depends on:** Phase 1 Script 1 (#20) — enriched dataset at `data/nvtts_enriched/nvtts_enriched.parquet`

## Files to Create

| File | Purpose |
|---|---|
| `scripts/generate_pairs_deepseek.py` | Script 2: async API calls, prompt templates, checkpointed JSONL output |
| `tests/data/test_generate_pairs.py` | Tests with mocked DeepSeek API responses |

---

## Tasks

### Task 0 (Prerequisite): Run Script 1

Before implementing Script 2, the enriched dataset must exist on disk.

- [ ] **Run Script 1 to produce enriched dataset:**

```bash
uv run python scripts/enrich_nvtts.py
```

- [ ] **Verify output exists:**

```bash
python -c "from datasets import Dataset; ds = Dataset.from_parquet('data/nvtts_enriched/nvtts_enriched.parquet'); print(f'{len(ds)} samples, {ds.column_names}')"
```

Expected: ~4045 samples, columns: `id`, `audio`, `emotion_label`, `speaker_name`, `speaker_gender`, `speaker_age_context`, `speaker_nationality`, `transcript_with_tags`, `bare_transcript`, `source`.

> **Note:** Script 1 uses `SpeakerCache.from_mock()` which has limited speaker coverage. For production, switch to `SpeakerCache.build()`. This is acceptable for Phase 2 testing — the pipeline mechanics are validated regardless of mock vs real speaker data.

---

### Task 1: Script 2 — DeepSeek Pair Generation

**Files:**
- Create: `scripts/generate_pairs_deepseek.py`
- Create: `tests/data/test_generate_pairs.py`

#### Prompt Templates

```
GOOD_PROMPT = """You are a conversation partner responding to someone who just spoke.

What the speaker said (including paralinguistic vocalizations in brackets):
{transcript_with_tags}

Emotion: {emotion_label}
Speaker: {speaker_name}, {speaker_gender}, {speaker_age_context}, {speaker_nationality}

Write a natural conversational response that accounts for the speaker's tone, emotion, and identity. Output as JSON with keys "rationale" and "response"."""

BAD_PROMPT = """You are a conversation partner responding to someone who just spoke.

What the speaker said (transcript only):
{bare_transcript}

Write a natural conversational response based purely on the literal words, without any emotional or paralinguistic context.
Output as JSON with keys "rationale" and "response"."""
```

#### Step 1: Write the test

```python
# tests/data/test_generate_pairs.py
import json
import os
import tempfile
from unittest.mock import AsyncMock, patch

import pytest
from datasets import Dataset


# ---------- synthetic enriched dataset ----------

def _make_enriched_dataset(num_samples: int = 3) -> Dataset:
    import numpy as np
    rng = np.random.RandomState(42)
    samples = []
    for i in range(num_samples):
        audio_arr = rng.randn(2000).astype(np.float32)  # tiny audio
        samples.append({
            "id": f"sample_{i}",
            "audio": {"path": None, "array": audio_arr, "sampling_rate": 16000},
            "emotion_label": ["happy", "sad", "neutral"][i % 3],
            "speaker_name": ["Jack", "Lisa", "Bert"][i % 3],
            "speaker_gender": ["Male", "Female", "Male"][i % 3],
            "speaker_age_context": "born 1985",
            "speaker_nationality": "American",
            "transcript_with_tags": f"hello [Breathing] world {i}",
            "bare_transcript": f"hello world {i}",
            "source": "Expresso",
        })
    return Dataset.from_list(samples)


# ---------- mock API responses ----------

def _mock_good_response(sample):
    return {
        "rationale": f"Responding to {sample['emotion_label']} tone from {sample['speaker_name']}.",
        "response": f"I hear you! That sounds {sample['emotion_label']}.",
    }

def _mock_bad_response(sample):
    return {
        "rationale": "Responding to literal transcript.",
        "response": "I acknowledge what you said.",
    }


class TestGeneratePairs:
    """Tests for DeepSeek pair generation script."""

    # --- prompt construction ---

    def test_build_good_prompt_includes_all_context(self):
        from scripts.generate_pairs_deepseek import build_good_prompt
        prompt = build_good_prompt(
            transcript_with_tags="hello [Breathing] world",
            emotion_label="happy",
            speaker_name="Jack",
            speaker_gender="Male",
            speaker_age_context="born 1985",
            speaker_nationality="American",
        )
        assert "[Breathing]" in prompt
        assert "happy" in prompt
        assert "Jack" in prompt
        assert "Male" in prompt
        assert "born 1985" in prompt
        assert "American" in prompt
        assert "rationale" in prompt
        assert "response" in prompt

    def test_build_bad_prompt_is_lean(self):
        from scripts.generate_pairs_deepseek import build_bad_prompt
        prompt = build_bad_prompt("hello world")
        assert "hello world" in prompt
        # bare prompt must NOT leak paralinguistic context
        assert "[Breathing]" not in prompt
        assert "happy" not in prompt
        assert "Speaker" not in prompt

    # --- json parsing ---

    def test_parse_json_response_valid(self):
        from scripts.generate_pairs_deepseek import parse_json_response
        result = parse_json_response('{"rationale": "test", "response": "hello"}')
        assert result == {"rationale": "test", "response": "hello"}

    def test_parse_json_response_invalid_raises(self):
        from scripts.generate_pairs_deepseek import parse_json_response
        with pytest.raises(ValueError):
            parse_json_response("not json")

    # --- checkpointing ---

    def test_load_existing_ids_empty_file(self, tmp_path):
        from scripts.generate_pairs_deepseek import load_existing_ids
        pairs_file = tmp_path / "pairs.jsonl"
        pairs_file.write_text("")
        ids = load_existing_ids(str(pairs_file))
        assert ids == set()

    def test_load_existing_ids_returns_completed(self, tmp_path):
        from scripts.generate_pairs_deepseek import load_existing_ids
        pairs_file = tmp_path / "pairs.jsonl"
        pairs_file.write_text(
            '{"id": "a", "chosen": "x"}\n'
            '{"id": "b", "chosen": "y"}\n'
        )
        ids = load_existing_ids(str(pairs_file))
        assert ids == {"a", "b"}

    # --- full pipeline with mock API ---

    @pytest.mark.asyncio
    async def test_process_sample_generates_pair(self):
        from scripts.generate_pairs_deepseek import process_single_sample

        ds = _make_enriched_dataset(1)
        sample = ds[0]

        async def mock_call(prompt):
            return {"rationale": "r", "response": "resp"}

        pair = await process_single_sample(sample, mock_call)
        assert pair["id"] == sample["id"]
        assert pair["chosen"] == "resp"
        assert pair["rejected"] == "resp"
        assert pair["rationale_chosen"] == "r"
        assert pair["rationale_rejected"] == "r"

    @pytest.mark.asyncio
    async def test_run_produces_jsonl_checkpoint(self, tmp_path):
        from scripts.generate_pairs_deepseek import run

        ds = _make_enriched_dataset(2)
        pairs_file = str(tmp_path / "pairs.jsonl")

        async def mock_api_call(prompt):
            return {"rationale": "r", "response": p}  # response contains prompt type

        await run(ds, pairs_file, mock_api_call, concurrency=1)

        assert os.path.exists(pairs_file)
        with open(pairs_file) as f:
            lines = f.readlines()
        assert len(lines) == 2
        for line in lines:
            entry = json.loads(line)
            assert "id" in entry
            assert "chosen" in entry
            assert "rejected" in entry
            assert "rationale_chosen" in entry
            assert "rationale_rejected" in entry

    def test_resume_skips_completed(self, tmp_path):
        """Resume loads existing IDs and only processes remaining samples."""
        from scripts.generate_pairs_deepseek import load_existing_ids, run
        import asyncio

        ds = _make_enriched_dataset(3)
        pairs_file = str(tmp_path / "pairs.jsonl")

        # Pre-populate with one completed sample
        with open(pairs_file, "w") as f:
            json.dump({
                "id": ds[0]["id"],
                "chosen": "pre-existing", "rejected": "pre-existing",
                "rationale_chosen": "pre", "rationale_rejected": "pre",
            }, f)
            f.write("\n")

        processed_ids = []

        async def mock_api(prompt):
            processed_ids.append("called")
            return {"rationale": "r", "response": "new"}

        asyncio.run(run(ds, pairs_file, mock_api, concurrency=1))

        # Only 2 new calls (3 total - 1 pre-existing)
        assert len(processed_ids) == 2
```

#### Step 2: Run test to verify it fails

```bash
uv run python -m pytest tests/data/test_generate_pairs.py -v
```

Expected: FAIL — `scripts.generate_pairs_deepseek` module not found.

#### Step 3: Write the script

```python
# scripts/generate_pairs_deepseek.py
"""Script 2: Generate chosen/rejected response pairs via DeepSeek V4 Flash.

Reads enriched NVTTS from data/nvtts_enriched/nvtts_enriched.parquet.
For each sample, makes 2 async API calls (good context vs bad bare transcript).
Checkpoints output to JSONL after each sample so partial results survive failures.

Requires: DEEPSEEK_API_KEY environment variable.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional, Set

import aiohttp

ENRICHED_PATH = "data/nvtts_enriched/nvtts_enriched.parquet"
OUTPUT_PATH = "data/nvtts_pairs/pairs.jsonl"

GOOD_PROMPT_TEMPLATE = """You are a conversation partner responding to someone who just spoke.

What the speaker said (including paralinguistic vocalizations in brackets):
{transcript_with_tags}

Emotion: {emotion_label}
Speaker: {speaker_name}, {speaker_gender}, {speaker_age_context}, {speaker_nationality}

Write a natural conversational response that accounts for the speaker's tone, emotion, and identity. Output as JSON with keys "rationale" and "response"."""

BAD_PROMPT_TEMPLATE = """You are a conversation partner responding to someone who just spoke.

What the speaker said (transcript only):
{bare_transcript}

Write a natural conversational response based purely on the literal words, without any emotional or paralinguistic context.
Output as JSON with keys "rationale" and "response"."""

MAX_RETRIES = 3
BACKOFF_BASE = 2.0


def build_good_prompt(
    transcript_with_tags: str,
    emotion_label: str,
    speaker_name: str,
    speaker_gender: str,
    speaker_age_context: str,
    speaker_nationality: str,
) -> str:
    return GOOD_PROMPT_TEMPLATE.format(
        transcript_with_tags=transcript_with_tags,
        emotion_label=emotion_label,
        speaker_name=speaker_name,
        speaker_gender=speaker_gender,
        speaker_age_context=speaker_age_context,
        speaker_nationality=speaker_nationality,
    )


def build_bad_prompt(bare_transcript: str) -> str:
    return BAD_PROMPT_TEMPLATE.format(bare_transcript=bare_transcript)


def parse_json_response(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try extracting JSON from markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        data = json.loads(text)

    if not isinstance(data, dict) or "response" not in data:
        raise ValueError(f"Unexpected JSON structure: {data}")
    return data


def load_existing_ids(pairs_path: str) -> Set[str]:
    if not os.path.exists(pairs_path):
        return set()
    ids = set()
    with open(pairs_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ids.add(entry.get("id", ""))
            except json.JSONDecodeError:
                pass
    return ids


async def call_deepseek(
    session: aiohttp.ClientSession,
    api_key: str,
    prompt: str,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "temperature": 0.7,
    }

    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                async with session.post(url, json=payload, headers=headers) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        content = data["choices"][0]["message"]["content"]
                        return parse_json_response(content)
                    elif resp.status == 429:
                        wait = BACKOFF_BASE ** attempt
                        print(f"  Rate limited, backing off {wait:.0f}s...")
                        await asyncio.sleep(wait)
                        continue
                    else:
                        body = await resp.text()
                        print(f"  API error {resp.status}: {body[:200]}")
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                print(f"  Network error (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(BACKOFF_BASE ** attempt)
                continue

        raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


async def process_single_sample(
    sample: Dict[str, Any],
    api_caller: Callable,
) -> Dict[str, Any]:
    good_prompt = build_good_prompt(
        transcript_with_tags=sample["transcript_with_tags"],
        emotion_label=sample["emotion_label"],
        speaker_name=sample["speaker_name"],
        speaker_gender=sample["speaker_gender"],
        speaker_age_context=sample["speaker_age_context"],
        speaker_nationality=sample["speaker_nationality"],
    )
    bad_prompt = build_bad_prompt(sample["bare_transcript"])

    good_result = await api_caller(good_prompt)
    bad_result = await api_caller(bad_prompt)

    return {
        "id": sample["id"],
        "chosen": good_result["response"],
        "rejected": bad_result["response"],
        "rationale_chosen": good_result.get("rationale", ""),
        "rationale_rejected": bad_result.get("rationale", ""),
    }


async def run(
    dataset: "Dataset",
    output_path: str,
    api_call_fn: Callable,
    concurrency: int = 5,
) -> None:
    existing_ids = load_existing_ids(output_path)
    pending = [s for s in dataset if s["id"] not in existing_ids]

    if not pending:
        print(f"All {len(dataset)} samples already processed in {output_path}")
        return

    print(f"Processing {len(pending)} samples ({len(existing_ids)} already done)")

    semaphore = asyncio.Semaphore(concurrency)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    async def process_and_write(sample):
        try:
            async with semaphore:
                pair = await process_single_sample(sample, api_call_fn)
        except Exception as e:
            print(f"  FAILED {sample['id']}: {e}")
            return

        with open(output_path, "a") as f:
            json.dump(pair, f)
            f.write("\n")

    tasks = [process_and_write(s) for s in pending]
    await asyncio.gather(*tasks)

    total = len(dataset)
    completed = load_existing_ids(output_path)
    print(f"Done: {len(completed)}/{total} samples in {output_path}")


async def main_async(concurrency: int = 5) -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.")
        sys.exit(1)

    from datasets import Dataset

    ds = Dataset.from_parquet(ENRICHED_PATH)
    print(f"Loaded {len(ds)} enriched samples from {ENRICHED_PATH}")

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:

        async def api_caller(prompt: str) -> Dict[str, Any]:
            return await call_deepseek(session, api_key, prompt, semaphore)

        await run(ds, OUTPUT_PATH, api_caller, concurrency=concurrency)


def main() -> None:
    concurrency = int(os.environ.get("DEEPSEEK_CONCURRENCY", "5"))
    asyncio.run(main_async(concurrency=concurrency))


if __name__ == "__main__":
    main()
```

#### Step 4: Run test to verify it passes

```bash
uv run python -m pytest tests/data/test_generate_pairs.py -v
```

#### Step 5: Commit

```bash
git add scripts/generate_pairs_deepseek.py tests/data/test_generate_pairs.py
git commit -m "feat: add DeepSeek pair generation script (Issue #22)"
```

---

## Phase Completion Criteria
- [ ] Script 1 has been run and `data/nvtts_enriched/nvtts_enriched.parquet` exists
- [ ] Script 2 (`scripts/generate_pairs_deepseek.py`) loads enriched dataset, builds good/bad prompts, calls DeepSeek API
- [ ] Checkpointing works: JSONL append, resume skips already-processed IDs
- [ ] Error handling: retry with backoff, log failures, skip after max retries
- [ ] All tests passing:
  ```bash
  uv run python -m pytest tests/data/test_generate_pairs.py -v
  ```

## Handoff Notes
- Requires `DEEPSEEK_API_KEY` in environment for actual API calls (tests use mocks)
- Requires `aiohttp` (may need `uv add aiohttp` if not already in dependencies)
- Output JSONL lines have columns: `id`, `chosen`, `rejected`, `rationale_chosen`, `rationale_rejected`
- Phase 3 reads this JSONL + enriched parquet, merges, embeds, computes cosine similarity
- `concurrency` default 5; override with `DEEPSEEK_CONCURRENCY` env var
- DeepSeek API uses `deepseek-chat` model (V4 Flash compatible) with `response_format={'type': 'json_object'}`
- For testing: entire API is mocked via dependency injection (`api_caller` callable)
