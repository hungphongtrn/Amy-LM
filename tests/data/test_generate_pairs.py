import json
import os
from unittest.mock import AsyncMock, patch

import pytest
from datasets import Dataset


# ---------- synthetic enriched dataset ----------

def _make_enriched_dataset(num_samples: int = 3) -> Dataset:
    import numpy as np
    rng = np.random.RandomState(42)
    samples = []
    for i in range(num_samples):
        audio_arr = rng.randn(2000).astype(np.float32)
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
            return {"rationale": "r", "response": "resp"}

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
        import asyncio
        from scripts.generate_pairs_deepseek import load_existing_ids, run

        ds = _make_enriched_dataset(3)
        pairs_file = str(tmp_path / "pairs.jsonl")

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

        # 2 pending samples × 2 API calls each (good + bad) = 4 calls
        assert len(processed_ids) == 4
