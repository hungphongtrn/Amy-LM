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
        "model": "deepseek-v4-flash",
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
