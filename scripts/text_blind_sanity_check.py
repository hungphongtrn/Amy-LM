#!/usr/bin/env python3
"""Text-blind sanity check for adversarial preference pairs.

Verifies that adversarial pairs are text-indistinguishable. Strips all
emotion context and asks DeepSeek V4 Flash to identify the ground-truth
response. Accuracy must be ≤ 55% (near chance for binary choice).

Usage:
    uv run python scripts/text_blind_sanity_check.py \\
        --pairs data/nvtts_adversarial/pairs.jsonl \\
        --num-samples 50

    # Use held-out samples only (last N after train/dev split):
    uv run python scripts/text_blind_sanity_check.py \\
        --pairs data/nvtts_adversarial/pairs.jsonl \\
        --num-samples 50 --held-out-after 1000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv()

TEXT_BLIND_PROMPT = """A speaker said: "{transcript}"

Two responses were given:

Response A: "{response_a}"
Response B: "{response_b}"

Which response better reflects the speaker's tone and emotional state?
Answer with just "A" or "B"."""


def load_pairs(jsonl_path: str, held_out_after: int, num_samples: int) -> list[dict]:
    pairs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    if held_out_after > 0:
        if len(pairs) <= held_out_after:
            print(f"Warning: only {len(pairs)} total pairs, "
                  f"fewer than held_out_after={held_out_after}. Using all pairs.")
        else:
            pairs = pairs[held_out_after:]

    if num_samples and len(pairs) > num_samples:
        rng = random.Random(42)
        pairs = rng.sample(pairs, num_samples)

    return pairs


async def check_single_pair(
    client: AsyncOpenAI,
    pair: dict[str, Any],
    semaphore: asyncio.Semaphore,
) -> dict[str, Any] | None:
    transcript = pair.get("transcript_with_tags", "")
    if not transcript:
        transcript = pair.get("bare_transcript", "")
    if not transcript:
        return None

    chosen = pair.get("chosen", "")
    rejected = pair.get("rejected", "")
    if not chosen or not rejected:
        return None

    swap = random.Random().random() < 0.5
    if swap:
        response_a, response_b = rejected, chosen
        ground_truth = "B"
    else:
        response_a, response_b = chosen, rejected
        ground_truth = "A"

    prompt = TEXT_BLIND_PROMPT.format(
        transcript=transcript,
        response_a=response_a,
        response_b=response_b,
    )

    async with semaphore:
        try:
            resp = await client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a careful text analyst. Answer concisely."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=10,
            )
        except Exception as e:
            print(f"  API error: {e}")
            return None

    answer = resp.choices[0].message.content.strip().upper()
    predicted = answer[0] if answer and answer[0] in ("A", "B") else "?"
    correct = predicted == ground_truth

    return {
        "id": pair.get("id", "?"),
        "emotion_label": pair.get("emotion_label", ""),
        "inverse_emotion": pair.get("inverse_emotion", ""),
        "swapped": swap,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "correct": correct,
    }


async def run_check(pairs: list[dict], concurrency: int) -> list[dict]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("Error: DEEPSEEK_API_KEY not found in environment")
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    semaphore = asyncio.Semaphore(concurrency)

    tasks = [check_single_pair(client, p, semaphore) for p in pairs]
    results = await asyncio.gather(*tasks)
    return [r for r in results if r is not None]


def print_report(results: list[dict]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total * 100 if total > 0 else 0.0

    print(f"\n{'='*50}")
    print(f" Text-Blind Sanity Check Results")
    print(f"{'='*50}")
    print(f"  Total pairs evaluated: {total}")
    print(f"  Correct: {correct}")
    print(f"  Incorrect: {total - correct}")
    print(f"  Accuracy: {accuracy:.1f}%")
    print(f"  Threshold: 55.0%")
    print(f"  Status: {'PASS' if accuracy <= 55.0 else 'FAIL — pairs may be text-distinguishable'}")
    print(f"{'='*50}")

    emotion_stats: dict[str, tuple[int, int]] = {}
    for r in results:
        e = r["emotion_label"]
        prev = emotion_stats.get(e, (0, 0))
        emotion_stats[e] = (prev[0] + 1, prev[1] + (1 if r["correct"] else 0))

    if emotion_stats:
        print(f"\n Per-emotion accuracy:")
        for emotion, (total_e, correct_e) in sorted(emotion_stats.items()):
            acc_e = correct_e / total_e * 100 if total_e > 0 else 0.0
            print(f"  {emotion:15s}: {correct_e}/{total_e} ({acc_e:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Text-blind sanity check for adversarial pairs"
    )
    parser.add_argument(
        "--pairs", required=True,
        help="Path to adversarial JSONL file",
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Number of pairs to evaluate (default: 50)",
    )
    parser.add_argument(
        "--held-out-after", type=int, default=1000,
        help="Use pairs after this index as held-out (default: 1000 = after train+dev)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=5,
        help="Number of concurrent API calls (default: 5)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.pairs):
        print(f"Error: pairs JSONL not found at {args.pairs}")
        sys.exit(1)

    pairs = load_pairs(args.pairs, args.held_out_after, args.num_samples)
    print(f"Loaded {len(pairs)} pairs for evaluation")

    if len(pairs) == 0:
        print("No pairs to evaluate")
        sys.exit(1)

    results = asyncio.run(run_check(pairs, args.concurrency))
    print_report(results)


if __name__ == "__main__":
    main()
