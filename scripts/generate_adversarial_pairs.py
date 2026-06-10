"""Generate adversarial preference pairs for DPO training.

Inverse emotion mapping is used to pair samples with opposite emotions
while preserving text content.

Subsequent tasks will add pair generation logic.
"""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter

import numpy as np

import asyncio
import argparse
import os
import signal
import sys

from datasets import Dataset
from dotenv import load_dotenv
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer
from tqdm.asyncio import tqdm as async_tqdm

load_dotenv()

# "other" is skipped from mapping but still generates pairs; quality filters decide survival
INVERSE_EMOTION = {
    "happy": "sad",
    "sad": "happy",
    "angry": "neutral",
    "disgusted": "neutral",
    "fearful": "sad",
    "surprised": "happy",
    "neutral": "sad",
}


def get_inverse_emotion(emotion: str) -> str | None:
    return INVERSE_EMOTION.get(emotion)


ADVERSARIAL_PROMPT_TEMPLATE = """You are a conversation partner responding to someone who just spoke.

Transcript of what the speaker said:
{transcript}

The speaker expressed the following emotion: {emotion_label}.
The opposite/inverse of this emotion is: {inverse_emotion}.

Your task:
1. Write a response that reflects the GROUND-TRUTH emotion ({emotion_label}). Label this "chosen".
2. Write a response that reflects the INVERSE emotion ({inverse_emotion}). Label this "rejected".
3. Briefly explain your strategy for making these responses textually similar but emotionally different. Label this "strategy".

Both responses should be natural conversation. The two responses should be structurally and lexically similar — a reader who cannot hear the audio should struggle to tell which is correct.

Output JSON with keys "strategy", "chosen", "rejected".

EXAMPLE JSON OUTPUT:
{{
    "strategy": "I made both responses express concern, but the chosen response reflects the speaker's upbeat tone while the rejected is flat.",
    "chosen": "That's great to hear! I'm really happy for you.",
    "rejected": "I understand. That must have been difficult."
}}
"""


def parse_adversarial_response(text: str) -> dict:
    data = {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        pass

    if not _validate_pair(data):
        if "```json" in text:
            block = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            block = text.split("```")[1].split("```")[0]
        else:
            block = text
        try:
            data = json.loads(block.strip())
        except json.JSONDecodeError:
            pass

    if not _validate_pair(data):
        data = _regex_extract_fields(text)

    if not _validate_pair(data):
        raise ValueError(f"Could not extract 'chosen' and 'rejected' from text: {text[:300]}")
    return data


def _validate_pair(data: dict) -> bool:
    return (
        isinstance(data, dict)
        and "chosen" in data
        and "rejected" in data
        and isinstance(data["chosen"], str)
        and isinstance(data["rejected"], str)
        and len(data["chosen"].strip()) > 0
        and len(data["rejected"].strip()) > 0
    )


def _regex_extract_fields(text: str) -> dict:
    result = {"strategy": "", "chosen": "", "rejected": ""}
    for key in ("strategy", "chosen", "rejected"):
        match = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if match:
            result[key] = json.loads(f'"{match.group(1)}"')
    return result


def compute_bleu1(text_a: str, text_b: str) -> float:
    ref_tokens = text_a.lower().split()
    hyp_tokens = text_b.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0

    ref_counts = Counter(ref_tokens)
    matches = 0
    for token in hyp_tokens:
        if ref_counts.get(token, 0) > 0:
            matches += 1
            ref_counts[token] -= 1

    precision = matches / len(hyp_tokens) if len(hyp_tokens) > 0 else 0.0

    bp = min(1.0, math.exp(1 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 1.0

    return precision * bp


def compute_length_parity_ratio(text_a: str, text_b: str) -> float:
    len_a = len(text_a.split())
    len_b = len(text_b.split())
    if len_a == 0 and len_b == 0:
        return 1.0
    if min(len_a, len_b) == 0:
        return 0.0
    return min(len_a, len_b) / max(len_a, len_b)


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


class SemanticGate:
    def __init__(
        self,
        length_ratio_min: float = 0.8,
        length_ratio_max: float = 1.2,
        bleu_min: float = 0.3,
        sim_min: float = 0.6,
        sim_max: float = 0.9,
    ):
        self.length_ratio_min = length_ratio_min
        self.length_ratio_max = length_ratio_max
        self.bleu_min = bleu_min
        self.sim_min = sim_min
        self.sim_max = sim_max

    def check(self, chosen: str, rejected: str, embedding_model) -> tuple[bool, dict]:
        length_ratio = compute_length_parity_ratio(chosen, rejected)
        if not (self.length_ratio_min < length_ratio < self.length_ratio_max):
            return False, {"reason": "length_parity", "value": length_ratio}

        bleu = compute_bleu1(chosen, rejected)
        if bleu <= self.bleu_min:
            return False, {"reason": "lexical_overlap", "value": bleu}

        chosen_emb = embedding_model.encode([chosen])[0]
        rejected_emb = embedding_model.encode([rejected])[0]
        sim = cosine_similarity(chosen_emb, rejected_emb)
        if not (self.sim_min < sim < self.sim_max):
            return False, {"reason": "semantic_similarity", "value": sim}

        return True, {
            "length_ratio": length_ratio,
            "bleu": bleu,
            "sim": sim,
        }


JUDGE_PROMPT_TEMPLATE = """You are evaluating the quality of a preference pair for speech-language DPO training.

The speaker's transcript: {transcript}
Two emotions are relevant to this pair: "{emotion_label}" and "{inverse_emotion}" — but you do NOT know which response is intended for which.

Here are two responses, presented in random order:

Response A: {response_a}
Response B: {response_b}

For each response, rate:
1. Emotional Expressiveness (1-5): How strongly does this response convey AN emotion (any emotion)? 5 = vivid emotional tone, 1 = completely flat/neutral.
2. Text-Ambiguity (1-5): How hard would it be for a text-only reader to identify WHICH SPECIFIC emotion this response is expressing? 1 = impossible to tell which emotion, 5 = the intended emotion is obvious from text alone.

Then answer:
3. Which response better matches the transcript's emotion "{emotion_label}"? Answer "A", "B", or "neither" (if text alone cannot distinguish them).

A good adversarial pair has:
- Both responses are emotionally expressive (fidelity >= 4)
- Neither response reveals its intended emotion from text alone (ambiguity <= 2)
- You cannot identify which is which from text ("identified_correct" = "neither")

Output JSON with keys:
- "fidelity_A": integer 1-5
- "fidelity_B": integer 1-5
- "ambiguity_A": integer 1-5
- "ambiguity_B": integer 1-5
- "identified_correct": "A" or "B" or "neither"
"""


async def judge_pair(
    client,
    transcript: str,
    emotion_label: str,
    inverse_emotion: str,
    chosen: str,
    rejected: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    if random.random() < 0.5:
        response_a, response_b = chosen, rejected
        a_is_chosen = True
    else:
        response_a, response_b = rejected, chosen
        a_is_chosen = False

    prompt = JUDGE_PROMPT_TEMPLATE.format(
        transcript=transcript,
        emotion_label=emotion_label,
        inverse_emotion=inverse_emotion,
        response_a=response_a,
        response_b=response_b,
    )

    for attempt in range(MAX_RETRIES):
        try:
            async with semaphore:
                response = await client.chat.completions.create(
                    model="deepseek-v4-flash",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=512,
                )
            content = response.choices[0].message.content
            try:
                judge_data = json.loads(content)
            except json.JSONDecodeError:
                if "```json" in content:
                    block = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    block = content.split("```")[1].split("```")[0]
                else:
                    block = content
                judge_data = json.loads(block.strip())
            break
        except Exception:
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(1.0 * (attempt + 1))
                continue
            raise

    if a_is_chosen:
        fidelity_chosen = judge_data.get("fidelity_A", 0)
        fidelity_rejected = judge_data.get("fidelity_B", 0)
        ambiguity_chosen = judge_data.get("ambiguity_A", 0)
        ambiguity_rejected = judge_data.get("ambiguity_B", 0)
    else:
        fidelity_chosen = judge_data.get("fidelity_B", 0)
        fidelity_rejected = judge_data.get("fidelity_A", 0)
        ambiguity_chosen = judge_data.get("ambiguity_B", 0)
        ambiguity_rejected = judge_data.get("ambiguity_A", 0)

    return {
        "fidelity_chosen": fidelity_chosen,
        "fidelity_rejected": fidelity_rejected,
        "ambiguity_chosen": ambiguity_chosen,
        "ambiguity_rejected": ambiguity_rejected,
        "identified_correct": judge_data.get("identified_correct", "neither"),
    }


def passes_judge_gate(judge_result: dict) -> tuple[bool, str]:
    if judge_result["fidelity_chosen"] < 4:
        return False, f"fidelity_chosen={judge_result['fidelity_chosen']} (< 4)"
    if judge_result["fidelity_rejected"] < 4:
        return False, f"fidelity_rejected={judge_result['fidelity_rejected']} (< 4)"
    if judge_result["ambiguity_chosen"] > 2:
        return False, f"ambiguity_chosen={judge_result['ambiguity_chosen']} (> 2)"
    if judge_result["ambiguity_rejected"] > 2:
        return False, f"ambiguity_rejected={judge_result['ambiguity_rejected']} (> 2)"
    if judge_result.get("identified_correct", "neither") != "neither":
        return False, f"identified_correct={judge_result.get('identified_correct')} (expected 'neither')"
    return True, "passed"


MAX_RETRIES = 3


async def generate_single_pair(
    client: AsyncOpenAI,
    transcript: str,
    emotion_label: str,
    inverse_emotion: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    prompt = ADVERSARIAL_PROMPT_TEMPLATE.format(
        transcript=transcript,
        emotion_label=emotion_label,
        inverse_emotion=inverse_emotion,
    )

    for attempt in range(MAX_RETRIES):
        temperature = 0.7 + attempt * 0.15
        async with semaphore:
            response = await client.chat.completions.create(
                model="deepseek-v4-flash",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=temperature,
                max_tokens=2048,
            )
        try:
            data = parse_adversarial_response(response.choices[0].message.content)
            return {
                "chosen": data["chosen"],
                "rejected": data["rejected"],
                "strategy": data.get("strategy", ""),
                "attempt": attempt + 1,
            }
        except ValueError:
            if attempt < MAX_RETRIES - 1:
                continue
            raise


def load_existing_ids(output_path: str) -> set:
    if not os.path.exists(output_path):
        return set()
    ids = set()
    with open(output_path, "r") as f:
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


async def run_adversarial_generation(
    dataset,
    output_path: str,
    concurrency: int = 5,
    smoke: bool = False,
    stop_event: asyncio.Event | None = None,
) -> None:
    existing_ids = load_existing_ids(output_path)
    pending = [s for s in dataset if s["id"] not in existing_ids]
    if smoke:
        pending = pending[:10]

    if not pending:
        print(f"All {len(dataset)} samples already processed in {output_path}", flush=True)
        return

    print(f"Processing {len(pending)} samples ({len(existing_ids)} already done)", flush=True)

    embedding_model = SentenceTransformer("google/embeddinggemma-300m")
    semantic_gate = SemanticGate()

    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("ERROR: DEEPSEEK_API_KEY environment variable not set.", flush=True)
        sys.exit(1)

    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
        max_retries=1,
        timeout=30.0,
    )
    semaphore = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    failed_count = 0
    rejected_count = 0
    passed_count = 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    async def process_sample(sample):
        nonlocal failed_count, rejected_count, passed_count
        emotion = sample.get("emotion_label", "")
        inverse = get_inverse_emotion(emotion)
        if inverse is None:
            return

        try:
            pair = await generate_single_pair(
                client,
                sample.get("transcript_with_tags", sample.get("bare_transcript", "")),
                emotion,
                inverse,
                semaphore,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            async with lock:
                failed_count += 1
            async_tqdm.write(f"  GEN_FAILED {sample['id']}: {e}")
            return

        ok, gate_info = semantic_gate.check(pair["chosen"], pair["rejected"], embedding_model)
        if not ok:
            async_tqdm.write(f"  SEMANTIC_GATE {sample['id']}: {gate_info}")
            async with lock:
                rejected_count += 1
            return

        try:
            judge_result = await judge_pair(
                client,
                sample.get("transcript_with_tags", sample.get("bare_transcript", "")),
                emotion,
                inverse,
                pair["chosen"],
                pair["rejected"],
                semaphore,
            )
        except Exception as e:
            async_tqdm.write(f"  JUDGE_FAILED {sample['id']}: {e}")
            async with lock:
                failed_count += 1
            return

        judge_ok, judge_reason = passes_judge_gate(judge_result)
        if not judge_ok:
            async_tqdm.write(f"  JUDGE_GATE {sample['id']}: {judge_reason}")
            async with lock:
                rejected_count += 1
            return

        output = {
            "id": sample.get("id", ""),
            "chosen": pair["chosen"],
            "rejected": pair["rejected"],
            "strategy": pair["strategy"],
            "emotion_label": emotion,
            "inverse_emotion": inverse,
            "judge_fidelity_chosen": judge_result["fidelity_chosen"],
            "judge_fidelity_rejected": judge_result["fidelity_rejected"],
            "judge_ambiguity_chosen": judge_result["ambiguity_chosen"],
            "judge_ambiguity_rejected": judge_result["ambiguity_rejected"],
            "judge_identified_correct": judge_result["identified_correct"],
            "generation_attempts": pair["attempt"],
        }
        async with lock:
            with open(output_path, "a") as f:
                f.write(json.dumps(output) + "\n")
            passed_count += 1

    tasks = [asyncio.create_task(process_sample(s)) for s in pending]

    with async_tqdm(total=len(pending), desc="Generating adversarial pairs") as pbar:
        for coro in asyncio.as_completed(tasks):
            try:
                await coro
            except asyncio.CancelledError:
                pass
            pbar.update(1)
            if stop_event and stop_event.is_set():
                async_tqdm.write("Cancelling remaining tasks...")
                for t in tasks:
                    t.cancel()

    completed = len(load_existing_ids(output_path))
    print(
        f"Done: {completed} pairs in {output_path} "
        f"({failed_count} failed, {rejected_count} rejected, {passed_count} passed this run)",
        flush=True,
    )


def count_pairs(jsonl_path: str) -> dict:
    """Count pairs in JSONL output, grouped by emotion label."""
    pairs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    emotion_counts = {}
    for p in pairs:
        e = p.get("emotion_label", "unknown")
        emotion_counts[e] = emotion_counts.get(e, 0) + 1

    return {
        "total": len(pairs),
        "emotions": emotion_counts,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate adversarial DPO preference pairs from NVTTS"
    )
    parser.add_argument(
        "--input",
        default="data/nvtts_enriched/nvtts_enriched.parquet",
        help="Enriched NVTTS parquet file",
    )
    parser.add_argument(
        "--output",
        default="data/nvtts_adversarial/pairs.jsonl",
        help="Output JSONL path",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("DEEPSEEK_CONCURRENCY", "5")),
        help="Number of concurrent API calls",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: process only 10 samples",
    )
    parser.add_argument(
        "--count",
        type=str,
        default=None,
        help="Count statistics for an existing JSONL file without generating",
    )
    args = parser.parse_args()

    if args.count:
        stats = count_pairs(args.count)
        print(f"Total pairs: {stats['total']}")
        print("Emotion distribution:")
        for emotion, count in sorted(stats["emotions"].items()):
            print(f"  {emotion}: {count}")
        return

    ds = Dataset.from_parquet(args.input)
    print(f"Loaded {len(ds)} samples from {args.input}")

    sys.stdout.reconfigure(line_buffering=True)

    stop_event = asyncio.Event()

    def _on_sigint(signum, frame):
        print("\nInterrupted, finishing in-flight requests...", flush=True)
        stop_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    try:
        asyncio.run(
            run_adversarial_generation(
                ds,
                args.output,
                concurrency=args.concurrency,
                smoke=args.smoke,
                stop_event=stop_event,
            )
        )
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
