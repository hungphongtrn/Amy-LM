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
The speaker's ground-truth emotion: {emotion_label}
The inverse emotion used for the rejected response: {inverse_emotion}

Here are two responses, presented in random order:

Response A: {response_a}
Response B: {response_b}

For each response, rate:
1. Emotion Fidelity (1-5): How well does this response reflect the ground-truth emotion ({emotion_label})?
2. Text-Ambiguity (1-5): How hard would it be for a text-only reader to identify the correct emotion from this response ALONE? (1 = impossible to tell, 5 = obvious from text)

A good adversarial pair should have HIGH fidelity (>=4) and LOW ambiguity (<=2) for both responses.

Output JSON with keys:
- "fidelity_A": integer 1-5
- "fidelity_B": integer 1-5
- "ambiguity_A": integer 1-5
- "ambiguity_B": integer 1-5
- "identified_correct": which response shows emotion "{emotion_label}" more clearly: "A" or "B" or "neither"
"""


async def judge_pair(
    client,
    transcript: str,
    emotion_label: str,
    inverse_emotion: str,
    chosen: str,
    rejected: str,
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

    response = await client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=512,
    )
    judge_data = json.loads(response.choices[0].message.content)

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
    }


def passes_judge_gate(judge_result: dict) -> bool:
    fidelity_ok = (
        judge_result["fidelity_chosen"] >= 4
        and judge_result["fidelity_rejected"] >= 4
    )
    ambiguity_ok = (
        judge_result["ambiguity_chosen"] <= 2
        and judge_result["ambiguity_rejected"] <= 2
    )
    return fidelity_ok and ambiguity_ok
