"""Generate adversarial preference pairs for DPO training.

Inverse emotion mapping is used to pair samples with opposite emotions
while preserving text content.

Subsequent tasks will add pair generation logic.
"""

from __future__ import annotations

import json
import re

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
