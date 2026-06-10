"""Generate adversarial preference pairs for DPO training.

Inverse emotion mapping is used to pair samples with opposite emotions
while preserving text content.

Subsequent tasks will add pair generation logic.
"""

from __future__ import annotations

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
