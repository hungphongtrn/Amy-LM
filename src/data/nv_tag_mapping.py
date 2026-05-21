"""NV Tag Emoji Mapping — Issue #17.

Maps NVTTS emoji-based paralinguistic vocalization tags to human-readable
[Tag] text labels for use in LLM prompt templates.

Emoji mapping derived from scanning the deepvk/NonverbalTTS dataset.
Each NV type may have multiple emoji representations (e.g. with/without
U+FE0F variation selector); only the canonical form is stored in the
primary mapping, and the transform handles variant forms.
"""

import re

NV_EMOJI_TO_TAG: dict[str, str] = {
    # Emojis observed in NVTTS dataset (deepvk/NonverbalTTS)
    # Mapping validated by scanning all 3,641 rows of the dataset.
    # 10 NV types: Breathing, Laughter, Sigh, Sneeze, Cough, Throat clear,
    #              Groan, Grunt, Snore, Sniff
    "\U0001f32c": "[Breathing]",    # 🌬  WIND BLOWING FACE
    "\U0001f923": "[Laughter]",     # 🤣 ROLLING ON THE FLOOR LAUGHING
    "\U0001f624": "[Sigh]",         # 😤 FACE WITH LOOK OF TRIUMPH
    "\U0001f927": "[Sneeze]",       # 🤧 SNEEZING FACE
    "\U0001f637": "[Cough]",        # 😷 FACE WITH MEDICAL MASK
    "\U0001f5e3": "[Throat clear]",# 🗣  SPEAKING HEAD IN SILHOUETTE
    "\U0001f616": "[Groan]",        # 😖 CONFOUNDED FACE
    "\U0001f416": "[Grunt]",        # 🐖 PIG
    "\U0001f634": "[Snore]",        # 😴 SLEEPING FACE
    "\U0001f443": "[Sniff]",        # 👃 NOSE
}

# Reverse mapping: text label -> canonical emoji (without FE0F)
NV_TAG_TO_EMOJI: dict[str, str] = {
    v: k.replace("\ufe0f", "") for k, v in NV_EMOJI_TO_TAG.items()
}


def _build_emoji_to_tag_extended() -> dict[str, str]:
    """Build extended mapping that includes FE0F variants."""
    extended: dict[str, str] = dict(NV_EMOJI_TO_TAG)
    for emoji, tag in NV_EMOJI_TO_TAG.items():
        if "\ufe0f" not in emoji:
            extended[emoji + "\ufe0f"] = tag
    return extended


_EMOJI_TO_TAG_EXTENDED: dict[str, str] = _build_emoji_to_tag_extended()


def strip_nv_emojis(text: str) -> str:
    """Remove all NV emoji symbols from text, collapsing resulting whitespace.

    Args:
        text: Text that may contain NV emoji paralinguistic markers.

    Returns:
        Text with all NV emojis removed and whitespace normalized.
    """
    result = text
    for emoji in sorted(_EMOJI_TO_TAG_EXTENDED, key=len, reverse=True):
        result = result.replace(emoji, " ")
    result = re.sub(r"\s+", " ", result).strip()
    return result


def emojis_to_tags(text: str) -> str:
    """Replace NV emoji symbols in text with [Tag] labels.

    Handles both canonical emojis and those with U+FE0F variation selectors.
    Multiple consecutive spaces are collapsed after replacement.

    Args:
        text: NVTTS Result column text containing emoji symbols.

    Returns:
        Text with emojis replaced by canonical [Tag] labels.
    """
    result = text
    for emoji in sorted(_EMOJI_TO_TAG_EXTENDED, key=len, reverse=True):
        result = result.replace(emoji, f" {_EMOJI_TO_TAG_EXTENDED[emoji]} ")
    # Collapse multiple spaces
    result = re.sub(r"\s+", " ", result).strip()
    return result
