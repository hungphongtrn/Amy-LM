#!/usr/bin/env python3
"""
Proactive-SAT Prosody Instructions Generator

Generates TTS-friendly speaker instructions based on emotion metadata.
Creates control (neutral) and trigger (style-specific) instructions for prosodic injection.
"""

from typing import Literal


#: Emotion keywords that map to frustrated prosody style
_FRUSTRATED_KEYWORDS = frozenset(
    {
        "anger",
        "angry",
        "annoyance",
        "annoyed",
        "disapproval",
        "disapprove",
        "frustration",
        "frustrated",
        "irritation",
        "irritated",
        "exasperation",
    }
)

#: Emotion keywords that map to distressed prosody style
_DISTRESSED_KEYWORDS = frozenset(
    {
        "anxiety",
        "anxious",
        "nervous",
        "nervousness",
        "sadness",
        "sad",
        "distress",
        "distressed",
        "fear",
        "fearful",
        "worry",
        "worried",
        "concern",
        "concerned",
        "unease",
        "uneasy",
        "panic",
        "panicked",
    }
)

#: Valid prosody styles
PROSODY_STYLES = frozenset({"sarcastic", "frustrated", "distressed"})

#: Type alias for prosody style
ProsodyStyle = Literal["sarcastic", "frustrated", "distressed"]


def determine_prosody_style(emotion: str | None) -> ProsodyStyle:
    """
    Determine the prosody style based on emotion field.

    Args:
        emotion: Comma-separated emotion tokens (e.g., "anger, frustration")

    Returns:
        ProsodyStyle: One of "sarcastic", "frustrated", or "distressed"
    """
    if not emotion:
        return "sarcastic"  # Default

    # Parse comma-separated tokens
    tokens = [tok.strip().lower() for tok in emotion.split(",") if tok.strip()]

    # Check for frustrated keywords first
    for token in tokens:
        for keyword in _FRUSTRATED_KEYWORDS:
            if keyword in token:
                return "frustrated"

    # Check for distressed keywords
    for token in tokens:
        for keyword in _DISTRESSED_KEYWORDS:
            if keyword in token:
                return "distressed"

    # Default to sarcastic for other emotions
    return "sarcastic"


def control_speaker_instruction(text: str) -> str:
    """
    Generate a flat, factual, emotionally neutral delivery instruction.

    Args:
        text: The text to be spoken (the neutral_text)

    Returns:
        Control speaker instruction string
    """
    return f'Deliver the following text in a flat, factual, emotionally neutral tone: "{text}"'


def trigger_speaker_instruction(
    text: str,
    prosody_style: ProsodyStyle,
    *,
    include_non_lexical: bool = True,
) -> str:
    """
    Generate a style-specific TTS-friendly instruction for prosodic injection.

    Args:
        text: The text to be spoken (the neutral_text)
        prosody_style: The target prosody style
        include_non_lexical: Whether to include mild non-lexical cues (default: True)

    Returns:
        Trigger speaker instruction string with pace, pitch, emphasis, and optional cues
    """
    if prosody_style == "sarcastic":
        return _sarcastic_instruction(text, include_non_lexical)
    elif prosody_style == "frustrated":
        return _frustrated_instruction(text, include_non_lexical)
    elif prosody_style == "distressed":
        return _distressed_instruction(text, include_non_lexical)
    else:
        raise ValueError(f"Unknown prosody style: {prosody_style!r}")


def _sarcastic_instruction(text: str, include_non_lexical: bool) -> str:
    """Generate sarcastic prosody instruction."""
    parts = [
        "Deliver the following text with a sarcastic tone:",
        f'Text: "{text}"',
    ]
    if include_non_lexical:
        parts.extend(
            [
                "Use a slight eye-roll quality in the voice.",
                "Add a subtle mocking emphasis on key words.",
                "Slightly longer pause before the final word.",
            ]
        )
    parts.extend(
        [
            "Maintain natural pacing but with a hint of irony.",
            "Pitch: slightly higher on emphasized words.",
        ]
    )
    return " ".join(parts)


def _frustrated_instruction(text: str, include_non_lexical: bool) -> str:
    """Generate frustrated prosody instruction."""
    parts = [
        "Deliver the following text with clear frustration:",
        f'Text: "{text}"',
    ]
    if include_non_lexical:
        parts.extend(
            [
                "Start with a heavy sigh before speaking.",
                "Add audible exhale at the start.",
            ]
        )
    parts.extend(
        [
            "Use a faster pace that breaks slightly at emotional peaks.",
            "Pitch: lower overall, rising slightly on complaint words.",
            "Add slight breathiness when emphasizing frustration.",
            "Shorter pauses between phrases to convey impatience.",
        ]
    )
    return " ".join(parts)


def _distressed_instruction(text: str, include_non_lexical: bool) -> str:
    """Generate distressed prosody instruction."""
    parts = [
        "Deliver the following text with visible distress:",
        f'Text: "{text}"',
    ]
    if include_non_lexical:
        parts.extend(
            [
                "Start with a shaky inhale before the first word.",
                "Include a brief pause after the first phrase.",
            ]
        )
    parts.extend(
        [
            "Use a slower, uneven pace with slight wavering.",
            "Pitch: slightly higher and more variable, with occasional trembling.",
            "Add slight vocal tremor on key emotional words.",
            "Slightly softer volume overall.",
        ]
    )
    return " ".join(parts)


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(description="Test prosody instruction generation")
    parser.add_argument(
        "emotion",
        nargs="?",
        default=None,
        help="Emotion field (comma-separated tokens)",
    )
    parser.add_argument(
        "--text",
        default="I need help with my order",
        help="Sample text to generate instructions for",
    )

    args = parser.parse_args()

    prosody_style = determine_prosody_style(args.emotion)
    control = control_speaker_instruction(args.text)
    trigger = trigger_speaker_instruction(args.text, prosody_style)

    print(f"Emotion: {args.emotion}")
    print(f"Prosody Style: {prosody_style}")
    print(f"\nControl Instruction:\n{control}")
    print(f"\nTrigger Instruction:\n{trigger}")
