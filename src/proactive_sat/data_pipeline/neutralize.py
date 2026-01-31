#!/usr/bin/env python3
"""
Proactive-SAT Lexical Neutralizer

Neutralizes emotional/sarcastic phrasing in text while preserving the underlying request/topic.

Supports two modes:
- rule_based: Pattern-based neutralization using regex and heuristics (default)
- openai: LLM-based neutralization via OpenAI API (requires OPENAI_API_KEY)
"""

import json
import os
import re
import urllib.request
from typing import Any


#: Prompt template for OpenAI-based neutralization
OPENAI_PROMPT = (
    "Rewrite the utterance to be lexically neutral while preserving the underlying request/topic; "
    "remove emotional/sarcastic phrasing; do not add new information. "
    "Return ONLY the rewritten text."
)

#: Default OpenAI model to use
DEFAULT_OPENAI_MODEL = "gpt-5-mini"

#: Regex pattern for leading interjections (case-insensitive)
_INTERJECTION_PATTERN = re.compile(
    r"^(?:\s*(?:Ugh|Hey|Hey\s+there|Oh\s+god|Oh\s+my|Man|Hey\s+yo|Umm+|Ah+|Um+|Hm+|Er+|So)\s*,?\s*)",
    re.IGNORECASE,
)

#: Regex pattern for repeated punctuation
_REPEATED_EXCLAMATION_PATTERN = re.compile(r"!{2,}")
_REPEATED_ELLIPSIS_PATTERN = re.compile(r"\.{3,}")


def _neutralize_rule_based(text: str) -> str:
    """
    Apply rule-based neutralization to text.

    Args:
        text: The input text to neutralize

    Returns:
        Neutralized text
    """
    result = text

    # Remove leading interjections when they're discourse markers
    result = _INTERJECTION_PATTERN.sub("", result)

    # Strip leading/trailing whitespace
    result = result.strip()

    # Normalize repeated exclamation marks
    result = _REPEATED_EXCLAMATION_PATTERN.sub("!", result)

    # Normalize excessive ellipses
    result = _REPEATED_ELLIPSIS_PATTERN.sub("...", result)

    # Final strip
    result = result.strip()

    return result


def _call_openai_api(text: str, model: str | None = None) -> str:
    """
    Call OpenAI API to neutralize text.

    Args:
        text: The input text to neutralize
        model: Optional model override (defaults to DEFAULT_OPENAI_MODEL)

    Returns:
        Neutralized text from the API

    Raises:
        ValueError: If OPENAI_API_KEY is not set
        RuntimeError: If the API call fails
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Set it before using openai mode, or use rule_based mode instead."
        )

    effective_model = model or os.environ.get(
        "PROACTIVE_SAT_OPENAI_MODEL", DEFAULT_OPENAI_MODEL
    )

    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": effective_model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"{OPENAI_PROMPT}\n\nInput: {text}",
                    }
                ],
            }
        ],
    }

    try:
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=30) as response:
            response_body = json.loads(response.read().decode("utf-8"))

        # Parse the first returned text from the response
        # OpenAI Responses API returns output in output[0].content[0].text
        output = response_body.get("output", [])
        if output and len(output) > 0:
            first_output = output[0]
            if "content" in first_output and len(first_output["content"]) > 0:
                return first_output["content"][0].get("text", "").strip()

        # Fallback: try looking at response_format.output if available
        if "response_format" in response_body:
            response_format = response_body["response_format"]
            if "output" in response_format:
                return response_format["output"].strip()

        raise RuntimeError("Could not parse neutralized text from OpenAI response")

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8") if e.fp else ""
        raise RuntimeError(
            f"OpenAI API request failed with status {e.code}: {error_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"OpenAI API request failed (network error): {e.reason}"
        ) from e
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse OpenAI API response: {e}") from e


def neutralize_text(text: str, *, mode: str = "rule_based") -> str:
    """
    Neutralize text by removing emotional/sarcastic phrasing.

    Args:
        text: The input text to neutralize
        mode: Neutralization mode - either "rule_based" (default) or "openai"

    Returns:
        Neutralized text

    Raises:
        ValueError: If mode is not recognized or API key is missing (for openai mode)

    Examples:
        >>> neutralize_text("Ugh, where's my package?")
        "Where's my package?"
        >>> neutralize_text("Hey there! I need help!", mode="rule_based")
        "I need help!"
    """
    if not text:
        return ""

    mode = mode.lower()
    if mode == "rule_based":
        return _neutralize_rule_based(text)
    elif mode == "openai":
        return _call_openai_api(text)
    else:
        raise ValueError(
            f"Unknown neutralizer mode: {mode!r}. "
            f"Valid modes are: 'rule_based', 'openai'"
        )


if __name__ == "__main__":
    # Simple CLI for testing
    import argparse

    parser = argparse.ArgumentParser(
        description="Neutralize emotional phrasing in text"
    )
    parser.add_argument("text", help="Text to neutralize")
    parser.add_argument(
        "--mode",
        choices=["rule_based", "openai"],
        default="rule_based",
        help="Neutralization mode (default: rule_based)",
    )

    args = parser.parse_args()
    result = neutralize_text(args.text, mode=args.mode)
    print(result)
