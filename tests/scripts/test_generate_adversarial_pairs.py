import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from generate_adversarial_pairs import INVERSE_EMOTION, get_inverse_emotion, parse_adversarial_response


def test_inverse_emotion_known():
    assert get_inverse_emotion("happy") == "sad"
    assert get_inverse_emotion("sad") == "happy"
    assert get_inverse_emotion("angry") == "neutral"
    assert get_inverse_emotion("disgusted") == "neutral"
    assert get_inverse_emotion("fearful") == "sad"
    assert get_inverse_emotion("surprised") == "happy"
    assert get_inverse_emotion("neutral") == "sad"


def test_inverse_emotion_skip():
    assert get_inverse_emotion("other") is None


def test_inverse_emotion_unknown():
    assert get_inverse_emotion("nonexistent") is None


def test_parse_direct_json():
    text = '{"strategy": "x", "chosen": "hello", "rejected": "world"}'
    result = parse_adversarial_response(text)
    assert result["chosen"] == "hello"
    assert result["rejected"] == "world"


def test_parse_markdown_block():
    text = '```json\n{"strategy": "x", "chosen": "a", "rejected": "b"}\n```'
    result = parse_adversarial_response(text)
    assert result["chosen"] == "a"


def test_parse_missing_fields_raises():
    with pytest.raises(ValueError):
        parse_adversarial_response('{"chosen": "only one"}')


def test_parse_empty_strings_raises():
    with pytest.raises(ValueError):
        parse_adversarial_response('{"strategy": "", "chosen": "", "rejected": ""}')
