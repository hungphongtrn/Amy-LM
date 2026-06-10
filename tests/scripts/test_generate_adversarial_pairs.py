import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from generate_adversarial_pairs import INVERSE_EMOTION, get_inverse_emotion


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
