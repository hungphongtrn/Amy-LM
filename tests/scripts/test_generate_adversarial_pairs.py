import json as _json
import os
import sys
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from generate_adversarial_pairs import (
    INVERSE_EMOTION,
    compute_bleu1,
    compute_length_parity_ratio,
    cosine_similarity,
    count_pairs,
    get_inverse_emotion,
    parse_adversarial_response,
    passes_judge_gate,
    SemanticGate,
)


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


class DeterministicEmbedder:
    def __init__(self, seed: int = 42):
        self._rng = np.random.RandomState(seed)

    def encode(self, texts):
        return np.array([self._rng.randn(128) for _ in texts], dtype=np.float32)


mock_embedder = DeterministicEmbedder()


def test_length_parity_equal():
    assert compute_length_parity_ratio("a b c", "a b c") == 1.0


def test_length_parity_asymmetric():
    ratio = compute_length_parity_ratio("a b c", "a b c d e")
    assert 0.5 < ratio < 0.7


def test_length_parity_one_empty():
    assert compute_length_parity_ratio("a b c", "") == 0.0


def test_length_parity_both_empty():
    assert compute_length_parity_ratio("", "") == 1.0


def test_bleu1_identical():
    assert compute_bleu1("hello world", "hello world") == 1.0


def test_bleu1_partial():
    score = compute_bleu1("hello world", "hello mars")
    assert 0.4 < score < 0.6


def test_bleu1_disjoint():
    score = compute_bleu1("hello world", "foo bar baz")
    assert score < 0.3


def test_bleu1_empty():
    assert compute_bleu1("hello", "") == 0.0


def test_cosine_similarity_identical():
    v = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_semantic_gate_length_fail():
    gate = SemanticGate()
    ok, info = gate.check("Hi", "A very long response " * 20, mock_embedder)
    assert not ok
    assert info["reason"] == "length_parity"


def test_semantic_gate_bleu_fail():
    gate = SemanticGate()
    ok, info = gate.check(
        "completely different words here",
        "totally unrelated text string",
        mock_embedder,
    )
    assert not ok


def test_passes_judge_gate_all_pass():
    result = {
        "fidelity_chosen": 5,
        "fidelity_rejected": 4,
        "ambiguity_chosen": 1,
        "ambiguity_rejected": 2,
    }
    assert passes_judge_gate(result)


def test_passes_judge_gate_fidelity_fail():
    result = {
        "fidelity_chosen": 3,
        "fidelity_rejected": 5,
        "ambiguity_chosen": 1,
        "ambiguity_rejected": 1,
    }
    assert not passes_judge_gate(result)


def test_passes_judge_gate_ambiguity_fail():
    result = {
        "fidelity_chosen": 5,
        "fidelity_rejected": 5,
        "ambiguity_chosen": 3,
        "ambiguity_rejected": 1,
    }
    assert not passes_judge_gate(result)


def test_count_pairs():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        _json.dump({"id": "1", "chosen": "a", "rejected": "b", "emotion_label": "happy"}, f)
        f.write("\n")
        _json.dump({"id": "2", "chosen": "c", "rejected": "d", "emotion_label": "sad"}, f)
        f.write("\n")
        _json.dump({"id": "3", "chosen": "e", "rejected": "f", "emotion_label": "happy"}, f)
        f.write("\n")
        tmp_path = f.name

    try:
        stats = count_pairs(tmp_path)
        assert stats["total"] == 3
        assert stats["emotions"]["happy"] == 2
        assert stats["emotions"]["sad"] == 1
    finally:
        os.unlink(tmp_path)
