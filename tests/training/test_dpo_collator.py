from __future__ import annotations

import numpy as np
import torch

from src.training.dpo_collator import DPOCollator


class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [max(1, ord(c) % 1000) for c in text if c != " "]


class _MockProcessor:
    import re

    _AUDIO_SPAN_RE = re.compile(r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>")

    def __init__(self):
        self._base_tokenizer = _MockTokenizer()

    def _extract_mel(self, waveform):
        t_mel = max(1, waveform.shape[0] // 160)
        return torch.randn(128, t_mel)

    def _conv3_downsample_len(self, raw_len):
        return max(1, raw_len * 2) // 2

    def _build_audio_placeholder_ids(self, num_audio_tokens):
        return [151654] * num_audio_tokens


def _make_example(audio_len=16000, prosody_len=80, timbre_dim=256):
    return {
        "audio": {
            "array": np.random.randn(audio_len).astype(np.float32),
            "sampling_rate": 16000,
        },
        "prosody_codebooks_idx": list(range(prosody_len)),
        "timbre_vector": np.random.randn(timbre_dim).astype(np.float32).tolist(),
        "chosen": "This is the chosen response.",
        "rejected": "This is the rejected response.",
        "cosine_similarity": 0.5,
    }


def test_output_shapes_batch_size_2():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    batch = collator([_make_example(), _make_example(audio_len=12000, prosody_len=64)])
    assert batch["input_ids"].shape[0] == 4
    assert batch["attention_mask"].shape[0] == 4
    assert batch["completion_mask"].shape[0] == 4
    assert batch["audio_data"].shape[0] == 4
    assert batch["audio_data_seqlens"].shape[0] == 4
    assert batch["audio_input_mask"].shape[0] == 4
    assert batch["prosody_indices"].shape[0] == 4
    assert batch["timbre_vector"].shape[0] == 4


def test_audio_fields_duplicated_identically():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    batch = collator([_make_example(), _make_example(audio_len=12000, prosody_len=64)])
    b = 2
    assert torch.equal(batch["audio_data"][:b], batch["audio_data"][b:])
    assert torch.equal(batch["audio_data_seqlens"][:b], batch["audio_data_seqlens"][b:])
    assert torch.equal(batch["prosody_indices"][:b], batch["prosody_indices"][b:])
    assert torch.equal(batch["timbre_vector"][:b], batch["timbre_vector"][b:])


def test_completion_mask_structure():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    examples = [_make_example()]
    batch = collator(examples)
    waveform = collator._extract_audio(examples[0]["audio"])
    mel_len = collator._extract_mel_batch([waveform])[1][0].item()
    chosen_meta = collator._tokenize_sample(mel_len, examples[0]["chosen"])
    prompt_len = min(chosen_meta["prompt_len"], batch["completion_mask"].shape[1])
    mask = batch["completion_mask"][0]
    assert torch.all(mask[:prompt_len] == 0)
    assert torch.all(mask[prompt_len: batch["attention_mask"][0].sum().item()] == 1)


def test_audio_input_mask_positions():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    batch = collator([_make_example()])
    ids = batch["input_ids"][0]
    audio_mask = batch["audio_input_mask"][0]
    assert torch.equal(audio_mask, ids == DPOCollator.AUDIO_TOKEN_ID)


def test_padding_consistent_across_fields():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0, pad_to_multiple_of=8)
    batch = collator([_make_example(), _make_example(audio_len=5000)])
    seq_len = batch["input_ids"].shape[1]
    assert batch["attention_mask"].shape[1] == seq_len
    assert batch["completion_mask"].shape[1] == seq_len
    assert batch["audio_input_mask"].shape[1] == seq_len
    assert seq_len % 8 == 0


def test_batch_size_1():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    batch = collator([_make_example()])
    assert batch["input_ids"].shape[0] == 2
    assert batch["audio_data"].shape[0] == 2


def test_empty_batch_raises():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    try:
        collator([])
        assert False, "Expected ValueError for empty batch"
    except ValueError as exc:
        assert "requires at least one example" in str(exc)


def test_missing_key_raises():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    example = _make_example()
    del example["chosen"]
    try:
        collator([example])
        assert False, "Expected KeyError for missing chosen key"
    except KeyError as exc:
        assert exc.args[0] == "chosen"


def test_truncation_respects_max_length():
    collator = DPOCollator(_MockProcessor(), pad_token_id=0, max_length=10)
    batch = collator([_make_example(audio_len=64000)])
    assert batch["input_ids"].shape[1] <= 10


def test_extract_audio_from_audiodecoder():
    """Regression: HF datasets 4.8+ returns torchcodec AudioDecoder, not dict."""
    samps = torch.randn(1, 8000, dtype=torch.float32)

    class _MockAudioSamples:
        data = samps

    class _MockMetadata:
        sample_rate = 16000

    class _MockAudioDecoder:
        def get_all_samples(self):
            return _MockAudioSamples

        metadata = _MockMetadata()

    collator = DPOCollator(_MockProcessor(), pad_token_id=0)
    waveform = collator._extract_audio(_MockAudioDecoder())
    assert isinstance(waveform, torch.Tensor)
    assert waveform.dtype == torch.float32
    assert waveform.ndim == 1
    assert waveform.shape[0] == 8000
    assert torch.equal(waveform, samps.flatten())
