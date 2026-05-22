# Phase 1: Dependencies + DPOCollator

## Phase Goal

DPOCollator produces correctly-shaped concatenated batches with all AmyLM-specific fields from raw NVTTS-FACodec rows. Unit tests verify shapes, masking, and audio field duplication.

## Files to Touch

- `pyproject.toml` — Add `trl`, `peft` dependencies
- `src/training/__init__.py` — May need to create or update
- `src/training/dpo_collator.py` — **Create**: DPOCollator class
- `tests/training/test_dpo_collator.py` — **Create**: Unit tests

## Tasks

### Task 1: Add dependencies

**Files:**
- Edit: `pyproject.toml`

- [ ] **Step 1: Add `trl`, `peft`, and `einops` to pyproject.toml**

```toml
"trl>=0.10.0",
"peft>=0.13.0",
"einops>=0.8.0",
```

- [ ] **Step 2: Install dependencies**

```bash
uv sync
```

- [ ] **Step 3: Verify imports work**

```bash
uv run python -c "import trl; import peft; import einops; print('OK')"
```

### Task 2: Create DPOCollator class skeleton

**Files:**
- Create: `src/training/dpo_collator.py`

- [ ] **Step 1: Define the class with `__init__`**

```python
"""DPO data collator for Amy LM — builds concatenated batches from NVTTS-FACodec rows."""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import torch

# Vendor path for MOSS-Audio processor access
_VENDOR_MOSS_AUDIO_SRC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src")
)
if _VENDOR_MOSS_AUDIO_SRC_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_MOSS_AUDIO_SRC_PATH)

from processing_moss_audio import MossAudioProcessor


class DPOCollator:
    """Collates NVTTS-FACodec rows into DPO batches for AmyLM.

    Each dataset row contains:
        - audio: raw waveform array or dict {"array": ..., "sampling_rate": ...}
        - prosody_codebooks_idx: list of ints [T80]
        - timbre_vector: list of 256 floats
        - chosen: response text string (preferred)
        - rejected: response text string (dispreferred)
        - cosine_similarity: float (for filtering, already applied upstream)

    Produces batches with AmyLM-specific fields duplicated along
    the batch dimension (B → 2×B) for the concatenated chosen+rejected format.
    """

    AUDIO_TOKEN_ID: int = 151654  # <|AUDIO|>
    AUDIO_BOS_ID: int = 151669    # <|audio_bos|>
    AUDIO_EOS_ID: int = 151670    # <|audio_eos|>
    SYSTEM_PROMPT = (
        "You are a helpful assistant. "
        "Listen carefully to the speaker's tone and respond appropriately "
        "to the following speech: <|audio_bos|><|AUDIO|><|audio_eos|>"
    )

    def __init__(
        self,
        processor: MossAudioProcessor,
        pad_token_id: int,
        max_length: int = 1024,
        pad_to_multiple_of: int | None = None,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor._base_tokenizer
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.pad_token_id
```

- [ ] **Step 2: Implement `_extract_audio` helper**

```python
    def _extract_audio(self, audio_input: Any) -> tuple[torch.Tensor, int]:
        """Extract waveform and sample rate from various audio formats.

        Returns:
            (waveform as float32 tensor, sample_rate as int)
        """
        if isinstance(audio_input, dict):
            wav = audio_input.get("array")
            sr = audio_input.get("sampling_rate", 16000)
            if wav is None:
                raise ValueError("Audio dict missing 'array' key")
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav).float()
            return wav, int(sr)
        elif isinstance(audio_input, np.ndarray):
            return torch.from_numpy(audio_input).float(), 16000
        elif isinstance(audio_input, torch.Tensor):
            return audio_input.float(), 16000
        else:
            raise ValueError(f"Unsupported audio format: {type(audio_input)}")
```

- [ ] **Step 3: Implement `_extract_mel_batch` helper**

```python
    def _extract_mel_batch(self, waveforms: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert raw waveforms to padded mel spectrograms.

        Args:
            waveforms: List of 1-D float32 waveforms, each [T_audio].

        Returns:
            (audio_data [B, 128, T_mel_max], audio_data_seqlens [B])
        """
        mels: list[torch.Tensor] = []
        for wav in waveforms:
            mels.append(self.processor._extract_mel(wav))
        seqlens = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long)
        max_len = int(seqlens.max().item())
        audio_data = torch.zeros(len(mels), mels[0].shape[0], max_len)
        for i, m in enumerate(mels):
            audio_data[i, :, : m.shape[-1]] = m
        return audio_data, seqlens
```

- [ ] **Step 4: Implement `_tokenize_sample` helper**

```python
    def _tokenize_sample(
        self,
        audio: torch.Tensor,
        response_text: str,
    ) -> dict[str, list[int]]:
        """Tokenize prompt + response for a single sample.

        Uses the processor's prompt-building logic: the system prompt
        includes "<|audio_bos|><|AUDIO|><|audio_eos|>" which gets expanded
        to the correct number of `<|AUDIO|>` tokens (12.5 per second).

        Returns dict with:
            - input_ids: list of token IDs (prompt + response)
            - audio_input_mask_positions: list of bools (True at <|AUDIO|>)
            - prompt_len: number of prompt tokens
        """
        num_audio_frames = self.processor._conv3_downsample_len(
            int(self.processor._extract_mel(audio).shape[-1])
        )
        audio_placeholder_ids = self.processor._build_audio_placeholder_ids(num_audio_frames)

        # Build prompt token ids:
        # System text + audio_bos + audio_tokens + audio_eos
        prompt = self.SYSTEM_PROMPT
        spans = list(self.processor._AUDIO_SPAN_RE.finditer(prompt))
        if len(spans) != 1:
            raise ValueError(f"Expected 1 audio span in prompt, got {len(spans)}")

        match = spans[0]
        prefix = prompt[: match.start()]
        suffix = prompt[match.end():]

        prefix_ids = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)

        prompt_ids = (
            prefix_ids
            + [self.AUDIO_BOS_ID]
            + audio_placeholder_ids
            + [self.AUDIO_EOS_ID]
            + suffix_ids
        )

        response_ids = self.tokenizer.encode(response_text, add_special_tokens=False)
        # Add eos token
        if self.tokenizer.eos_token_id is not None:
            response_ids.append(self.tokenizer.eos_token_id)

        full_ids = prompt_ids + response_ids

        # Build audio input mask: True at all <|AUDIO|> positions
        audio_input_mask_positions = []
        for tid in full_ids:
            audio_input_mask_positions.append(tid == self.AUDIO_TOKEN_ID)

        return {
            "input_ids": full_ids,
            "audio_input_mask_positions": audio_input_mask_positions,
            "prompt_len": len(prompt_ids),
        }
```

- [ ] **Step 5: Implement the main `__call__` method**

```python
    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch of NVTTS-FACodec rows into DPO format.

        Returns dict with keys:
            input_ids:          [2*B, S_max]  concatenated chosen+rejected
            attention_mask:     [2*B, S_max]
            completion_mask:     [2*B, S_max]  1=response, 0=prompt
            audio_data:          [2*B, 128, T_max]  duplicated
            audio_data_seqlens:  [2*B]              duplicated
            audio_input_mask:    [2*B, S_max]       bool
            prosody_indices:     [2*B, 1, T80_max]  duplicated
            timbre_vector:       [2*B, 256]         duplicated
        """
        batch_size = len(examples)

        # Extract audio waveforms
        waveforms: list[torch.Tensor] = []
        for ex in examples:
            wav, _ = self._extract_audio(ex["audio"])
            waveforms.append(wav)

        # Extract mel spectrograms (B non-duplicated)
        audio_data_nodup, audio_data_seqlens_nodup = self._extract_mel_batch(waveforms)

        # Extract FACodec features
        prosody_list: list[torch.Tensor] = []
        timbre_list: list[torch.Tensor] = []
        for ex in examples:
            p = torch.tensor(ex["prosody_codebooks_idx"], dtype=torch.long)          # [T80]
            p = p.unsqueeze(0)                                                       # [1, T80]
            prosody_list.append(p)

            t = torch.tensor(ex["timbre_vector"], dtype=torch.float32)               # [256]
            timbre_list.append(t)

        # Pad prosody_indices to max T80
        max_t80 = max(p.shape[-1] for p in prosody_list)
        prosody_padded = torch.zeros(batch_size, 1, max_t80, dtype=torch.long)
        for i, p in enumerate(prosody_list):
            prosody_padded[i, :, : p.shape[-1]] = p

        timbre_stacked = torch.stack(timbre_list)  # [B, 256]

        # Tokenize chosen and rejected per sample
        chosen_sequences: list[dict] = []
        rejected_sequences: list[dict] = []
        for ex, wav in zip(examples, waveforms):
            chosen_sequences.append(self._tokenize_sample(wav, ex["chosen"]))
            rejected_sequences.append(self._tokenize_sample(wav, ex["rejected"]))

        # Build padded tensors for chosen + rejected (separately, then concat)
        def pad_sequences(seqs: list[dict], token_pad: int, mask_pad: int) -> dict:
            max_s = max(len(s["input_ids"]) for s in seqs)
            if self.pad_to_multiple_of is not None:
                max_s = ((max_s + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            if self.max_length is not None:
                max_s = min(max_s, self.max_length)

            input_ids = torch.full((len(seqs), max_s), token_pad, dtype=torch.long)
            attention_mask = torch.full((len(seqs), max_s), 0, dtype=torch.long)
            completion_mask = torch.full((len(seqs), max_s), 0, dtype=torch.long)
            audio_mask = torch.full((len(seqs), max_s), False, dtype=torch.bool)

            for i, s in enumerate(seqs):
                seq_len = min(len(s["input_ids"]), max_s)
                input_ids[i, :seq_len] = torch.tensor(s["input_ids"][:seq_len], dtype=torch.long)
                attention_mask[i, :seq_len] = 1
                # Completion mask: 1 for response tokens (after prompt)
                prompt_len = min(s["prompt_len"], max_s)
                completion_mask[i, prompt_len:seq_len] = 1
                audio_mask[i, :seq_len] = torch.tensor(
                    s["audio_input_mask_positions"][:seq_len], dtype=torch.bool
                )

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "completion_mask": completion_mask,
                "audio_input_mask": audio_mask,
            }

        chosen_padded = pad_sequences(chosen_sequences, self.pad_token_id, 0)
        rejected_padded = pad_sequences(rejected_sequences, self.pad_token_id, 0)

        # Concatenate chosen + rejected [B] → [2*B] (standard DPO format)
        input_ids = torch.cat([chosen_padded["input_ids"], rejected_padded["input_ids"]], dim=0)
        attention_mask = torch.cat([chosen_padded["attention_mask"], rejected_padded["attention_mask"]], dim=0)
        completion_mask = torch.cat([chosen_padded["completion_mask"], rejected_padded["completion_mask"]], dim=0)
        audio_input_mask = torch.cat([chosen_padded["audio_input_mask"], rejected_padded["audio_input_mask"]], dim=0)

        # Duplicate audio/FACodec fields: same speech for chosen and rejected
        audio_data = torch.cat([audio_data_nodup, audio_data_nodup], dim=0)
        audio_data_seqlens = torch.cat([audio_data_seqlens_nodup, audio_data_seqlens_nodup], dim=0)
        prosody_indices = torch.cat([prosody_padded, prosody_padded], dim=0)
        timbre_vector = torch.cat([timbre_stacked, timbre_stacked], dim=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "audio_data": audio_data,
            "audio_data_seqlens": audio_data_seqlens,
            "audio_input_mask": audio_input_mask,
            "prosody_indices": prosody_indices,
            "timbre_vector": timbre_vector,
        }
```

### Task 3: Write unit tests for DPOCollator

**Files:**
- Create: `tests/training/test_dpo_collator.py`

- [ ] **Step 1: Create test file with fixtures and basic shape tests**

```python
"""Unit tests for DPOCollator."""

import pytest
import torch
import numpy as np

# Mock processor that doesn't need GPU
class _MockTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        # Deterministic mock: each char → token ID
        return [max(1, ord(c) % 1000) for c in text if c != " "]


class _MockProcessor:
    """Minimal mock of MossAudioProcessor for unit tests."""

    import re
    _AUDIO_SPAN_RE = re.compile(
        r"<\|audio_bos\|>(?:<\|AUDIO\|>)+<\|audio_eos\|>"
    )

    def __init__(self):
        self._base_tokenizer = _MockTokenizer()

    def _extract_mel(self, waveform):
        """Return a fake mel: 128 x ceil(duration * 100)."""
        # waveform is [T_audio]; T_mel ~ T_audio / 160
        t_mel = max(1, waveform.shape[0] // 160)
        return torch.randn(128, t_mel)

    def _conv3_downsample_len(self, raw_len):
        return max(1, raw_len * 2) // 2

    def _build_audio_placeholder_ids(self, num_audio_tokens):
        return [151654] * num_audio_tokens


@pytest.fixture
def processor():
    return _MockProcessor()


@pytest.fixture
def collator(processor):
    from src.training.dpo_collator import DPOCollator

    return DPOCollator(
        processor=processor,
        pad_token_id=0,
        max_length=512,
    )


def _make_example(audio_len=16000, prosody_len=80, timbre_dim=256):
    """Create a single synthetic NVTTS-FACodec row."""
    return {
        "audio": np.random.randn(audio_len).astype(np.float32),
        "prosody_codebooks_idx": list(np.random.randint(0, 1024, prosody_len)),
        "timbre_vector": list(np.random.randn(timbre_dim).astype(np.float32)),
        "chosen": "The speaker sounds happy and engaged.",
        "rejected": "The speaker sounds neutral.",
        "cosine_similarity": 0.72,
    }
```

- [ ] **Step 2: Test basic shape correctness**

```python
class TestDPOCollatorShapes:
    def test_output_shapes_batch_size_2(self, collator):
        examples = [_make_example() for _ in range(2)]
        batch = collator(examples)

        B = 2
        assert batch["input_ids"].shape[0] == 2 * B          # concatenated
        assert batch["attention_mask"].shape[0] == 2 * B
        assert batch["completion_mask"].shape[0] == 2 * B
        assert batch["audio_data"].shape[0] == 2 * B
        assert batch["audio_data"].shape[1] == 128            # mel_dim
        assert batch["prosody_indices"].shape[0] == 2 * B
        assert batch["prosody_indices"].shape[1] == 1         # single codebook
        assert batch["timbre_vector"].shape[0] == 2 * B
        assert batch["timbre_vector"].shape[1] == 256
        assert batch["audio_input_mask"].shape[0] == 2 * B
        assert batch["audio_input_mask"].shape[1] == batch["input_ids"].shape[1]

    def test_audio_fields_duplicated_identically(self, collator):
        examples = [_make_example() for _ in range(2)]
        batch = collator(examples)
        B = 2
        # First half == second half (for audio fields)
        assert torch.equal(batch["audio_data"][:B], batch["audio_data"][B:])
        assert torch.equal(batch["prosody_indices"][:B], batch["prosody_indices"][B:])
        assert torch.equal(batch["timbre_vector"][:B], batch["timbre_vector"][B:])

    def test_completion_mask_structure(self, collator):
        examples = [_make_example() for _ in range(2)]
        batch = collator(examples)
        # completion_mask should have 0s for prompt, 1s for response
        # Check that no mask starts with 1 (prompt always first)
        for i in range(batch["completion_mask"].shape[0]):
            mask = batch["completion_mask"][i]
            nonzero = (mask == 1).nonzero(as_tuple=True)[0]
            if len(nonzero) > 0:
                # First 1 marks start of response
                first_one = nonzero[0].item()
                assert mask[:first_one].sum() == 0  # all zeros before first 1
                assert mask[first_one:].sum() > 0   # some ones in response

    def test_audio_input_mask_positions(self, collator):
        examples = [_make_example(audio_len=16000) for _ in range(1)]
        batch = collator(examples)
        audio_mask = batch["audio_input_mask"][0]  # first chosen sequence
        # Should have at least one True position (<|AUDIO|> tokens)
        assert audio_mask.any(), "audio_input_mask has no True positions"
        # All True positions should correspond to token ID 151654
        input_ids = batch["input_ids"][0]
        for pos in audio_mask.nonzero(as_tuple=True)[0]:
            assert input_ids[pos].item() == 151654

    def test_padding_consistent_across_fields(self, collator):
        examples = [_make_example(audio_len=16000) for _ in range(2)]
        batch = collator(examples)
        S = batch["input_ids"].shape[1]
        assert batch["attention_mask"].shape[1] == S
        assert batch["completion_mask"].shape[1] == S
        assert batch["audio_input_mask"].shape[1] == S

    def test_batch_size_1(self, collator):
        examples = [_make_example() for _ in range(1)]
        batch = collator(examples)
        assert batch["input_ids"].shape[0] == 2  # 1*2 for concatenated
```

### Task 4: Run tests and fix failures

- [ ] **Step 1: Run the collator tests**

```bash
uv run python -m pytest tests/training/test_dpo_collator.py -v
```

Expected: All 7 tests pass after implementation fixes.

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml uv.lock src/training/dpo_collator.py tests/training/test_dpo_collator.py
git commit -m "feat: add DPOCollator for AmyLM DPO training"
```

## Phase Completion Criteria
- [ ] `trl`, `peft` importable
- [ ] `DPOCollator.__call__` produces batch with correct shapes and key fields
- [ ] Audio fields are identically duplicated (B → 2×B)
- [ ] `completion_mask` correctly identifies prompt vs. response tokens
- [ ] `audio_input_mask` correctly flags `<|AUDIO|>` (151654) positions
- [ ] All 7 unit tests pass

## Handoff Notes

Phase 2 (AmyDPOTrainer) needs the collator to be fully functional. The collator produces batches that `DPOTrainer._compute_loss` passes to `model()` via standard `model_kwargs` passthrough. The `audio_data`, `prosody_indices`, `timbre_vector`, `audio_data_seqlens`, `audio_input_mask` fields are already duplicated to `[2×B]` shape, matching the concatenated `input_ids` layout.
