# Phase 1: NV Tag Emoji Mapping — Issue #17

## Phase Goal
Module `src/data/nv_tag_mapping.py` with bidirectional emoji→text mapping and transform function. All 10 NV types covered, tested.

## Files to Touch

| File | Action | Purpose |
|------|--------|---------|
| `src/data/nv_tag_mapping.py` | Create | Mapping dict + `emojis_to_tags()` function |
| `src/data/__init__.py` | Create | Package init, exports `emojis_to_tags`, `NV_TAG_MAP` |
| `tests/data/test_nv_tag_mapping.py` | Create | Unit tests for mapping + transform |
| `tests/data/__init__.py` | Create | Package init (may exist) |

## Architecture

```python
# src/data/nv_tag_mapping.py
import re

# Bidirectional: emoji → text label
NV_EMOJI_TO_TAG: dict[str, str] = {
    "🌬️": "[Breathing]",
    "🤣": "[Laughter]",
    "😮‍💨": "[Sigh]",
    "🤧": "[Sneeze]",
    "😷": "[Cough]",
    "😤": "[Throat clear]",
    "😩": "[Groan]",
    "😫": "[Grunt]",
    "😴": "[Snore]",
    "😤": "[Sniff]",   # Note: may overlap — resolve per NVTTS spec
}

# Reverse mapping: text label → emoji
NV_TAG_TO_EMOJI: dict[str, str] = {v: k for k, v in NV_EMOJI_TO_TAG.items()}

def emojis_to_tags(text: str) -> str:
    """Replace NV emoji symbols in text with [Tag] labels.
    
    Args:
        text: NVTTS Result column text containing emoji symbols
        
    Returns:
        Text with emojis replaced by canonical [Tag] labels
    """
    result = text
    for emoji, tag in NV_EMOJI_TO_TAG.items():
        result = result.replace(emoji, f" {tag} ")
    # Collapse multiple spaces
    result = re.sub(r'\s+', ' ', result).strip()
    return result
```

## Tasks

### Task 1.1: Write the failing test

**Files:**
- Create: `tests/data/test_nv_tag_mapping.py`
- Create: `tests/data/__init__.py` (empty)

```python
"""Tests for NV Tag Emoji Mapping — Issue #17."""
import pytest
from src.data.nv_tag_mapping import NV_EMOJI_TO_TAG, NV_TAG_TO_EMOJI, emojis_to_tags


class TestNVTagMapping:
    
    def test_all_10_nv_types_covered(self):
        """All 10 NV types are in the mapping."""
        expected_tags = [
            "[Breathing]", "[Laughter]", "[Sigh]", "[Sneeze]", "[Cough]",
            "[Throat clear]", "[Groan]", "[Grunt]", "[Snore]", "[Sniff]",
        ]
        for tag in expected_tags:
            assert tag in NV_EMOJI_TO_TAG.values(), f"Missing mapping for {tag}"
        assert len(set(NV_EMOJI_TO_TAG.values())) == 10, (
            f"Expected 10 unique tags, got {len(set(NV_EMOJI_TO_TAG.values()))}"
        )
    
    def test_bidirectional_mapping(self):
        """Every emoji maps to a tag and vice versa."""
        for emoji, tag in NV_EMOJI_TO_TAG.items():
            assert NV_TAG_TO_EMOJI[tag] == emoji, f"Roundtrip failed for {emoji} ↔ {tag}"
        assert len(NV_TAG_TO_EMOJI) == len(NV_EMOJI_TO_TAG)
    
    def test_emojis_to_tags_single_emoji(self):
        """Single emoji replaced by its tag."""
        result = emojis_to_tags("Hello 🤣")
        assert result == "Hello [Laughter]"
    
    def test_emojis_to_tags_multiple_emojis(self):
        """Multiple emojis replaced by their tags."""
        result = emojis_to_tags("I am 🌬️ and then 🤣 at the same time")
        assert "[Breathing]" in result
        assert "[Laughter]" in result
    
    def test_emojis_to_tags_no_emojis(self):
        """Text with no emojis is unchanged (except whitespace normalization)."""
        result = emojis_to_tags("Plain text without any emoji")
        assert result == "Plain text without any emoji"
    
    def test_emojis_to_tags_empty_string(self):
        """Empty string returns empty string."""
        result = emojis_to_tags("")
        assert result == ""
    
    def test_emojis_to_tags_whitespace_normalization(self):
        """Multiple spaces collapsed after replacement."""
        result = emojis_to_tags("Word  🤣  word")
        assert result == "Word [Laughter] word"
    
    def test_representative_nvtts_row(self):
        """Transform on a sample NVTTS Result string."""
        sample = "Hello 🤣 everyone. I am feeling 🌬️ today."
        result = emojis_to_tags(sample)
        assert "[Laughter]" in result
        assert "[Breathing]" in result
        assert "Hello [Laughter] everyone" in result
```

- [ ] **Step 1: Write the failing test**

Write the test file above.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/data/test_nv_tag_mapping.py -v
```
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.nv_tag_mapping'`

- [ ] **Step 3: Write minimal implementation**

Create `src/data/__init__.py` (empty) and `src/data/nv_tag_mapping.py` with the mapping dict and function.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/data/test_nv_tag_mapping.py -v
```
Expected: PASS (all 8 tests)

- [ ] **Step 5: Verify no regressions**

```bash
uv run python -m pytest tests/ -x -q
```

- [ ] **Step 6: Commit**

```bash
git add src/data/ tests/data/
git commit -m "feat: NV Tag Emoji Mapping — emoji to [Tag] text transform (#17)"
```

## Gotchas

1. **Emoji overlapping**: 😤 is used for BOTH "Throat clearing" and "Sniffing" in some NVTTS documentation. The issue spec says all 10 are distinct. Verify the actual emoji mapping from the NVTTS dataset metadata — the emojis listed above are placeholders. May need to check `deepvk/NonverbalTTS` directly for the exact emoji encodings in the `Result` column.

2. **Emoji encoding**: Some emojis use ZWJ sequences (zero-width joiner) — e.g., 😮‍💨 for "Sigh". These are multi-codepoint sequences. The Python string must match exactly, including invisible ZWJ characters.

3. **Whitespace**: Tags should have a space before/after so they tokenize as separate tokens. The collapse step ensures no double spaces.

## Phase Completion Criteria
- [ ] `src/data/nv_tag_mapping.py` exists with `NV_EMOJI_TO_TAG`, `NV_TAG_TO_EMOJI`, `emojis_to_tags()`
- [ ] All 10 NV types covered (verified by test)
- [ ] Bidirectional mapping works (roundtrip test)
- [ ] Transform handles empty string, no emoji input, multiple emojis
- [ ] All tests pass
