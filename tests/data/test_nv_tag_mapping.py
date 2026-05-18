"""Tests for NV Tag Emoji Mapping -- Issue #17."""
import pytest
from src.data.nv_tag_mapping import NV_EMOJI_TO_TAG, NV_TAG_TO_EMOJI, emojis_to_tags


EXPECTED_TAGS = [
    "[Breathing]",
    "[Laughter]",
    "[Sigh]",
    "[Sneeze]",
    "[Cough]",
    "[Throat clear]",
    "[Groan]",
    "[Grunt]",
    "[Snore]",
    "[Sniff]",
]


class TestNVTagMappingDict:
    def test_all_10_nv_types_covered(self):
        """All 10 NV types have at least one emoji mapping."""
        tag_values = set(NV_EMOJI_TO_TAG.values())
        for tag in EXPECTED_TAGS:
            assert tag in tag_values, f"Missing tag: {tag}"
        assert len(tag_values) <= 10, (
            f"Expected at most 10 unique tags, got {len(tag_values)}"
        )

    def test_all_values_are_bracket_format(self):
        """All text tags use the [Tag] bracket format."""
        for tag in NV_EMOJI_TO_TAG.values():
            assert tag.startswith("[") and tag.endswith("]"), (
                f"Tag '{tag}' should use [Tag] format"
            )

    def test_bidirectional_roundtrip(self):
        """Every emoji maps to a tag and tag maps back to some emoji."""
        for emoji, tag in NV_EMOJI_TO_TAG.items():
            assert tag in NV_TAG_TO_EMOJI, f"Missing reverse mapping for {tag}"
            # Reverse mapping may map to a different emoji variant (e.g., with FE0F)
            reverse_emoji = NV_TAG_TO_EMOJI[tag]
            # Strip FE0F for comparison
            emoji_stripped = emoji.replace("\ufe0f", "")
            reverse_stripped = reverse_emoji.replace("\ufe0f", "")
            assert emoji_stripped == reverse_stripped, (
                f"Roundtrip mismatch: {emoji} -> {tag} -> {reverse_emoji}"
            )

    def test_no_duplicate_tags(self):
        """Each tag value appears at most once (excluding FE0F variants)."""
        tags_used = set()
        for tag in NV_EMOJI_TO_TAG.values():
            assert tag not in tags_used, f"Duplicate tag: {tag}"
            tags_used.add(tag)


class TestEmojisToTagsTransform:
    def test_single_emoji_replaced(self):
        """Single emoji replaced by its [Tag] label."""
        e = list(NV_EMOJI_TO_TAG.keys())[0]
        tag = NV_EMOJI_TO_TAG[e]
        result = emojis_to_tags(f"Hello {e}")
        assert result == f"Hello {tag}"

    def test_multiple_emojis_replaced(self):
        """Multiple different emojis all replaced."""
        emojis = list(NV_EMOJI_TO_TAG.keys())
        if len(emojis) < 2:
            pytest.skip("Need at least 2 emojis")
        text = f"Start {emojis[0]} middle {emojis[1]} end"
        result = emojis_to_tags(text)
        for e in emojis[:2]:
            tag = NV_EMOJI_TO_TAG[e]
            assert tag in result, f"Missing tag {tag} for emoji {e}"

    def test_no_emojis_unchanged(self):
        """Text with no emojis is returned unchanged (except whitespace normalization)."""
        result = emojis_to_tags("Plain text without any emoji")
        assert result == "Plain text without any emoji"

    def test_empty_string(self):
        """Empty string returns empty string."""
        result = emojis_to_tags("")
        assert result == ""

    def test_whitespace_normalization(self):
        """Multiple spaces collapsed after replacement."""
        e = list(NV_EMOJI_TO_TAG.keys())[0]
        tag = NV_EMOJI_TO_TAG[e]
        result = emojis_to_tags(f"Word  {e}  word")
        assert result == f"Word {tag} word"

    def test_emoji_with_variation_selector(self):
        """Emoji with U+FE0F variation selector is handled."""
        e = list(NV_EMOJI_TO_TAG.keys())[0]
        tag = NV_EMOJI_TO_TAG[e]
        # Add FE0F if not already present
        if "\ufe0f" not in e:
            e_variant = e + "\ufe0f"
        else:
            e_variant = e
        result = emojis_to_tags(f"Hi {e_variant} there")
        assert tag in result, f"Failed to handle {repr(e_variant)}"

    def test_representative_nvtts_row(self):
        """Transform a representative NVTTS Result string."""
        e = list(NV_EMOJI_TO_TAG.keys())[0]
        tag = NV_EMOJI_TO_TAG[e]
        sample = f"Hello {e} everyone. I am feeling great today."
        result = emojis_to_tags(sample)
        assert tag in result
        assert "Hello" in result

    def test_consecutive_emojis(self):
        """Consecutive emojis are each replaced."""
        emojis = list(NV_EMOJI_TO_TAG.keys())
        if len(emojis) < 2:
            pytest.skip("Need at least 2 emojis")
        text = f"Text {emojis[0]}{emojis[1]} here"
        result = emojis_to_tags(text)
        for e in emojis[:2]:
            tag = NV_EMOJI_TO_TAG[e]
            assert tag in result
