# Phase 2: Speaker Context Lookup Cache — Issue #18

## Phase Goal
Module `src/data/speaker_cache.py` that merges 3 VoxCeleb metadata sources + Expresso into a cascading-fallback JSON cache at `data/speaker_lookup.json`.

## Files to Touch

| File | Action | Purpose |
|------|--------|---------|
| `src/data/speaker_cache.py` | Create | Download, merge, cache speaker metadata |
| `data/speaker_lookup.json` | Generate (by script) | Cached speaker lookup dict |
| `tests/data/test_speaker_cache.py` | Create | Unit tests with mocked HF sources |
| `src/data/__init__.py` | Update | Export `SpeakerCache`, `build_speaker_cache` |

## Architecture

```python
# src/data/speaker_cache.py

EXPRESSO_SPEAKERS: dict[str, dict] = {
    "ex01": {"name": "Jack", "gender": "Male", "nationality": "American", "age": None, "birth_year": None},
    "ex02": {"name": "Lisa", "gender": "Female", "nationality": "American", "age": None, "birth_year": None},
    "ex03": {"name": "Bert", "gender": "Male", "nationality": "American", "age": None, "birth_year": None},
    "ex04": {"name": "Emma", "gender": "Female", "nationality": "American", "age": None, "birth_year": None},
}

# Expresso speakers: hardcoded, North American English
# Age/birth_year not available for Expresso VAs

class SpeakerCache:
    def build(...) -> dict[str, dict]:
        """Download and merge all sources → {speaker_id: {name, gender, age, birth_year, nationality}}"""
    
    def resolve_age_context(speaker: dict) -> str:
        """Format age context: '{age}, born {birth_year}' or 'unknown'"""
    
    def save(path) / load(path)
```

### Cascading Fallback Priority
For each speaker_id, each field resolved independently:
1. **enrichment** (hechmik/voxceleb_enrichment_age_gender) — richest: age, birth_year, name, gender, nationality
2. **vox1_meta** (ProgramComputer/voxceleb vox1_meta.csv) — name, gender, nationality
3. **language-metadata** (johbac/voxceleb-language-metadata) — name only (fallback)
4. **Expresso hardcoded** — name, gender, nationality (no age)
5. **"unknown"** — if all miss

### Speaker ID format resolution
- Expresso: `"ex01"` through `"ex04"`
- VoxCeleb: `"id00012"`, `"id03621"` (7-char zero-padded)
- NVTTS uses: `"ex03"` for Expresso, `"id03621"` for VoxCeleb

## Tasks

### Task 2.1: Write the failing test

**Files:**
- Create: `tests/data/test_speaker_cache.py`
- Create: `tests/data/__init__.py` (if not already)

```python
"""Tests for Speaker Context Lookup Cache — Issue #18."""
import json
import os
import tempfile
import pytest
from src.data.speaker_cache import (
    SpeakerCache,
    EXPRESSO_SPEAKERS,
    resolve_age_context,
    build_speaker_cache,
)


class TestExpressoSpeakers:
    
    def test_all_four_expresso_speakers(self):
        """All 4 Expresso speakers are defined."""
        assert len(EXPRESSO_SPEAKERS) == 4
        assert "ex01" in EXPRESSO_SPEAKERS
        assert "ex02" in EXPRESSO_SPEAKERS
        assert "ex03" in EXPRESSO_SPEAKERS
        assert "ex04" in EXPRESSO_SPEAKERS
    
    def test_expresso_names(self):
        """Expresso speakers have correct names."""
        assert EXPRESSO_SPEAKERS["ex01"]["name"] == "Jack"
        assert EXPRESSO_SPEAKERS["ex02"]["name"] == "Lisa"
        assert EXPRESSO_SPEAKERS["ex03"]["name"] == "Bert"
        assert EXPRESSO_SPEAKERS["ex04"]["name"] == "Emma"
    
    def test_expresso_genders(self):
        """2 Male, 2 Female."""
        genders = [s["gender"] for s in EXPRESSO_SPEAKERS.values()]
        assert genders.count("Male") == 2
        assert genders.count("Female") == 2
    
    def test_expresso_nationality(self):
        """All Expresso speakers are North American English."""
        for sid, spk in EXPRESSO_SPEAKERS.items():
            assert spk["nationality"] == "American", f"{sid} should be American"


class TestAgeContextFormatting:
    
    def test_age_with_birth_year(self):
        result = resolve_age_context({"age": 45, "birth_year": 1979})
        assert result == "45, born 1979"
    
    def test_age_without_birth_year(self):
        result = resolve_age_context({"age": 45})
        assert result == "45"
    
    def test_birth_year_without_age(self):
        result = resolve_age_context({"birth_year": 1979})
        assert result == "born 1979"
    
    def test_neither_age_nor_birth_year(self):
        result = resolve_age_context({})
        assert result == "unknown"


class TestSpeakerCache:
    
    @pytest.fixture
    def cache(self):
        """Build cache from mock data (no network)."""
        return SpeakerCache.from_mock()
    
    def test_expresso_speaker_resolves(self, cache):
        """Expresso speaker ex03 resolves with hardcoded data."""
        speaker = cache.lookup("ex03")
        assert speaker is not None
        assert speaker["name"] == "Bert"
        assert speaker["gender"] == "Male"
    
    def test_voxceleb_speaker_resolves(self, cache):
        """Known VoxCeleb ID resolves from mock data."""
        speaker = cache.lookup("id03621")
        assert speaker is not None
        assert speaker["name"] != "unknown"
    
    def test_unknown_speaker_returns_unknown(self, cache):
        """Unknown speaker ID returns 'unknown' for all fields."""
        speaker = cache.lookup("id99999")
        assert speaker is not None
        assert speaker["name"] == "unknown"
        assert speaker["gender"] == "unknown"
        assert speaker["nationality"] == "unknown"
    
    def test_lookup_returns_required_fields(self, cache):
        """Lookup returns all required fields."""
        speaker = cache.lookup("ex01")
        for field in ["name", "gender", "nationality", "age", "birth_year"]:
            assert field in speaker, f"Missing field: {field}"
    
    def test_age_context_on_resolved_speaker(self, cache):
        """Age context can be derived from resolved speaker."""
        speaker = cache.lookup("ex01")
        age_ctx = resolve_age_context(speaker)
        assert isinstance(age_ctx, str)
        assert len(age_ctx) > 0


class TestCachePersistence:
    
    def test_save_and_load_roundtrip(self):
        """Cache can be saved to JSON and loaded back."""
        cache = SpeakerCache.from_mock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache.save(f.name)
            save_path = f.name
        
        try:
            loaded = SpeakerCache.load(save_path)
            assert loaded.lookup("ex03")["name"] == "Bert"
        finally:
            os.unlink(save_path)
    
    def test_json_is_valid(self):
        """Saved JSON is parseable by json.load."""
        cache = SpeakerCache.from_mock()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            cache.save(f.name)
            save_path = f.name
        
        try:
            with open(save_path) as f:
                data = json.load(f)
            assert isinstance(data, dict)
            assert "ex01" in data
        finally:
            os.unlink(save_path)
```

- [ ] **Step 1: Write the failing test**

Write the test file above.

- [ ] **Step 2: Run test to verify it fails**

```bash
uv run python -m pytest tests/data/test_speaker_cache.py -v
```
Expected: FAIL with import errors

- [ ] **Step 3: Write minimal implementation**

Create `src/data/speaker_cache.py` with:
- `EXPRESSO_SPEAKERS` dict
- `resolve_age_context()` function
- `SpeakerCache` class with `from_mock()`, `lookup()`, `save()`, `load()`, `build()`

The `from_mock()` method provides test data without network access. The `build()` method downloads real data from HF.

- [ ] **Step 4: Run test to verify it passes**

```bash
uv run python -m pytest tests/data/test_speaker_cache.py -v
```
Expected: PASS (all tests)

- [ ] **Step 5: Build real cache (optional — may need network)**

```bash
uv run python -c "from src.data.speaker_cache import build_speaker_cache; build_speaker_cache()"
```
Expected: `data/speaker_lookup.json` created

- [ ] **Step 6: Commit**

```bash
git add src/data/speaker_cache.py tests/data/test_speaker_cache.py src/data/__init__.py
git commit -m "feat: Speaker Context Lookup Cache — cascading VoxCeleb+Expresso merge (#18)"
```

## Gotchas

1. **HF downloads**: The 3 VoxCeleb sources require `datasets` and `requests`. Mock in tests to avoid network dependency. `from_mock()` provides enough data for testing the merge logic.

2. **Speaker ID normalization**: VoxCeleb1 uses `id00012` format. NVTTS may drop leading zeros (`id12`). The lookup must handle both or normalize to a canonical format.

3. **CSV parsing**: `vox1_meta.csv` columns: `VoxCeleb1 ID, Name, Gender, Nationality`. The `Name` field is just the speaker's first/last name.

4. **Age field**: enrichment has an `Age` field — this is a snapshot age (e.g., "Howard was 35 when recorded"), not continuously updated. The `birth_year` allows computing age at any reference date — but for the prompt, just format as-is.

## Phase Completion Criteria
- [ ] `EXPRESSO_SPEAKERS` dict with all 4 speakers
- [ ] `resolve_age_context()` formats age correctly
- [ ] `SpeakerCache.lookup()` resolves Expresso IDs
- [ ] `SpeakerCache.lookup()` resolves VoxCeleb IDs (from mock data)
- [ ] Unknown speakers return "unknown" gracefully
- [ ] Save/load roundtrip preserves data
- [ ] All tests pass
