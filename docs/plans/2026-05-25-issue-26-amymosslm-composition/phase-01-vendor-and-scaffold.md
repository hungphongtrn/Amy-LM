# Phase 1: Vendor and Scaffold

## Phase Goal
`src/models/moss_audio_model.py` exists as a single vendored file containing both `MossAudioConfig` and `MossAudioModel` classes. `src/models/moss_audio.py` is deleted. `src/models/__init__.py` exports reflect the new `AmyMossLM`/`AmyMossLMConfig` naming. Old `AmyLM`/`AmyLMConfig` imports removed from `amy_lm.py`.

## Files to Touch

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/moss_audio_model.py` | **Create** | Vendored MossAudio source (config + model) |
| `src/models/amy_lm.py` | **Edit** | Remove vendor path hacks; update imports to local |
| `src/models/moss_audio.py` | **Delete** | Replaced by `moss_audio_model.py` |
| `src/models/__init__.py` | **Edit** | Update exports for new naming |

## Pre-requisite Verification

- [ ] Verify `liger_kernel` is available: `uv run python -c "from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3; print('liger OK')"`
- [ ] Verify vendor source compiles: `uv run python -c "import sys; sys.path.insert(0, 'vendor/MOSS-Audio/src'); from modeling_moss_audio import MossAudioModel; print('vendor OK')"`
- [ ] Existing test suite passes pre-refactor: `uv run python -m pytest tests/ -x --timeout=60 -q`

## Tasks

### Task 1: Create vendored `moss_audio_model.py`

**Files:**
- Create: `src/models/moss_audio_model.py`

- [ ] **Step 1: Concatenate vendor source files into a single file**

Concatenate `vendor/MOSS-Audio/src/configuration_moss_audio.py` (lines 1-126) + `vendor/MOSS-Audio/src/modeling_moss_audio.py` (lines 1-606) into `src/models/moss_audio_model.py`. 

Changes from the vendor originals:
1. **MossAudioConfig language_config**: Set `_attn_implementation` to `"flash_attention_2"` in `language_config` before constructing. Add this inside `MossAudioConfig.__init__()`:

```python
# Inject flash_attention_2 into language_config (ADR #0001, AmyMossLM composition)
if isinstance(language_config, dict):
    language_config["_attn_implementation"] = "flash_attention_2"
elif hasattr(language_config, "_attn_implementation"):
    language_config._attn_implementation = "flash_attention_2"
```

2. **Liger kernel activation**: Add at the top of the file, after imports:

```python
try:
    from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3()
except ImportError:
    import warnings
    warnings.warn("liger_kernel not available. Qwen3 will use default attention implementation.")
```

3. **Vendor-to-local import fix**: Since both config and model are now in the same file, change `from configuration_moss_audio import ...` to just use the classes defined above in the same file. Remove the vendor import line entirely (classes are already defined earlier in the file).

4. **Add Amy-specific `_no_split_modules`**: Since `AmyMossLM` composes `MossAudioModel` but won't inherit `MossAudioPreTrainedModel`, add `MossAudioModel._no_split_modules = ["Qwen3DecoderLayer", "WhisperEncoderLayer"]` as a class attribute for gradient checkpointing support.

The final file structure:
```
1-126:   MossAudioEncoderConfig + MossAudioConfig (from configuration_moss_audio.py, with flash_attn + liger additions)
127-732: SinusoidsPositionEmbedding + MossAudioEncoder + GatedMLP + MossAudioPreTrainedModel + MossAudioModel (from modeling_moss_audio.py)
```

- [ ] **Step 2: Verify the vendored file compiles without errors**

```bash
uv run python -c "from src.models.moss_audio_model import MossAudioConfig, MossAudioModel, MossAudioEncoderConfig; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Verify liger kernel applied**

```bash
uv run python -c "from src.models.moss_audio_model import MossAudioConfig; print('liger imported OK')"
```

Expected: `liger imported OK` (no ImportError warning)

- [ ] **Step 4: Verify flash attention configured**

```bash
uv run python -c "
from src.models.moss_audio_model import MossAudioConfig
config = MossAudioConfig()
print(config.language_config._attn_implementation)
"
```

Expected: `flash_attention_2`

- [ ] **Step 5: Commit**

```bash
git add src/models/moss_audio_model.py
git commit -m "feat(phase-1): vendor MossAudio source into moss_audio_model.py with liger + flash_attn"
```

### Task 2: Update `src/models/amy_lm.py` imports

**Files:**
- Edit: `src/models/amy_lm.py`

- [ ] **Step 1: Remove vendor path setup block (lines 22-26)**

Delete the entire vendor path setup:
```python
# Vendor path setup (same pattern as src/models/moss_audio.py)
_VENDOR_MOSS_AUDIO_SRC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src")
)
if os.path.isdir(_VENDOR_MOSS_AUDIO_SRC_PATH) and _VENDOR_MOSS_AUDIO_SRC_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_MOSS_AUDIO_SRC_PATH)
```

- [ ] **Step 2: Replace vendor imports with local imports**

Replace:
```python
from configuration_moss_audio import MossAudioConfig
from modeling_moss_audio import MossAudioModel
```

With:
```python
from .moss_audio_model import MossAudioConfig, MossAudioModel
```

- [ ] **Step 3: Verify file still compiles (will fail due to old class names in forward tests, but imports should resolve)**

```bash
uv run python -c "from src.models.amy_lm import AmyLMConfig, AmyLM; print('imports OK')"
```

Expected: `imports OK`

- [ ] **Step 4: Commit**

```bash
git add src/models/amy_lm.py
git commit -m "refactor(phase-1): remove vendor path from amy_lm.py, use local moss_audio_model imports"
```

### Task 3: Delete `src/models/moss_audio.py`

**Files:**
- Delete: `src/models/moss_audio.py`

- [ ] **Step 1: Verify no other files import from `moss_audio.py`**

```bash
rg "from.*moss_audio import|import.*moss_audio" --include="*.py" src/ tests/ scripts/ | grep -v moss_audio_model | grep -v "vendor/"
```

Expected: Only hits from `src/models/__init__.py` (which we'll update in Task 4) and possibly from test files that import `MossAudioWrapper`. Note any additional consumers for context.

- [ ] **Step 2: Delete the file**

```bash
rm src/models/moss_audio.py
```

- [ ] **Step 3: Commit**

```bash
git rm src/models/moss_audio.py
git commit -m "refactor(phase-1): delete moss_audio.py (replaced by moss_audio_model.py)"
```

### Task 4: Update `src/models/__init__.py` exports

**Files:**
- Edit: `src/models/__init__.py`

- [ ] **Step 1: Replace old exports with new naming**

Replace the imports section:
```python
from .embedding import ProsodyEmbedding, TimbreProjection, AcousticEmbedding, ContentEmbedding
from .pooling import TemporalPool
from .fusion import ResidualFusion
from .moss_audio import MossAudioWrapper
from .amy_classifier import AmyForProsodyClassification
from .baseline_classifier import BaselineClassifier
from .amy_lm import AmyLMConfig, AmyLM

__all__ = [
    "ProsodyEmbedding", "TimbreProjection", "AcousticEmbedding", "ContentEmbedding",
    "TemporalPool", "ResidualFusion",
    "MossAudioWrapper",
    "AmyForProsodyClassification", "BaselineClassifier",
    "AmyLMConfig", "AmyLM",
]
```

With:
```python
from .embedding import ProsodyEmbedding, TimbreProjection, AcousticEmbedding, ContentEmbedding
from .pooling import TemporalPool
from .fusion import ResidualFusion
from .moss_audio_model import MossAudioConfig, MossAudioModel
from .amy_classifier import AmyForProsodyClassification
from .baseline_classifier import BaselineClassifier
from .amy_lm import AmyMossLMConfig, AmyMossLM

__all__ = [
    "ProsodyEmbedding", "TimbreProjection", "AcousticEmbedding", "ContentEmbedding",
    "TemporalPool", "ResidualFusion",
    "MossAudioConfig", "MossAudioModel",
    "AmyForProsodyClassification", "BaselineClassifier",
    "AmyMossLMConfig", "AmyMossLM",
]
```

Note: `MossAudioWrapper` is deleted. If consumers exist (e.g., `amy_classifier.py`), they will break in this phase and be fixed in Phase 3 (or a follow-up). For now, we're doing the foundation rename.

- [ ] **Step 2: Verify the module compiles**

```bash
uv run python -c "from src.models import MossAudioConfig, MossAudioModel; print('exports OK')"
```

Expected: `exports OK` (AmyMossLM imports will fail until Phase 2 rewrites the class)

- [ ] **Step 3: Commit**

```bash
git add src/models/__init__.py
git commit -m "refactor(phase-1): update __init__.py exports for MossAudio vendoring"
```

## Phase Completion Criteria
- [ ] `src/models/moss_audio_model.py` exists and compiles with liger + flash_attn
- [ ] `src/models/moss_audio.py` is deleted
- [ ] `src/models/amy_lm.py` imports from local `moss_audio_model` (no vendor path)
- [ ] `src/models/__init__.py` exports reflect new naming (`MossAudioConfig`, `MossAudioModel`, `AmyMossLMConfig`, `AmyMossLM`)
- [ ] `git log --oneline -5` shows 4 commits for this phase

## Handoff Notes
Phase 2 needs the vendored file to be correct because `AmyMossLM.__init__()` will call `MossAudioModel(config)` directly from this local import. The `MossAudioConfig` flash_attn injection must work because the `MossAudioModel` constructor immediately builds `self.language_model = Qwen3Model(config.language_config)`.
