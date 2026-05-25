# Phase 2: Rewrite AmyMossLM

## Phase Goal
`amy_lm.py` is completely rewritten. `AmyMossLM(PreTrainedModel, GenerationMixin)` composes `self.moss = MossAudioModel(...)`, supports constructor injection for pre-loaded backbones, and includes `prepare_base_checkpoint()` for HuggingFace Hub bootstrap. `AmyMossLMConfig` no longer extends `MossAudioConfig` — it wraps it.

## Files to Touch

| File | Action | Responsibility |
|------|--------|---------------|
| `src/models/amy_lm.py` | **Rewrite** | `AmyMossLMConfig(PretrainedConfig)` + `AmyMossLM(PreTrainedModel, GenerationMixin)` |
| `src/models/__init__.py` | **Edit** | Update exports from `AmyLM` to `AmyMossLM` |

## Pre-requisite Verification

- [ ] Phase 1 complete — `moss_audio_model.py` exists and imports cleanly
- [ ] `uv run python -c "from src.models.moss_audio_model import MossAudioConfig, MossAudioModel; print('OK')"`
- [ ] Confirm liger-kernel available: `uv run python -c "from liger_kernel.transformers.monkey_patch import apply_liger_kernel_to_qwen3; print('OK')"`

---

## Tasks

### Task 1: Rewrite `AmyMossLMConfig` (not extending MossAudioConfig)

**Files:**
- Rewrite: `src/models/amy_lm.py` (config class only first)

- [ ] **Step 1: Write failing import test for new config name**

Write a quick smoke test:
```bash
uv run python -c "from src.models.amy_lm import AmyMossLMConfig; print(type(AmyMossLMConfig))"
```
Expected: `ImportError` (class doesn't exist yet)

- [ ] **Step 2: Implement `AmyMossLMConfig`**

Replace the entire config class. `AmyMossLMConfig` extends `PretrainedConfig` directly (NOT `MossAudioConfig`). It stores a nested `MossAudioConfig` as `self.moss_config`.

```python
class AmyMossLMConfig(PretrainedConfig):
    """Configuration for AmyMossLM — composition-based Speech LM with FACodec enrichment.

    Wraps MossAudioConfig as self.moss_config rather than extending it.
    Exposes FACodec stream fields directly for serialization.
    """
    model_type = "amy_moss_lm"

    def __init__(
        self,
        moss_config: dict | MossAudioConfig | None = None,
        # FACodec stream config
        prosody_vocab_size: int = 1024,
        prosody_init_strategy: str = "random",
        prosody_init_std: float = 0.02,
        prosody_input_rate: float = 80.0,
        prosody_output_rate: float = 12.5,
        timbre_dim: int = 256,
        hidden_dim: int = 2560,
        # Freeze control
        freeze_audio_encoder: bool = True,
        freeze_audio_adapter: bool = True,
        freeze_llm: bool = True,
        **kwargs,
    ):
        self.prosody_vocab_size = prosody_vocab_size
        self.prosody_init_strategy = prosody_init_strategy
        self.prosody_init_std = prosody_init_std
        self.prosody_input_rate = prosody_input_rate
        self.prosody_output_rate = prosody_output_rate
        self.timbre_dim = timbre_dim
        self.hidden_dim = hidden_dim
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_audio_adapter = freeze_audio_adapter
        self.freeze_llm = freeze_llm

        # Propagate key language model fields to parent PretrainedConfig
        if moss_config is not None:
            if isinstance(moss_config, dict):
                moss_config = MossAudioConfig(**moss_config)
            self.moss_config = moss_config
            lang = moss_config.language_config
            kwargs.setdefault("vocab_size", lang.vocab_size)
            kwargs.setdefault("hidden_size", lang.hidden_size)
            kwargs.setdefault("num_hidden_layers", lang.num_hidden_layers)
        else:
            self.moss_config = MossAudioConfig()
            lang = self.moss_config.language_config
            kwargs.setdefault("vocab_size", lang.vocab_size)
            kwargs.setdefault("hidden_size", lang.hidden_size)

        super().__init__(tie_word_embeddings=False, **kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["moss_config"] = self.moss_config.to_dict()
        output["prosody_vocab_size"] = self.prosody_vocab_size
        output["prosody_init_strategy"] = self.prosody_init_strategy
        output["prosody_init_std"] = self.prosody_init_std
        output["prosody_input_rate"] = self.prosody_input_rate
        output["prosody_output_rate"] = self.prosody_output_rate
        output["timbre_dim"] = self.timbre_dim
        output["hidden_dim"] = self.hidden_dim
        output["freeze_audio_encoder"] = self.freeze_audio_encoder
        output["freeze_audio_adapter"] = self.freeze_audio_adapter
        output["freeze_llm"] = self.freeze_llm
        return output

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """Load config from a MOSS-Audio checkpoint directory.
        
        If the checkpoint has model_type "moss_audio", wraps it as moss_config.
        """
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)
```

Key design points:
- `model_type = "amy_moss_lm"` for AutoModel registration
- `moss_config` is a nested `MossAudioConfig` instance (NOT inherited)
- Propagates `vocab_size`, `hidden_size` from `moss_config.language_config` to parent `PretrainedConfig` (required by transformers for model construction)
- `to_dict()` serializes `moss_config` as a dict
- Handles `from_pretrained` loading of MOSS-Audio checkpoints (model_type "moss_audio") by wrapping the loaded config as `moss_config`

- [ ] **Step 3: Verify config instantiation**

```bash
uv run python -c "
from src.models.amy_lm import AmyMossLMConfig
# Default
c = AmyMossLMConfig()
assert c.model_type == 'amy_moss_lm'
assert hasattr(c, 'moss_config')
assert c.prosody_vocab_size == 1024
assert c.moss_config.language_config._attn_implementation == 'flash_attention_2'
# With custom moss_config dict
c2 = AmyMossLMConfig(moss_config={'language_config': {'vocab_size': 1000}})
assert c2.moss_config.language_config.vocab_size == 1000
assert c2.vocab_size == 1000
# to_dict roundtrip
d = c.to_dict()
assert d['model_type'] == 'amy_moss_lm'
assert 'moss_config' in d
c3 = AmyMossLMConfig.from_dict(d)
assert c3.prosody_vocab_size == c.prosody_vocab_size
print('config OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/models/amy_lm.py
git commit -m "feat(phase-2): AmyMossLMConfig with nested MossAudioConfig (composition)"
```

### Task 2: Implement `AmyMossLM` composition architecture

**Files:**
- Edit: `src/models/amy_lm.py` (add AmyMossLM class)

- [ ] **Step 1: Implement `AmyMossLM.__init__` with constructor injection**

```python
class AmyMossLM(PreTrainedModel, GenerationMixin):
    """Amy Moss LM — composition-based Speech LM with FACodec enrichment.
    
    Composes (HAS-A) a MossAudioModel as self.moss rather than inheriting it.
    Supports constructor injection for pre-loaded (e.g., 4-bit quantized) backbones.
    
    Trainable: ProsodyEmbedding, TimbreProjection, TemporalPool, ResidualFusion.
    Frozen (default): moss.audio_encoder, moss.audio_adapter, moss.language_model.
    """
    config_class = AmyMossLMConfig
    base_model_prefix = "moss"
    _no_split_modules = ["Qwen3DecoderLayer", "WhisperEncoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    supports_gradient_checkpointing = True
    _tied_weights_keys: List[str] = []

    def __init__(self, config: AmyMossLMConfig, moss: MossAudioModel | None = None):
        super().__init__(config)
        
        if moss is not None:
            # Pre-loaded backbone (e.g., 4-bit quantized)
            self.moss = moss
        else:
            self.moss = MossAudioModel(config.moss_config)
        
        self._add_facodec_modules(config)
        self._apply_freeze(config)
        self.post_init()
```

- [ ] **Step 2: Add FACodec modules (same as old `_add_facodec_modules`)**

```python
    def _add_facodec_modules(self, config: AmyMossLMConfig) -> None:
        self.prosody_embedding = ProsodyEmbedding(
            vocab_size=config.prosody_vocab_size,
            embed_dim=config.hidden_dim,
            init_strategy=config.prosody_init_strategy,
            init_std=config.prosody_init_std,
        )
        self.timbre_projection = TimbreProjection(
            timbre_dim=config.timbre_dim,
            output_dim=config.hidden_dim,
        )
        self.temporal_pool = TemporalPool(
            input_rate=config.prosody_input_rate,
            output_rate=config.prosody_output_rate,
        )
        self.residual_fusion = ResidualFusion(hidden_dim=config.hidden_dim)
```

- [ ] **Step 3: Add freeze logic (freeze self.moss.* attributes)**

```python
    def _apply_freeze(self, config: AmyMossLMConfig) -> None:
        if config.freeze_audio_encoder:
            for p in self.moss.audio_encoder.parameters():
                p.requires_grad = False
        if config.freeze_audio_adapter:
            for p in self.moss.audio_adapter.parameters():
                p.requires_grad = False
        if config.freeze_llm:
            for p in self.moss.language_model.parameters():
                p.requires_grad = False
```

- [ ] **Step 4: Add delegation methods for input/output embeddings**

```python
    def get_input_embeddings(self):
        return self.moss.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.moss.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.moss.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.moss.set_output_embeddings(new_embeddings)
```

- [ ] **Step 5: Add `_enrich_audio_embeds()` (same as current, uses self.* not self.moss.*)**

Same logic as current `AmyLM._enrich_audio_embeds`. The FACodec modules live on `self` (not `self.moss`), so this method is unchanged from the current inheritance version.

- [ ] **Step 6: Implement `forward()` with composition delegation**

The forward() method ports MossAudioModel.forward() but delegates heavy calls to `self.moss.*` and injects FACodec enrichment. Key delegation points:

| Inheritance reference (old) | Composition reference (new) |
|---|---|
| `self.config` | `self.config` (AmyMossLMConfig, NOT MossAudioConfig) |
| `self.config.ignore_index` | `self.config.moss_config.ignore_index` |
| `self.config.language_config.vocab_size` | `self.config.moss_config.language_config.vocab_size` |
| `self.get_audio_features()` | `self.moss.get_audio_features()` |
| `self.audio_adapter()` | `self.moss.audio_adapter()` |
| `self.language_model()` | `self.moss.language_model()` |
| `self.lm_head()` | `self.moss.lm_head()` |
| `self.deepstack_audio_merger_list` | `self.moss.deepstack_audio_merger_list` |
| `self._register_llm_deepstack_hooks()` | `self.moss._register_llm_deepstack_hooks()` |

The FACodec enrichment is injected between `self.moss.audio_adapter()` and `masked_scatter_()`, exactly as in the current `AmyLM.forward()`.

- [ ] **Step 7: Add `prepare_inputs_for_generation()`**

Port from `MossAudioModel.prepare_inputs_for_generation()`. No FACodec enrichment during generation (prosody/timbre not available at generation time).

- [ ] **Step 8: Add `prepare_base_checkpoint()` classmethod**

```python
    @classmethod
    def prepare_base_checkpoint(cls, save_path: str, moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking"):
        """Bootstrap a full AmyMossLM checkpoint from MOSS-Audio weights.
        
        Loads MossAudioModel weights, wraps in AmyMossLM composition, saves
        the full checkpoint (including randomly-initialized FACodec modules)
        to the specified path. Use this once to create the base checkpoint.
        """
        moss = MossAudioModel.from_pretrained(
            moss_model_id,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        config = AmyMossLMConfig(moss_config=moss.config)
        model = cls(config, moss=moss)
        model.save_pretrained(save_path)
        config.save_pretrained(save_path)
        return model
```

- [ ] **Step 9: Register for AutoModel**

At module level (after class definition):
```python
AmyMossLMConfig.register_for_auto_class()
AmyMossLM.register_for_auto_class("AutoModelForCausalLM")
```

- [ ] **Step 10: Verify basic construction**

```bash
uv run python -c "
from src.models.amy_lm import AmyMossLMConfig, AmyMossLM
config = AmyMossLMConfig(moss_config={'language_config': {'vocab_size': 1000, 'hidden_size': 2560}})
model = AmyMossLM(config)
assert hasattr(model, 'moss')
assert hasattr(model, 'prosody_embedding')
assert model.config.model_type == 'amy_moss_lm'
print('model construction OK')
"
```

- [ ] **Step 11: Commit**

```bash
git add src/models/amy_lm.py
git commit -m "feat(phase-2): AmyMossLM composition architecture with bootstrap"
```

### Task 3: Update `__init__.py` exports

**Files:**
- Edit: `src/models/__init__.py`

- [ ] **Step 1: Change exports from `AmyLM` to `AmyMossLM`**

Replace:
```python
from .amy_lm import AmyLMConfig, AmyLM
```
With:
```python
from .amy_lm import AmyMossLMConfig, AmyMossLM
```

And in `__all__`:
```python
    "AmyMossLMConfig",
    "AmyMossLM",
```

- [ ] **Step 2: Remove the lazy `__getattr__` for classifiers (eager imports only)**

The `__getattr__` function was only needed because classifier imports were broken during Phase 1. Now both classifiers import cleanly, so remove the lazy `__getattr__` and import them eagerly instead:

```python
from .amy_classifier import AmyForProsodyClassification
from .baseline_classifier import BaselineClassifier
```

- [ ] **Step 3: Verify exports**

```bash
uv run python -c "
from src.models import AmyMossLMConfig, AmyMossLM
print('new exports OK')
"
```

- [ ] **Step 4: Commit**

```bash
git add src/models/__init__.py
git commit -m "feat(phase-2): update __init__.py exports to AmyMossLM naming"
```

---

## Phase Completion Criteria
- [ ] `AmyMossLMConfig(PretrainedConfig)` with `model_type = "amy_moss_lm"` compiles and serializes
- [ ] `AmyMossLM(PreTrainedModel, GenerationMixin)` composes `self.moss`
- [ ] Constructor injection works: `AmyMossLM(config, moss=pre_loaded)` 
- [ ] `forward()` replicates MossAudioModel orchestration with FACodec enrichment hook
- [ ] `prepare_inputs_for_generation()` supports generation without FACodec
- [ ] `prepare_base_checkpoint()` bootstraps and saves a full checkpoint
- [ ] `AutoModelForCausalLM` registration works
- [ ] `__init__.py` exports `AmyMossLMConfig`/`AmyMossLM`
- [ ] Old `AmyLM`/`AmyLMConfig` classes deleted from `amy_lm.py`

## Handoff Notes
Phase 3 needs the fully working `AmyMossLM` to update `train_amy_dpo.py`'s `init_model()` and to write tests. The `moss.*` key prefix from composition enables LoRA scoping via `^moss\..*`.
