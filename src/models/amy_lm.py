"""AmyMossLM — HF-compatible Speech Language Model with FACodec enrichment.

Composes (HAS-A) MossAudioModel as self.moss rather than inheriting it.
Supports constructor injection for pre-loaded (e.g., 4-bit quantized) backbones.

Trainable: ProsodyEmbedding, TimbreProjection, TemporalPool, ResidualFusion.
Frozen (default): moss.audio_encoder, moss.audio_adapter, moss.language_model.

Issue #26.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.utils import GenerationMixin

from .moss_audio_model import MossAudioConfig, MossAudioModel

from .embedding import ProsodyEmbedding, TimbreProjection
from .pooling import TemporalPool
from .fusion import ResidualFusion

_VENDOR_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src")
)
if os.path.isdir(_VENDOR_SRC) and _VENDOR_SRC not in sys.path:
    sys.path.insert(0, _VENDOR_SRC)

from processing_moss_audio import MossAudioProcessor  # noqa: E402


class AmyMossLMConfig(PretrainedConfig):
    """Configuration for AmyMossLM — composition-based Speech LM with FACodec enrichment.

    Wraps MossAudioConfig as self.moss_config rather than extending it.
    Exposes FACodec stream fields directly for serialization.
    """

    model_type = "amy_moss_lm"

    def __init__(
        self,
        moss_config: dict | MossAudioConfig | None = None,
        prosody_vocab_size: int = 1024,
        prosody_init_strategy: str = "random",
        prosody_init_std: float = 0.02,
        prosody_input_rate: float = 80.0,
        prosody_output_rate: float = 12.5,
        prosody_warm_start_vectors_path: str | None = None,
        timbre_dim: int = 256,
        hidden_dim: int = 2560,
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
        self.prosody_warm_start_vectors_path = prosody_warm_start_vectors_path
        self.timbre_dim = timbre_dim
        self.hidden_dim = hidden_dim
        self.freeze_audio_encoder = freeze_audio_encoder
        self.freeze_audio_adapter = freeze_audio_adapter
        self.freeze_llm = freeze_llm

        if moss_config is not None:
            if isinstance(moss_config, dict):
                moss_config = MossAudioConfig(**moss_config)
            self.moss_config = moss_config
        else:
            self.moss_config = MossAudioConfig()

        lang = self.moss_config.language_config
        kwargs.setdefault("vocab_size", lang.vocab_size)
        kwargs.setdefault("hidden_size", lang.hidden_size)
        kwargs.setdefault("num_hidden_layers", lang.num_hidden_layers)

        kwargs.setdefault("tie_word_embeddings", False)
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        output["moss_config"] = self.moss_config.to_dict()
        output["prosody_vocab_size"] = self.prosody_vocab_size
        output["prosody_init_strategy"] = self.prosody_init_strategy
        output["prosody_init_std"] = self.prosody_init_std
        output["prosody_input_rate"] = self.prosody_input_rate
        output["prosody_output_rate"] = self.prosody_output_rate
        output["prosody_warm_start_vectors_path"] = self.prosody_warm_start_vectors_path
        output["timbre_dim"] = self.timbre_dim
        output["hidden_dim"] = self.hidden_dim
        output["freeze_audio_encoder"] = self.freeze_audio_encoder
        output["freeze_audio_adapter"] = self.freeze_audio_adapter
        output["freeze_llm"] = self.freeze_llm
        return output


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
            self.moss = moss
        else:
            self.moss = MossAudioModel(config.moss_config)

        self._processor = None

        self._add_facodec_modules(config)
        self._apply_freeze(config)
        self.post_init()

        if moss is not None:
            moss_dtype = next(moss.parameters()).dtype
            for module in (
                self.prosody_embedding,
                self.timbre_projection,
                self.residual_fusion,
            ):
                module.to(dtype=moss_dtype)

    def _init_weights(self, module):
        """No-op: prevent post_init() from re-initializing pre-trained moss backbone weights.

        All submodules are already initialized — moss backbone via
        from_pretrained and FACodec modules via their constructors.
        """

    def _add_facodec_modules(self, config: AmyMossLMConfig) -> None:
        from .codebook_utils import load_prosody_codebook_vectors

        warm_start_vectors = None
        init_strategy = config.prosody_init_strategy
        if config.prosody_warm_start_vectors_path is not None:
            warm_start_vectors = load_prosody_codebook_vectors(config.prosody_warm_start_vectors_path)
            init_strategy = "warm_start"
        self.prosody_embedding = ProsodyEmbedding(
            vocab_size=config.prosody_vocab_size,
            embed_dim=config.hidden_dim,
            init_strategy=init_strategy,
            init_std=config.prosody_init_std,
            warm_start_vectors=warm_start_vectors,
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

    def get_input_embeddings(self):
        return self.moss.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.moss.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.moss.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.moss.set_output_embeddings(new_embeddings)

    def enrich_audio_embeds(
        self,
        audio_embeds: torch.Tensor,
        prosody_indices: Optional[torch.Tensor] = None,
        timbre_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Enrich audio embeddings with FACodec prosody and timbre streams.

        Args:
            audio_embeds: Audio embeddings from audio_adapter [B, T, D].
            prosody_indices: Optional prosody codebook indices [B, 1, T80] int64.
            timbre_vector: Optional timbre vector [B, 256] float32.

        Returns:
            Enriched audio embeddings [B, T, D].
        """
        if prosody_indices is None and timbre_vector is None:
            return audio_embeds

        streams: dict[str, torch.Tensor] = {}
        fusion_dtype = self.residual_fusion.norm.weight.dtype
        semantic = audio_embeds.to(dtype=fusion_dtype)

        if prosody_indices is not None:
            p_emb = self.prosody_embedding(prosody_indices)
            p_emb = self.temporal_pool(p_emb)
            if p_emb.shape[1] < audio_embeds.shape[1]:
                pad = torch.zeros(
                    p_emb.shape[0],
                    audio_embeds.shape[1] - p_emb.shape[1],
                    p_emb.shape[2],
                    device=p_emb.device,
                    dtype=p_emb.dtype,
                )
                p_emb = torch.cat([p_emb, pad], dim=1)
            elif p_emb.shape[1] > audio_embeds.shape[1]:
                p_emb = p_emb[:, : audio_embeds.shape[1], :]
            streams["prosody"] = p_emb.to(device=audio_embeds.device, dtype=fusion_dtype)

        if timbre_vector is not None:
            timbre_dtype = self.timbre_projection.linear.weight.dtype
            timbre_vector = timbre_vector.to(dtype=timbre_dtype)
            t_emb = self.timbre_projection(timbre_vector)
            t_emb = t_emb.unsqueeze(1).expand(-1, audio_embeds.shape[1], -1)
            streams["timbre"] = t_emb.to(device=audio_embeds.device, dtype=fusion_dtype)

        fused = self.residual_fusion(
            semantic,
            prosody=streams.get("prosody"),
            content=None,
            acoustic=None,
            timbre=streams.get("timbre"),
        )
        return fused.to(dtype=audio_embeds.dtype)

    def _get_processor(self):
        if self._processor is None:
            self._processor = MossAudioProcessor.from_pretrained(
                self.config.moss_config._name_or_path,
                trust_remote_code=True,
                enable_time_marker=True,
            )
        return self._processor

    def encode_enriched_audio_embeds(
        self,
        audio: torch.Tensor,
        prosody_indices: Optional[torch.Tensor] = None,
        timbre_vector: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode raw waveform to enriched audio embeddings.

        Mel extraction + audio encoder + audio adapter + FACodec enrichment.
        Frozen backbone. Used by classifier probing without token/text paths.

        Args:
            audio: Raw waveform [B, T_audio] at 16 kHz.
            prosody_indices: Optional prosody VQ indices [B, 1, T80].
            timbre_vector: Optional timbre vector [B, 256].

        Returns:
            (embeds, audio_out_lens) tuple:
                embeds: Enriched embeddings [B, T_moss_max, 2560] at ~12.5 Hz, padded.
                audio_out_lens: Valid frame count per sample [B], int64.
        """
        device = next(self.moss.parameters()).device
        dtype = next(self.moss.parameters()).dtype
        audio = audio.to(device)
        if audio.dim() != 2:
            raise ValueError(f"Expected audio shape [B, T_audio], got {tuple(audio.shape)}")
        if audio.shape[0] == 0:
            return (
                torch.empty(0, 0, self.config.hidden_dim, device=device, dtype=dtype),
                torch.empty(0, device=device, dtype=torch.long),
            )

        processor = self._get_processor()
        B = audio.shape[0]

        samples: list[torch.Tensor] = []
        out_lens: list[int] = []

        for i in range(B):
            mel = processor._extract_mel(audio[i].detach().cpu())
            audio_data = mel.unsqueeze(0).to(dtype=dtype).to(device)
            audio_data_seqlens = torch.tensor(
                [mel.shape[-1]], dtype=torch.long, device=device
            )

            with torch.no_grad():
                audio_embeds, _ = self.moss.get_audio_features(audio_data, audio_data_seqlens)
                audio_embeds = self.moss.audio_adapter(audio_embeds)

            embedded = self.enrich_audio_embeds(
                audio_embeds,
                prosody_indices=prosody_indices[i : i + 1] if prosody_indices is not None else None,
                timbre_vector=timbre_vector[i : i + 1] if timbre_vector is not None else None,
            )
            samples.append(embedded[0])
            out_lens.append(int(embedded.shape[1]))

        out_lens_t = torch.tensor(out_lens, device=device, dtype=torch.long)
        T_max = int(out_lens_t.max().item())
        D = self.config.hidden_dim

        padded = torch.zeros(B, T_max, D, device=device, dtype=dtype)
        for i, s in enumerate(samples):
            padded[i, : s.shape[0], :] = s

        return padded, out_lens_t

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        audio_data: Optional[torch.FloatTensor] = None,
        audio_data_seqlens: Optional[torch.Tensor] = None,
        audio_input_mask: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Forward pass with FACodec prosody/timbre enrichment.

        Accepts prosody_indices and timbre_vector via **kwargs (HF-compatible
        passthrough). Enriches audio embeddings after audio_adapter and before
        masked_scatter_. Without FACodec inputs, behaves identically to
        MossAudioModel.

        Args:
            input_ids: Token indices [B, S].
            attention_mask: Attention mask [B, S].
            position_ids: Position indices [B, S].
            past_key_values: Cached KV states.
            inputs_embeds: Pre-computed input embeddings.
            labels: Token labels for loss computation [B, S].
            use_cache: Whether to use KV cache.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dict.
            audio_data: Mel spectrogram features [B, 128, T_mel].
            audio_data_seqlens: Audio sequence lengths [B].
            audio_input_mask: Boolean mask marking audio token positions [B, S].
            cache_position: Cache position for generation.
            **kwargs: Additional kwargs including:
                prosody_indices: [B, 1, T80] int64
                timbre_vector: [B, 256] float32

        Returns:
            CausalLMOutputWithPast or tuple of (loss, logits, ...).
        """
        prosody_indices = kwargs.pop("prosody_indices", None)
        timbre_vector = kwargs.pop("timbre_vector", None)

        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if inputs_embeds is None:
            inputs_embeds = self.moss.get_input_embeddings()(input_ids)

        hook_handles: list = []
        _saved_gc_states: dict[int, bool] = {}
        _llm_layers = getattr(self.moss.language_model, "layers", None)
        if audio_data is not None:
            if audio_input_mask is None:
                raise ValueError("audio_input_mask is required when audio_data is provided.")

            audio_embeds, deepstack = self.moss.get_audio_features(
                audio_data.to(dtype=inputs_embeds.dtype), audio_data_seqlens
            )
            audio_embeds = self.moss.audio_adapter(audio_embeds)

            audio_embeds = self.enrich_audio_embeds(
                audio_embeds,
                prosody_indices=prosody_indices,
                timbre_vector=timbre_vector.to(dtype=inputs_embeds.dtype)
                if timbre_vector is not None
                else None,
            )

            audio_token_count = int(audio_input_mask.to(torch.int32).sum().item())
            if audio_token_count != int(audio_embeds.shape[1]):
                raise ValueError(
                    f"Audio token count mismatch: audio_input_mask has {audio_token_count} audio tokens, "
                    f"but audio_embeds has length {int(audio_embeds.shape[1])}."
                )

            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds.to(dtype=inputs_embeds.dtype))

            if deepstack is not None and len(self.moss.deepstack_audio_merger_list) > 0:
                deepstack_audio_embeds = []
                for i, x in enumerate(deepstack[: len(self.moss.deepstack_audio_merger_list)]):
                    ds = self.moss.deepstack_audio_merger_list[i](x)
                    if int(ds.shape[1]) != audio_token_count:
                        raise ValueError(
                            f"DeepStack audio seq_len mismatch at index {i}: "
                            f"expected {audio_token_count}, got {int(ds.shape[1])}."
                        )
                    deepstack_audio_embeds.append(ds)

                if _llm_layers is not None:
                    num_ds = len(deepstack_audio_embeds)
                    for i in range(min(num_ds, len(_llm_layers))):
                        layer = _llm_layers[i]
                        _saved_gc_states[i] = layer.gradient_checkpointing
                        layer.gradient_checkpointing = False

                try:
                    hook_handles = self.moss._register_llm_deepstack_hooks(
                        audio_input_mask, deepstack_audio_embeds
                    )
                except Exception:
                    for h in hook_handles:
                        h.remove()
                    raise

        try:
            outputs = self.moss.language_model(
                input_ids=None,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            for h in hook_handles:
                h.remove()
            if _llm_layers is not None:
                for i, saved in _saved_gc_states.items():
                    if i < len(_llm_layers):
                        _llm_layers[i].gradient_checkpointing = saved

        hidden_states = outputs[0]
        logits = self.moss.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.moss_config.ignore_index)
            shift_logits = shift_logits.view(-1, self.config.moss_config.language_config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids", None)
        if cache_position is not None and cache_position[0] > 0:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
            audio_data = None
            audio_input_mask = None
            audio_data_seqlens = None
        else:
            audio_data = kwargs.get("audio_data", None)
            audio_input_mask = kwargs.get("audio_input_mask", None)
            audio_data_seqlens = kwargs.get("audio_data_seqlens", None)

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "audio_data": audio_data,
                "audio_input_mask": audio_input_mask,
                "audio_data_seqlens": audio_data_seqlens,
            }
        )

        return model_inputs

    @torch.no_grad()
    def _warm_start_prosody(self, vectors: torch.Tensor):
        """Replace prosody_embedding with warm-started version (in-place).

        Used when warm-start vectors are provided as a tensor rather than a file path.
        """
        from .embedding import ProsodyEmbedding
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype
        new_embedding = ProsodyEmbedding(
            vocab_size=self.config.prosody_vocab_size,
            embed_dim=self.config.hidden_dim,
            init_strategy="warm_start",
            warm_start_vectors=vectors,
        ).to(device=device, dtype=dtype)
        self.prosody_embedding = new_embedding

    FACODEC_PREFIXES = (
        "prosody_embedding.",
        "timbre_projection.",
        "temporal_pool.",
        "residual_fusion.",
    )

    @property
    def moss_model_id(self) -> str:
        return self.config.moss_config._name_or_path

    def facodec_state_dict(self) -> dict[str, torch.Tensor]:
        """Return FACodec module weights for cross-task transfer (e.g. MUStARD → DPO)."""
        state = self.state_dict()
        return {k: v for k, v in state.items() if k.startswith(self.FACODEC_PREFIXES)}

    def load_facodec_state_dict(
        self, state_dict: dict[str, torch.Tensor], strict: bool = False
    ):
        """Load FACodec module weights from a classifier checkpoint.

        Strips `amy_moss.` prefix if present (from classifier-owned AmyMossLM checkpoints).
        """
        normalized = {}
        for k, v in state_dict.items():
            key = k.removeprefix("amy_moss.")
            if key.startswith(self.FACODEC_PREFIXES) and "moss." not in key:
                normalized[key] = v
        return self.load_state_dict(normalized, strict=strict)

    @classmethod
    def prepare_base_checkpoint(
        cls,
        save_path: str,
        moss_model_id: str = "OpenMOSS-Team/MOSS-Audio-4B-Thinking",
    ):
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


AmyMossLMConfig.register_for_auto_class()
AmyMossLM.register_for_auto_class("AutoModelForCausalLM")
