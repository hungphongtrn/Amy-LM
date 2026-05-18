"""AmyLM — HF-compatible Speech Language Model with FACodec enrichment.

Inherits MossAudioModel, adding ProsodyEmbedding, TimbreProjection,
TemporalPool, and ResidualFusion as permanent architecture modules.
Overrides forward() to enrich audio embeddings with prosody/timbre
before the <audio> placeholder-replacement step.

Issue #19, Group A3.
"""

from __future__ import annotations

import os
import sys
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast

# Vendor path setup (same pattern as src/models/moss_audio.py)
_VENDOR_MOSS_AUDIO_SRC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "MOSS-Audio", "src")
)
if os.path.isdir(_VENDOR_MOSS_AUDIO_SRC_PATH) and _VENDOR_MOSS_AUDIO_SRC_PATH not in sys.path:
    sys.path.insert(0, _VENDOR_MOSS_AUDIO_SRC_PATH)

from configuration_moss_audio import MossAudioConfig
from modeling_moss_audio import MossAudioModel

from .embedding import ProsodyEmbedding, TimbreProjection
from .pooling import TemporalPool
from .fusion import ResidualFusion


class AmyLMConfig(MossAudioConfig):
    """Configuration for AmyLM — extends MossAudioConfig with FACodec stream settings.

    Additional fields beyond MossAudioConfig:
        prosody_vocab_size: Codebook vocabulary size (default 1024)
        prosody_init_strategy: 'random' or 'warm_start' (default 'random')
        prosody_init_std: Std for random init (default 0.02)
        prosody_input_rate: FACodec frame rate in Hz (default 80.0)
        prosody_output_rate: Output frame rate in Hz (default 12.5)
        timbre_dim: Timbre vector dimension (default 256)
        hidden_dim: MOSS-Audio hidden dimension (default 2560)
        freeze_audio_encoder: Freeze audio encoder (default True)
        freeze_audio_adapter: Freeze audio adapter (default True)
        freeze_llm: Freeze Qwen3 language model (default True)
    """

    model_type = "amy_lm"

    def __init__(
        self,
        audio_config=None,
        language_config=None,
        adapter_hidden_size=8192,
        ignore_index=-100,
        deepstack_num_inject_layers: Optional[int] = None,
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
        super().__init__(
            audio_config=audio_config,
            language_config=language_config,
            adapter_hidden_size=adapter_hidden_size,
            ignore_index=ignore_index,
            deepstack_num_inject_layers=deepstack_num_inject_layers,
            **kwargs,
        )
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

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
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


class AmyLM(MossAudioModel):
    """Amy LM — HF-compatible Speech LM with prosody/timbre enrichment.

    Inherits MossAudioModel's full architecture (Whisper encoder + GatedMLP
    adapter + Qwen3 LM) and adds FACodec prosody/timbre modules with ResidualFusion.
    The forward() method enriches audio embeddings between audio_adapter()
    and masked_scatter_(), enabling gradient flow through the FACodec pathway
    during DPO training.

    Trainable: ProsodyEmbedding (incl. projector), TimbreProjection, λ gates.
    Frozen (default): audio_encoder, audio_adapter, language_model.
    """

    config_class = AmyLMConfig

    def __init__(self, config: AmyLMConfig):
        super().__init__(config)

        # FACodec enrichment modules (prosody + timbre only, per Stream Activation Config)
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

        self._apply_freeze(config)
        self.post_init()

    def _apply_freeze(self, config: AmyLMConfig) -> None:
        """Apply freeze configuration to backbone modules."""
        if config.freeze_audio_encoder:
            for p in self.audio_encoder.parameters():
                p.requires_grad = False
        if config.freeze_audio_adapter:
            for p in self.audio_adapter.parameters():
                p.requires_grad = False
        if config.freeze_llm:
            for p in self.language_model.parameters():
                p.requires_grad = False

    def _enrich_audio_embeds(
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

        if prosody_indices is not None:
            p_emb = self.prosody_embedding(prosody_indices)      # [B, T80, D]
            p_emb = self.temporal_pool(p_emb)                     # [B, T12, D]
            # Align to audio_embeds length
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
            streams["prosody"] = p_emb.to(device=audio_embeds.device, dtype=audio_embeds.dtype)

        if timbre_vector is not None:
            t_emb = self.timbre_projection(timbre_vector)           # [B, D]
            t_emb = t_emb.unsqueeze(1).expand(-1, audio_embeds.shape[1], -1)  # [B, T, D]
            streams["timbre"] = t_emb.to(device=audio_embeds.device, dtype=audio_embeds.dtype)

        return self.residual_fusion(
            audio_embeds,
            prosody=streams.get("prosody"),
            content=None,
            acoustic=None,
            timbre=streams.get("timbre"),
        )

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
        # Extract FACodec inputs from kwargs
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
            inputs_embeds = self.get_input_embeddings()(input_ids)

        hook_handles: list = []
        if audio_data is not None:
            if audio_input_mask is None:
                raise ValueError("audio_input_mask is required when audio_data is provided.")

            audio_embeds, deepstack = self.get_audio_features(audio_data, audio_data_seqlens)
            audio_embeds = self.audio_adapter(audio_embeds)

            # Enrich audio embeddings with FACodec prosody/timbre BEFORE scattering
            audio_embeds = self._enrich_audio_embeds(
                audio_embeds,
                prosody_indices=prosody_indices,
                timbre_vector=timbre_vector,
            )

            audio_token_count = int(audio_input_mask.to(torch.int32).sum().item())
            if audio_token_count != int(audio_embeds.shape[1]):
                raise ValueError(
                    f"Audio token count mismatch: audio_input_mask has {audio_token_count} audio tokens, "
                    f"but audio_embeds has length {int(audio_embeds.shape[1])}."
                )

            mask_expanded = audio_input_mask.unsqueeze(-1).expand_as(inputs_embeds)
            inputs_embeds = inputs_embeds.clone()
            inputs_embeds.masked_scatter_(mask_expanded, audio_embeds)

            if deepstack is not None and len(self.deepstack_audio_merger_list) > 0:
                deepstack_audio_embeds = []
                for i, x in enumerate(deepstack[: len(self.deepstack_audio_merger_list)]):
                    ds = self.deepstack_audio_merger_list[i](x)
                    if int(ds.shape[1]) != audio_token_count:
                        raise ValueError(
                            f"DeepStack audio seq_len mismatch at index {i}: "
                            f"expected {audio_token_count}, got {int(ds.shape[1])}."
                        )
                    deepstack_audio_embeds.append(ds)

                try:
                    hook_handles = self._register_llm_deepstack_hooks(
                        audio_input_mask, deepstack_audio_embeds
                    )
                except Exception:
                    for h in hook_handles:
                        h.remove()
                    raise

        try:
            outputs = self.language_model(
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

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.config.ignore_index)
            shift_logits = shift_logits.view(-1, self.config.language_config.vocab_size)
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
