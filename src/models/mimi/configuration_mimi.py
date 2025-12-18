from dataclasses import dataclass


@dataclass
class SEANETConfig:
    channels: int
    dimension: int
    causal: bool
    n_filters: int
    n_residual_layers: int
    activation: str
    compress: int
    dilation_base: int
    disable_norm_outer_blocks: int
    kernel_size: int
    residual_kernel_size: int
    last_kernel_size: int
    norm: str
    pad_mode: str
    ratios: list[int]
    true_skip: bool


@dataclass
class QuantizerConfig:
    dimension: int
    n_q: int
    bins: int
    input_dimension: int
    output_dimension: int


@dataclass
class TransformerConfig:
    d_model: int
    num_heads: int
    num_layers: int
    causal: bool
    layer_scale: float
    context: int
    conv_layout: bool
    max_period: int
    gating: str
    norm: str
    positional_embedding: str
    dim_feedforward: int
    input_dimension: int
    output_dimensions: list[int]


@dataclass
class MimiConfig:
    sample_rate: float
    channels: int
    frame_rate: float
    seanet: SEANETConfig
    quantizer: QuantizerConfig
    transformer: TransformerConfig


DEFAULT_SEANET_CONFIG = SEANETConfig(
    **{
        "channels": 1,
        "dimension": 512,
        "causal": True,
        "n_filters": 64,
        "n_residual_layers": 1,
        "activation": "ELU",
        "compress": 2,
        "dilation_base": 2,
        "disable_norm_outer_blocks": 0,
        "kernel_size": 7,
        "residual_kernel_size": 3,
        "last_kernel_size": 3,
        # We train using weight_norm but then the weights are pre-processed for inference so
        # that we can use a normal convolution.
        "norm": "none",
        "pad_mode": "constant",
        "ratios": [8, 6, 5, 4],
        "true_skip": True,
    }
)

DEFAULT_QUANTIZER_CONFIG = QuantizerConfig(
    **{
        "dimension": 256,
        "n_q": 32,
        "bins": 2048,
        "input_dimension": DEFAULT_SEANET_CONFIG.dimension,
        "output_dimension": DEFAULT_SEANET_CONFIG.dimension,
    }
)


DEFAULT_TRANSFORMER_CONFIG = TransformerConfig(
    **{
        "d_model": DEFAULT_SEANET_CONFIG.dimension,
        "num_heads": 8,
        "num_layers": 8,
        "causal": True,
        "layer_scale": 0.01,
        "context": 250,
        "conv_layout": True,
        "max_period": 10000,
        "gating": "none",
        "norm": "layer_norm",
        "positional_embedding": "rope",
        "dim_feedforward": 2048,
        "input_dimension": DEFAULT_SEANET_CONFIG.dimension,
        "output_dimensions": [DEFAULT_SEANET_CONFIG.dimension],
    }
)

DEFAULT_MIMI_CONFIG = MimiConfig(
    **{
        "sample_rate": 24000,
        "channels": 1,
        "frame_rate": 12.5,
        "seanet": DEFAULT_SEANET_CONFIG,
        "quantizer": DEFAULT_QUANTIZER_CONFIG,
        "transformer": DEFAULT_TRANSFORMER_CONFIG,
    }
)


def _merge_config(new_config: dict) -> MimiConfig:
    for k, v in new_config.items():
        if k == "seanet":
            DEFAULT_MIMI_CONFIG.seanet = SEANETConfig(**v)
        elif k == "quantizer":
            DEFAULT_MIMI_CONFIG.quantizer = QuantizerConfig(**v)
        elif k == "transformer":
            DEFAULT_MIMI_CONFIG.transformer = TransformerConfig(**v)
        elif k in ["sample_rate", "channels", "frame_rate"]:
            DEFAULT_MIMI_CONFIG.k = v
    return DEFAULT_MIMI_CONFIG
