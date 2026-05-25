"""Amy-LM model modules."""
from .embedding import ProsodyEmbedding, TimbreProjection, AcousticEmbedding, ContentEmbedding
from .pooling import TemporalPool
from .fusion import ResidualFusion
from .moss_audio_model import MossAudioConfig, MossAudioModel
from .amy_lm import AmyLMConfig, AmyLM

__all__ = [
    "ProsodyEmbedding",
    "TimbreProjection",
    "AcousticEmbedding",
    "ContentEmbedding",
    "TemporalPool",
    "ResidualFusion",
    "MossAudioConfig",
    "MossAudioModel",
    "AmyForProsodyClassification",
    "BaselineClassifier",
    "AmyLMConfig",
    "AmyLM",
]


def __getattr__(name: str):
    if name == "AmyForProsodyClassification":
        from .amy_classifier import AmyForProsodyClassification

        return AmyForProsodyClassification
    if name == "BaselineClassifier":
        from .baseline_classifier import BaselineClassifier

        return BaselineClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
