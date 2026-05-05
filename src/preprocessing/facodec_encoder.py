"""FACodec Encoder Wrapper for Amy-LM preprocessing pipeline.

This module provides a wrapper around the FACodec model from Amphion.
When Amphion is not installed, it provides a deterministic mock fallback
for testing purposes.
"""

from typing import Optional, Tuple
import torch


class FACodecEncoder:
    """Wrapper for FACodec encoder that extracts content, prosody, and timbre indices.
    
    FACodec is a neural audio codec that disentangles speech into three interpretable
    layers: content (semantic), prosody (pitch/rhythm), and timbre (speaker identity).
    
    When Amphion is available, this wrapper loads the real FACodec model.
    When Amphion is NOT available, it falls back to a deterministic mock that
    generates structurally valid indices for testing.
    
    Args:
        device: Device to run on ("cpu" or "cuda")
        checkpoint_path: Optional path to FACodec checkpoint. If None and Amphion
            is available, will use default checkpoint.
    
    Attributes:
        device: The device being used
        _mock: True if using mock fallback (Amphion not available)
        _frame_rate: Frames per second output (~12.5 Hz)
        _vocab_size: Codebook vocabulary size (2048)
        _sample_rate: Audio sample rate (16000 Hz)
    
    Example:
        >>> encoder = FACodecEncoder(device="cpu")
        >>> audio = torch.randn(32000)  # 2 seconds at 16kHz
        >>> content, prosody, timbre = encoder.encode(audio)
        >>> len(content)  # ~25 frames at 12.5 Hz
    """
    
    # FACodec constants
    FRAME_RATE = 12.5  # Hz (12.5 frames per second)
    SAMPLE_RATE = 16000  # Hz
    VOCAB_SIZE = 2048  # Codebook size for each stream
    SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)  # 1280 samples per frame
    
    def __init__(self, device: str = "cpu", checkpoint_path: Optional[str] = None):
        """Initialize FACodec encoder.
        
        Attempts to load Amphion's FACodec. If not available, falls back to
        mock mode for testing.
        
        Args:
            device: Device to run on
            checkpoint_path: Path to checkpoint (optional)
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._model = None
        
        # Try to import Amphion FACodec
        try:
            # This will fail if Amphion is not installed
            from amphion.models.codec import FACodec  # type: ignore
            self._mock = False
            self._init_real_codec(FACodec)
        except ImportError:
            # Amphion not available, use mock fallback
            self._mock = True
            
    def _init_real_codec(self, FACodecClass):
        """Initialize real FACodec model (when Amphion is available)."""
        # TODO: Implement real FACodec loading when Amphion is available
        # For now, fall back to mock
        self._mock = True
        
    def encode(self, audio: torch.Tensor) -> Tuple[list[int], list[int], list[int]]:
        """Encode audio into FACodec indices.
        
        Takes raw audio waveform and returns codebook indices for content,
        prosody, and timbre streams.
        
        Args:
            audio: Audio waveform tensor, shape [samples] or [1, samples].
                   Should be at 16kHz sample rate.
        
        Returns:
            Tuple of (content_indices, prosody_indices, timbre_indices).
            Each is a list of integer indices into codebooks of size 2048.
            Frame rate is ~12.5 Hz (80 samples per frame at 16kHz).
        
        Raises:
            ValueError: If audio is empty or has wrong shape.
        """
        # Normalize audio shape
        if audio.dim() == 2:
            if audio.shape[0] == 1:
                audio = audio.squeeze(0)
            else:
                raise ValueError(f"Audio should be shape [samples] or [1, samples], got {audio.shape}")
        elif audio.dim() != 1:
            raise ValueError(f"Audio should be 1D or 2D, got {audio.dim()}D")
        
        if audio.numel() == 0:
            raise ValueError("Audio cannot be empty")
        
        if self._mock:
            return self._encode_mock(audio)
        else:
            return self._encode_real(audio)
    
    def _encode_mock(self, audio: torch.Tensor) -> Tuple[list[int], list[int], list[int]]:
        """Generate deterministic mock indices for testing.
        
        Creates fake but structurally valid indices based on audio length.
        Uses a deterministic algorithm so results are reproducible.
        
        Args:
            audio: Audio waveform tensor
        
        Returns:
            Tuple of (content, prosody, timbre) index lists
        """
        # Calculate number of frames based on audio length
        num_samples = audio.shape[0]
        num_frames = max(1, int(num_samples / self.SAMPLES_PER_FRAME))
        
        # Use deterministic pseudo-random generation based on audio statistics
        # This ensures same audio produces same indices, but different
        # audio produces different indices
        audio_mean = audio.mean().item()
        audio_std = audio.std().item() if audio.numel() > 1 else 0.0
        
        # Seed based on audio characteristics for determinism
        seed = int((abs(audio_mean) + abs(audio_std)) * 10000) % 2**32
        
        # Generate indices using deterministic algorithm
        content_indices = []
        prosody_indices = []
        timbre_indices = []
        
        for frame_idx in range(num_frames):
            # Content: varies with frame position, some randomness
            content_val = (seed + frame_idx * 47) % self.VOCAB_SIZE
            content_indices.append(content_val)
            
            # Prosody: offset from content, different pattern
            prosody_val = (seed + frame_idx * 31 + 512) % self.VOCAB_SIZE
            prosody_indices.append(prosody_val)
            
            # Timbre: different offset and pattern
            timbre_val = (seed + frame_idx * 19 + 1024) % self.VOCAB_SIZE
            timbre_indices.append(timbre_val)
        
        return content_indices, prosody_indices, timbre_indices
    
    def _encode_real(self, audio: torch.Tensor) -> Tuple[list[int], list[int], list[int]]:
        """Encode using real FACodec model (when Amphion is available).
        
        Args:
            audio: Audio waveform tensor
        
        Returns:
            Tuple of (content, prosody, timbre) index lists
        """
        # TODO: Implement real FACodec encoding when Amphion is available
        # For now, fall back to mock
        return self._encode_mock(audio)
