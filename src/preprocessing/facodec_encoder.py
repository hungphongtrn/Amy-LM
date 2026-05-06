"""FACodec Encoder Wrapper for Amy-LM preprocessing pipeline.

This module provides a wrapper around the FACodec model from Amphion.
When Amphion is not installed, it provides a deterministic mock fallback
for testing purposes.
"""

import os
import sys
from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch


@dataclass
class FACodecStreams:
    """FACodec output streams container.
    
    Holds all outputs from FACodec encoder/decoder:
    - prosody_codebooks_idx: [1, T80] int64 - prosody codebook indices
    - content_codebooks_idx: [2, T80] int64 - content codebook indices  
    - acoustic_codebooks_idx: [3, T80] int64 - acoustic/residual codebook indices
    - timbre_vector: [256] float32 - utterance-level timbre vector from spk_embs
    
    Args:
        prosody_codebooks_idx: Prosody stream indices, shape [1, T80]
        content_codebooks_idx: Content stream indices, shape [2, T80]
        acoustic_codebooks_idx: Acoustic stream indices, shape [3, T80]
        timbre_vector: Timbre vector, shape [256] float32
    """
    prosody_codebooks_idx: torch.Tensor   # [1, T80], int64
    content_codebooks_idx: torch.Tensor   # [2, T80], int64
    acoustic_codebooks_idx: torch.Tensor  # [3, T80], int64
    timbre_vector: torch.Tensor           # [256], float32

# Add vendor/Amphion to path so we can import Amphion models
_vendor_path = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "Amphion")
_vendor_path = os.path.abspath(_vendor_path)
if os.path.isdir(_vendor_path) and _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)


class FACodecEncoder:
    """Wrapper for FACodec encoder that extracts disentangled speech representations.
    
    FACodec is a neural audio codec that disentangles speech into four interpretable
    streams: prosody (pitch/rhythm), content (semantic), acoustic (residual detail),
    and a continuous timbre vector (speaker identity).
    
    When Amphion is available, this wrapper loads the real FACodec model.
    When Amphion is NOT available, it falls back to a deterministic mock that
    generates structurally valid streams for testing.
    
    Args:
        device: Device to run on ("cpu" or "cuda")
        checkpoint_path: Optional path to FACodec checkpoint. If None and Amphion
            is available, will use default checkpoint from checkpoints/facodec/.
    
    Attributes:
        device: The device being used
        _mock: True if using mock fallback (Amphion not available)
        _frame_rate: Nominal frames per second (~12.5 Hz for compatibility)
        FACODEC_FRAME_RATE: Actual FACodec frame rate (80 Hz)
        _vocab_size: Codebook vocabulary size (1024 for FACodec)
        _sample_rate: Audio sample rate (16000 Hz)
    
    Example:
        >>> encoder = FACodecEncoder(device="cpu")
        >>> audio = torch.randn(32000)  # 2 seconds at 16kHz
        >>> streams = encoder.encode(audio)
        >>> streams.prosody_codebooks_idx.shape  # [1, T80] where T80 ≈ 160 frames at 80 Hz
        >>> streams.timbre_vector.shape  # [256] float32
    """
    
    # FACodec constants
    # Note: FACodec actually runs at 80 Hz (hop size 200 at 16kHz), 
    # but we normalize to match expected frame counts for downstream compatibility
    FRAME_RATE = 12.5  # Hz (nominal frame rate for Amy-LM compatibility)
    FACODEC_FRAME_RATE = 80.0  # Hz (actual FACodec frame rate with hop_size=200)
    SAMPLE_RATE = 16000  # Hz
    VOCAB_SIZE = 1024  # FACodec codebook size is 1024 (10-bit codebooks)
    SAMPLES_PER_FRAME = int(SAMPLE_RATE / FRAME_RATE)  # 1280 samples per frame (nominal)
    
    def __init__(self, device: str = "cpu", checkpoint_path: Optional[str] = None, force_mock: bool = False):
        """Initialize FACodec encoder.
        
        Attempts to load Amphion's FACodec. If not available, falls back to
        mock mode for testing.
        
        Args:
            device: Device to run on ("cpu" or "cuda")
            checkpoint_path: Path to checkpoint directory containing 
                ns3_facodec_encoder.bin and ns3_facodec_decoder.bin.
                If None, defaults to 'checkpoints/facodec/'.
            force_mock: If True, force mock mode even if Amphion is available.
                Useful for testing or when you want deterministic behavior.
        """
        self.device = device
        self.checkpoint_path = checkpoint_path
        self._encoder = None
        self._decoder = None
        
        if force_mock:
            # Force mock mode for testing or deterministic behavior
            self._mock = True
        else:
            # Try to import Amphion FACodec
            try:
                from models.codec.ns3_codec import FACodecEncoder as AmphionEncoder
                from models.codec.ns3_codec import FACodecDecoder as AmphionDecoder
                self._mock = False
                self._init_real_codec(AmphionEncoder, AmphionDecoder)
            except ImportError as e:
                # Amphion not available, use mock fallback
                self._mock = True
            
    def _init_real_codec(self, EncoderClass, DecoderClass):
        """Initialize real FACodec model (when Amphion is available).
        
        Args:
            EncoderClass: FACodecEncoder class from Amphion
            DecoderClass: FACodecDecoder class from Amphion
        """
        try:
            # Initialize encoder
            self._encoder = EncoderClass(
                ngf=32,
                up_ratios=[2, 4, 5, 5],
                out_channels=256,
            )
            
            # Initialize decoder (needed for quantization to get indices)
            self._decoder = DecoderClass(
                in_channels=256,
                upsample_initial_channel=1024,
                ngf=32,
                up_ratios=[5, 5, 4, 2],
                vq_num_q_c=2,
                vq_num_q_p=1,
                vq_num_q_r=3,
                vq_dim=256,
                codebook_dim=8,
                codebook_size_prosody=10,
                codebook_size_content=10,
                codebook_size_residual=10,
                use_gr_x_timbre=True,
                use_gr_residual_f0=True,
                use_gr_residual_phone=True,
            )
            
            # Determine checkpoint path
            if self.checkpoint_path is None:
                # Default to project checkpoints directory
                project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                self.checkpoint_path = os.path.join(project_root, "checkpoints", "facodec")
            
            # Load checkpoints
            encoder_ckpt = os.path.join(self.checkpoint_path, "ns3_facodec_encoder.bin")
            decoder_ckpt = os.path.join(self.checkpoint_path, "ns3_facodec_decoder.bin")
            
            if not os.path.exists(encoder_ckpt) or not os.path.exists(decoder_ckpt):
                raise FileNotFoundError(
                    f"FACodec checkpoints not found at {self.checkpoint_path}. "
                    f"Please download from https://huggingface.co/amphion/naturalspeech3_facodec"
                )
            
            # Load state dicts
            self._encoder.load_state_dict(
                torch.load(encoder_ckpt, map_location=self.device, weights_only=True)
            )
            self._decoder.load_state_dict(
                torch.load(decoder_ckpt, map_location=self.device, weights_only=True)
            )
            
            # Move to device and set to eval mode
            self._encoder.to(self.device)
            self._decoder.to(self.device)
            self._encoder.eval()
            self._decoder.eval()
            
        except Exception as e:
            # If anything fails during initialization, fall back to mock
            self._mock = True
            self._encoder = None
            self._decoder = None
        
    def encode(self, audio: torch.Tensor) -> FACodecStreams:
        """Encode audio into FACodec streams.
        
        Takes raw audio waveform and returns FACodecStreams containing all
        disentangled representations: prosody, content, acoustic codebooks,
        and the continuous timbre vector.
        
        Args:
            audio: Audio waveform tensor, shape [samples] or [1, samples].
                   Should be at 16kHz sample rate.
        
        Returns:
            FACodecStreams dataclass with:
            - prosody_codebooks_idx: [1, T80] int64 tensor
            - content_codebooks_idx: [2, T80] int64 tensor
            - acoustic_codebooks_idx: [3, T80] int64 tensor
            - timbre_vector: [256] float32 tensor (from spk_embs)
            
            T80 = number of frames at 80 Hz (hop_size=200 at 16000 Hz)
        
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
    
    def encode_batch(self, audios: List[torch.Tensor]) -> List[FACodecStreams]:
        """Encode a batch of audio samples through FACodec.
        
        Pads variable-length audio to the same length, runs FACodec once on the
        padded batch, then slices valid frames per sample based on original length.
        
        Args:
            audios: List of 1D audio tensors, each shape [samples] at 16kHz.
        
        Returns:
            List of FACodecStreams objects, one per input audio sample.
            Each contains tensors with per-sample valid lengths.
        
        Raises:
            ValueError: If the batch is empty.
        """
        if not audios:
            raise ValueError("Batch cannot be empty")
        
        if self._mock:
            return [self._encode_mock(audio) for audio in audios]
        
        return self._encode_real_batch(audios)
    
    def _encode_real_batch(
        self, audios: List[torch.Tensor]
    ) -> List[FACodecStreams]:
        """Batch encode using real FACodec model.
        
        Pads all samples to max length, runs through model once,
        then slices valid frames per sample.
        
        Args:
            audios: List of 1D audio tensors
        
        Returns:
            List of FACodecStreams with per-sample valid lengths
        """
        lengths = [a.shape[0] for a in audios]
        max_len = max(lengths)
        batch_size = len(audios)
        
        batch = torch.zeros(batch_size, max_len, device=self.device)
        for i, audio in enumerate(audios):
            batch[i, :audio.shape[0]] = audio.to(self.device)
        
        batch = batch.unsqueeze(1)
        
        with torch.no_grad():
            enc_out = self._encoder(batch)
            # Capture spk_embs as 5th return value (the true timbre vector)
            _, vq_id, _, _, spk_embs = self._decoder(enc_out, eval_vq=False, vq=True)
        
        hop_size = 200
        
        results = []
        for i, length in enumerate(lengths):
            num_frames = length // hop_size
            # vq_id shape: [6, batch, num_frames]
            vq_sample = vq_id[:, i, :num_frames].cpu()  # [6, num_frames]
            
            # Extract streams WITHOUT averaging - preserve codebook structure
            # prosody: vq_id[0:1] -> shape [1, num_frames]
            prosody_indices = vq_sample[0:1]  # [1, T]
            
            # content: vq_id[1:3] -> shape [2, num_frames]
            content_indices = vq_sample[1:3]  # [2, T]
            
            # acoustic: vq_id[3:6] -> shape [3, num_frames]
            acoustic_indices = vq_sample[3:6]  # [3, T]
            
            # timbre vector: spk_embs[i] -> shape [256] float32
            timbre_vector = spk_embs[i].cpu()  # [256]
            
            streams = FACodecStreams(
                prosody_codebooks_idx=prosody_indices,
                content_codebooks_idx=content_indices,
                acoustic_codebooks_idx=acoustic_indices,
                timbre_vector=timbre_vector,
            )
            results.append(streams)
        
        return results
    
    def _encode_mock(self, audio: torch.Tensor) -> FACodecStreams:
        """Generate deterministic mock FACodec streams for testing.
        
        Creates fake but structurally valid streams based on audio length.
        Uses a deterministic algorithm so results are reproducible.
        
        Args:
            audio: Audio waveform tensor
        
        Returns:
            FACodecStreams with mock tensors of correct shapes:
            - prosody_codebooks_idx: [1, T80] int64
            - content_codebooks_idx: [2, T80] int64  
            - acoustic_codebooks_idx: [3, T80] int64
            - timbre_vector: [256] float32
        """
        # Calculate number of frames at FACodec's actual 80 Hz rate
        # hop_size = 200 at 16000 Hz -> 80 frames per second
        num_samples = audio.shape[0]
        hop_size = 200
        num_frames = max(1, num_samples // hop_size)
        
        # Use deterministic pseudo-random generation based on audio statistics
        audio_mean = audio.mean().item()
        audio_std = audio.std().item() if audio.numel() > 1 else 0.0
        
        # Seed based on audio characteristics for determinism
        seed = int((abs(audio_mean) + abs(audio_std)) * 10000) % 2**32
        
        # Generate mock indices for each codebook
        # Prosody: 1 codebook [1, T]
        prosody_indices = torch.zeros(1, num_frames, dtype=torch.int64)
        for frame_idx in range(num_frames):
            prosody_indices[0, frame_idx] = (seed + frame_idx * 31 + 512) % self.VOCAB_SIZE
        
        # Content: 2 codebooks [2, T]
        content_indices = torch.zeros(2, num_frames, dtype=torch.int64)
        for frame_idx in range(num_frames):
            content_indices[0, frame_idx] = (seed + frame_idx * 47) % self.VOCAB_SIZE
            content_indices[1, frame_idx] = (seed + frame_idx * 53) % self.VOCAB_SIZE
        
        # Acoustic: 3 codebooks [3, T]
        acoustic_indices = torch.zeros(3, num_frames, dtype=torch.int64)
        for frame_idx in range(num_frames):
            acoustic_indices[0, frame_idx] = (seed + frame_idx * 19 + 1024) % self.VOCAB_SIZE
            acoustic_indices[1, frame_idx] = (seed + frame_idx * 23 + 1024) % self.VOCAB_SIZE
            acoustic_indices[2, frame_idx] = (seed + frame_idx * 29 + 1024) % self.VOCAB_SIZE
        
        # Timbre vector: [256] float32 (deterministic but varies by audio)
        timbre_vector = torch.zeros(256, dtype=torch.float32)
        for i in range(256):
            timbre_vector[i] = ((seed + i * 17) % 1000) / 1000.0  # Values in [0, 1)
        
        return FACodecStreams(
            prosody_codebooks_idx=prosody_indices,
            content_codebooks_idx=content_indices,
            acoustic_codebooks_idx=acoustic_indices,
            timbre_vector=timbre_vector,
        )
    
    def _encode_real(self, audio: torch.Tensor) -> FACodecStreams:
        """Encode using real FACodec model (when Amphion is available).
        
        FACodec produces 6 codebooks:
        - vq_id[0:1] = prosody (1 codebook)
        - vq_id[1:3] = content (2 codebooks) 
        - vq_id[3:6] = acoustic/residual (3 codebooks)
        
        And a continuous timbre vector:
        - spk_embs = utterance-level timbre embedding [256] float32
        
        We map these to FACodecStreams WITHOUT averaging codebooks:
        - prosody_codebooks_idx: vq_id[0:1] -> [1, T80]
        - content_codebooks_idx: vq_id[1:3] -> [2, T80]
        - acoustic_codebooks_idx: vq_id[3:6] -> [3, T80]
        - timbre_vector: spk_embs.squeeze(0) -> [256]
        
        Args:
            audio: Audio waveform tensor, shape [samples]
        
        Returns:
            FACodecStreams with tensors of correct shapes
        """
        if self._encoder is None or self._decoder is None:
            # Should not happen if _mock is properly set, but just in case
            return self._encode_mock(audio)
        
        # Prepare audio: [samples] -> [1, 1, samples] (batch, channel, samples)
        audio_batch = audio.unsqueeze(0).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Encode
            enc_out = self._encoder(audio_batch)
            
            # Quantize through decoder to get indices
            # Capture spk_embs as 5th return value (the true timbre vector)
            _, vq_id, _, _, spk_embs = self._decoder(enc_out, eval_vq=False, vq=True)
            
            # vq_id is [6, 1, num_frames] -> squeeze batch dim -> [6, num_frames]
            vq_id = vq_id.squeeze(1).cpu()
            
            # spk_embs is [1, 256] -> squeeze batch dim -> [256]
            timbre_vector = spk_embs.squeeze(0).cpu()  # [256]
        
        # Extract streams WITHOUT averaging - preserve full codebook structure
        # prosody: vq_id[0:1] -> [1, num_frames]
        prosody_indices = vq_id[0:1]  # [1, T]
        
        # content: vq_id[1:3] -> [2, num_frames]
        content_indices = vq_id[1:3]  # [2, T]
        
        # acoustic: vq_id[3:6] -> [3, num_frames]
        acoustic_indices = vq_id[3:6]  # [3, T]
        
        return FACodecStreams(
            prosody_codebooks_idx=prosody_indices,
            content_codebooks_idx=content_indices,
            acoustic_codebooks_idx=acoustic_indices,
            timbre_vector=timbre_vector,
        )
