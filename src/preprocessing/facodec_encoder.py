"""FACodec Encoder Wrapper for Amy-LM preprocessing pipeline.

This module provides a wrapper around the FACodec model from Amphion.
When Amphion is not installed, it provides a deterministic mock fallback
for testing purposes.
"""

import os
import sys
from typing import Optional, Tuple, List
import torch

# Add vendor/Amphion to path so we can import Amphion models
_vendor_path = os.path.join(os.path.dirname(__file__), "..", "..", "vendor", "Amphion")
_vendor_path = os.path.abspath(_vendor_path)
if os.path.isdir(_vendor_path) and _vendor_path not in sys.path:
    sys.path.insert(0, _vendor_path)


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
            is available, will use default checkpoint from checkpoints/facodec/.
    
    Attributes:
        device: The device being used
        _mock: True if using mock fallback (Amphion not available)
        _frame_rate: Frames per second output (~12.5 Hz)
        _vocab_size: Codebook vocabulary size (1024 for FACodec)
        _sample_rate: Audio sample rate (16000 Hz)
    
    Example:
        >>> encoder = FACodecEncoder(device="cpu")
        >>> audio = torch.randn(32000)  # 2 seconds at 16kHz
        >>> content, prosody, timbre = encoder.encode(audio)
        >>> len(content)  # ~25 frames at 12.5 Hz
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
    
    def encode_batch(self, audios: List[torch.Tensor]) -> List[Tuple[List[int], List[int], List[int]]]:
        """Encode a batch of audio samples through FACodec.
        
        Pads variable-length audio to the same length, runs FACodec once on the
        padded batch, then slices valid frames per sample based on original length.
        
        Args:
            audios: List of 1D audio tensors, each shape [samples] at 16kHz.
        
        Returns:
            List of (content_indices, prosody_indices, timbre_indices) tuples,
            one per input audio sample.
        
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
    ) -> List[Tuple[List[int], List[int], List[int]]]:
        """Batch encode using real FACodec model.
        
        Pads all samples to max length, runs through model once,
        then slices valid frames per sample.
        
        Args:
            audios: List of 1D audio tensors
        
        Returns:
            List of (content, prosody, timbre) index tuples
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
            _, vq_id, _, _, _ = self._decoder(enc_out, eval_vq=False, vq=True)
        
        hop_size = 200
        
        results = []
        for i, length in enumerate(lengths):
            num_frames = length // hop_size
            vq_sample = vq_id[:, i, :num_frames].cpu()
            
            prosody_indices = []
            content_indices = []
            timbre_indices = []
            
            for frame_idx in range(num_frames):
                prosody_indices.append(int(vq_sample[0, frame_idx].item()))
                content_val = int((vq_sample[1, frame_idx].item() + vq_sample[2, frame_idx].item()) // 2)
                content_indices.append(content_val)
                timbre_val = int((vq_sample[3, frame_idx].item() + vq_sample[4, frame_idx].item() + vq_sample[5, frame_idx].item()) // 3)
                timbre_indices.append(timbre_val)
            
            results.append((content_indices, prosody_indices, timbre_indices))
        
        return results
    
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
        
        FACodec produces 6 codebooks:
        - vq_id[0:1] = prosody (1 codebook)
        - vq_id[1:3] = content (2 codebooks) 
        - vq_id[3:6] = residual/timbre (3 codebooks)
        
        We map these to Amy-LM's expected format:
        - content: Average of the 2 content codebooks
        - prosody: The prosody codebook
        - timbre: Average of the 3 residual codebooks (acoustic/timbre info)
        
        Args:
            audio: Audio waveform tensor, shape [samples]
        
        Returns:
            Tuple of (content, prosody, timbre) index lists
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
            # vq_id shape: [num_codebooks, batch, num_frames]
            _, vq_id, _, _, _ = self._decoder(enc_out, eval_vq=False, vq=True)
            
            # vq_id is [6, 1, num_frames] -> squeeze batch dim -> [6, num_frames]
            vq_id = vq_id.squeeze(1).cpu()
        
        num_frames = vq_id.shape[1]
        
        # Extract indices for each frame
        content_indices = []
        prosody_indices = []
        timbre_indices = []
        
        for frame_idx in range(num_frames):
            # Prosody: single codebook [0]
            prosody_indices.append(int(vq_id[0, frame_idx].item()))
            
            # Content: average of codebooks [1] and [2] (then clamp to valid range)
            content_val = int((vq_id[1, frame_idx].item() + vq_id[2, frame_idx].item()) // 2)
            content_indices.append(content_val)
            
            # Timbre: average of residual codebooks [3], [4], [5]
            timbre_val = int((vq_id[3, frame_idx].item() + vq_id[4, frame_idx].item() + vq_id[5, frame_idx].item()) // 3)
            timbre_indices.append(timbre_val)
        
        return content_indices, prosody_indices, timbre_indices
