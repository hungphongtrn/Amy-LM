"""Temporal pooling for downsampling embedded prosody streams."""
import torch.nn as nn
import torch.nn.functional as F


class TemporalPool(nn.Module):
    """Compress embedded prosody using adaptive average pooling.

    Takes embedded prosody at ``input_rate`` Hz (typically 80 Hz from FACodec)
    and outputs at ``output_rate`` Hz (typically 12.5 Hz for Amy-LM).

    Uses ``F.adaptive_avg_pool1d`` internally, which handles non-integer
    downsampling ratios (e.g., 80/12.5 = 6.4) by varying window sizes:
    most output frames average 6 input frames, with occasional 7-frame
    windows to absorb the 0.4 remainder. This preserves exact frame
    alignment with all downstream codebooks at 12.5 Hz.

    Args:
        input_rate: Input sequence frame rate in Hz (default 80.0).
        output_rate: Target output frame rate in Hz (default 12.5).
    """

    def __init__(self, input_rate: float = 80.0, output_rate: float = 12.5):
        super().__init__()
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.ratio = input_rate / output_rate

    def forward(self, x):
        """Pool embedded prosody along the time dimension.

        Args:
            x: Tensor shape ``(batch, time, embed_dim)`` at input_rate Hz.

        Returns:
            Tensor shape ``(batch, target_time, embed_dim)`` at output_rate Hz.
        """
        batch, length, embed_dim = x.shape
        # Calculate target length based on time duration for accurate alignment
        # Using round() to preserve exact frame alignment with downstream codebooks
        duration_sec = length / self.input_rate
        target_len = max(1, round(duration_sec * self.output_rate))
        x_t = x.transpose(1, 2)
        pooled = F.adaptive_avg_pool1d(x_t, target_len)
        return pooled.transpose(1, 2)
