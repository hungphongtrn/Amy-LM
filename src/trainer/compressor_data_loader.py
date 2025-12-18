"""
Audio Feature Alignment and Preprocessing Module.

This module provides functionality to synchronize and process multi-modal audio datasets
for the "Amy" model training pipeline. It specifically handles the alignment of heterogeneous
feature streams (raw audio, WavLM features, and LLM semantic features) into a unified
temporal grid defined by the Mimi codec.

Key Operations:
    1.  **Audio Resampling**: Standardizes raw audio to a specific codec sample rate (e.g., 24kHz).
    2.  **WavLM Alignment**: Down-samples dense acoustic features using adaptive average pooling
        to match the target frame rate.
    3.  **LLM Feature Alignment**: Projects sparse, interval-based semantic features onto the
        continuous target time grid using an integral image (prefix sum) approach for efficient
        temporal averaging.

Constants:
    DATASET (str): Path to the input dataset.
    MIMI_FPS (float): Target frames per second for the alignment grid (12.5 Hz).
    MIMI_SAMPLE_RATE (int): Target sampling rate for audio waveforms (24000 Hz).

Each batch will contain:
    - audio: (B, 1, T_samples) at 
    - wavlm_feat: (B, T_frames, D_wavlm)
    - llm_feat: (B, T_frames, D_llm)
"""

import lightning as L
import torch
import torchaudio
import numpy as np
from torch.utils.data import DataLoader, random_split
from datasets import load_from_disk
from typing import Optional, List, Dict


class CompressorDataLoader(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for Mimi Model training.

    Features:
    - Loads a HuggingFace dataset from disk.
    - Splits into 80% Train / 20% Test (Validation).
    - Uses a custom Collator to resample audio and align features on-the-fly.
    - Handles random cropping to fixed durations for batch training.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        sample_rate: int = 24000,
        fps: float = 12.5,
        segment_duration: float = 1.0,  # Duration in seconds for training crops
        seed: int = 42,
    ):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_rate = sample_rate
        self.fps = fps
        self.segment_duration = segment_duration
        self.seed = seed
        self.save_hyperparameters()

    def setup(self, stage: Optional[str] = None):
        # Load dataset from disk
        # We assume the dataset has columns: 'audio', 'wavlm_feat', 'llm_feat', 'llm_times'
        full_dataset = load_from_disk(self.data_path)

        # Calculate split sizes
        total_size = len(full_dataset)
        train_size = int(0.8 * total_size)
        val_size = total_size - train_size

        # reproducible split
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(
            full_dataset, [train_size, val_size], generator=generator
        )

    def train_dataloader(self):
        # For training, we enable random cropping in the collator
        collator = AudioCollator(
            sample_rate=self.sample_rate,
            fps=self.fps,
            crop_duration=self.segment_duration,
            training=True,
        )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )

    def val_dataloader(self):
        # For validation, we can either crop deterministically or return full sequences.
        # Here we crop to ensure consistent batch shapes for metric calculation.
        collator = AudioCollator(
            sample_rate=self.sample_rate,
            fps=self.fps,
            crop_duration=self.segment_duration,
            training=False,
        )
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
            pin_memory=True,
        )


class AudioCollator:
    """
    Custom Collator that handles:
    1. Resampling Audio to target codec sample rate.
    2. Aligning WavLM and LLM features to the target frame rate.
    3. Randomly cropping the signal and features to a fixed duration.
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        fps: float = 12.5,
        crop_duration: float = 1.0,
        training: bool = True,
    ):
        self.sample_rate = sample_rate
        self.fps = fps
        self.crop_duration = crop_duration
        self.training = training

        # Calculate target sizes
        self.target_samples = int(self.sample_rate * self.crop_duration)
        self.target_frames = int(np.ceil(self.crop_duration * self.fps))

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        batch_audio = []
        batch_wavlm = []
        batch_llm = []

        for item in batch:
            # --- 1. Audio Processing ---
            # Load and resample
            src_audio = torch.tensor(item["audio"]["array"]).float()
            src_sr = item["audio"]["sampling_rate"]

            if src_sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(src_sr, self.sample_rate)
                audio = resampler(src_audio.unsqueeze(0)).squeeze(0)
            else:
                audio = src_audio

            # Determine crop offsets
            audio_len = audio.shape[0]
            if audio_len > self.target_samples:
                if self.training:
                    start_sample = torch.randint(
                        0, audio_len - self.target_samples, (1,)
                    ).item()
                else:
                    start_sample = 0  # Deterministic for validation
            else:
                start_sample = 0

            # --- 2. Align Features (Simplified Single-Item Logic) ---
            # We align the *full* sequence first, then crop.
            # (Optimization: You could align only the crop window, but handling LLM boundaries is trickier)

            # A. WavLM Alignment
            wavlm_raw = (
                torch.tensor(item["wavlm_feat"]).float().transpose(0, 1).unsqueeze(0)
            )  # (1, D, T_orig)

            # Calculate how many frames correspond to the CURRENT audio length
            # Note: We use the actual duration of the resampled audio to stay in sync
            duration = audio.shape[0] / self.sample_rate
            total_frames = int(np.ceil(duration * self.fps))

            wavlm_aligned = torch.nn.functional.adaptive_avg_pool1d(
                wavlm_raw, total_frames
            )
            wavlm_aligned = wavlm_aligned.squeeze(0).transpose(0, 1)  # (T_frames, D)

            # B. LLM Alignment
            # (Reusing the robust logic from process_batch, adapted for single item)
            llm_feat = np.array(item["llm_feat"])
            llm_times = np.array(item["llm_times"])

            # Grid for this specific item
            frame_indices = np.arange(total_frames)
            grid_starts = frame_indices / self.fps
            grid_ends = (frame_indices + 1) / self.fps

            # Fix LLM times (-1 means end of audio)
            t_starts = llm_times[:, 0]
            t_ends = llm_times[:, 1].copy()
            t_ends[t_ends == -1] = duration

            # Align
            idx_closest_start = np.searchsorted(t_starts, grid_starts, side="right") - 1
            idx_closest_start = np.clip(idx_closest_start, 0, len(t_starts) - 1)
            val_closest_start = t_starts[idx_closest_start]
            final_idx_starts = np.searchsorted(t_starts, val_closest_start, side="left")

            idx_closest_end = np.searchsorted(t_ends, grid_ends, side="left")
            idx_closest_end = np.clip(idx_closest_end, 0, len(t_ends) - 1)
            val_closest_end = t_ends[idx_closest_end]
            final_idx_ends = np.searchsorted(t_ends, val_closest_end, side="right") - 1

            # Prefix sum
            feat_cumsum = np.vstack(
                [np.zeros((1, llm_feat.shape[1])), np.cumsum(llm_feat, axis=0)]
            )
            sums = feat_cumsum[final_idx_ends + 1] - feat_cumsum[final_idx_starts]
            counts = (final_idx_ends - final_idx_starts + 1).reshape(-1, 1)
            counts = np.maximum(counts, 1)
            llm_aligned = torch.tensor((sums / counts).astype(np.float32))

            # --- 3. Cropping & Padding ---
            # Calculate frame start based on sample start
            start_frame = int(np.floor((start_sample / self.sample_rate) * self.fps))

            # Crop Audio
            audio_crop = audio[start_sample : start_sample + self.target_samples]
            # Pad Audio if needed
            if audio_crop.shape[0] < self.target_samples:
                pad_amt = self.target_samples - audio_crop.shape[0]
                audio_crop = torch.nn.functional.pad(audio_crop, (0, pad_amt))

            # Crop Features
            wavlm_crop = wavlm_aligned[start_frame : start_frame + self.target_frames]
            llm_crop = llm_aligned[start_frame : start_frame + self.target_frames]

            # Pad Features if needed
            if wavlm_crop.shape[0] < self.target_frames:
                pad_amt = self.target_frames - wavlm_crop.shape[0]
                wavlm_crop = torch.nn.functional.pad(wavlm_crop, (0, 0, 0, pad_amt))
            if llm_crop.shape[0] < self.target_frames:
                pad_amt = self.target_frames - llm_crop.shape[0]
                llm_crop = torch.nn.functional.pad(llm_crop, (0, 0, 0, pad_amt))

            batch_audio.append(audio_crop)
            batch_wavlm.append(wavlm_crop)
            batch_llm.append(llm_crop)

        # Stack into batches
        return {
            "audio": torch.stack(batch_audio).unsqueeze(1),  # (B, 1, T_samples)
            "wavlm_feat": torch.stack(batch_wavlm),  # (B, T_frames, D_wavlm)
            "llm_feat": torch.stack(batch_llm),  # (B, T_frames, D_llm)
        }


if __name__ == "__main__":
    # Test configuration
    TEST_DATA_PATH = "data/Amy-LM-Dataset"  # Ensure this path exists or change it
    BATCH_SIZE = 2
    DURATION = 1.0  # seconds

    print(f"--- Testing CompressorDataLoader with path: {TEST_DATA_PATH} ---")

    try:
        # 1. Initialize Module
        data_module = CompressorDataLoader(
            data_path=TEST_DATA_PATH,
            batch_size=BATCH_SIZE,
            segment_duration=DURATION,
            num_workers=0,  # Set to 0 for easier debugging
        )

        # 2. Setup (Load & Split)
        print("Setting up data module...")
        data_module.setup()
        print(f"Train set size: {len(data_module.train_ds)}")
        print(f"Val set size: {len(data_module.val_ds)}")

        # 3. Fetch a single batch
        print("\nFetching one batch from train dataloader...")
        loader = data_module.train_dataloader()
        batch = next(iter(loader))

        # 4. Inspect Shapes
        print("\n--- Batch Inspection ---")
        audio = batch["audio"]
        wavlm = batch["wavlm_feat"]
        llm = batch["llm_feat"]

        print(f"Audio Shape  [B, C, T_samples]: {audio.shape}")
        print(f"WavLM Shape  [B, T_frames, D] : {wavlm.shape}")
        print(f"LLM Shape    [B, T_frames, D] : {llm.shape}")

        # 5. Validation Checks
        expected_samples = int(data_module.sample_rate * DURATION)
        expected_frames = int(np.ceil(DURATION * data_module.fps))

        assert audio.shape[-1] == expected_samples, (
            f"Audio length mismatch! Got {audio.shape[-1]}, expected {expected_samples}"
        )
        assert wavlm.shape[1] == expected_frames, (
            f"Feature frame count mismatch! Got {wavlm.shape[1]}, expected {expected_frames}"
        )
        assert wavlm.shape[1] == llm.shape[1], (
            "WavLM and LLM features are not aligned temporally!"
        )

        print("\n✅ Test Passed: Shapes align with target duration and sample rate.")

    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        # Hint: check if the dataset path is correct if this fails immediately.
