"""
Prepare the data for training the compressor model.
During training, we will use the following data:
- Speech data; 
- WavLM features for distillation
- Qwen hidden states for distillation
- Duration of each word in the speech data

Each training sample will be:
- Speech data: [B, 1, 24000*time_in_seconds]
- WavLM features: [B, T1, 768]
- Qwen hidden states: [B, T2, 2048]
- Duration of each word: [B, T]
"""

import json
import torch
import torch.nn.functional as F
import torchaudio
import numpy as np
import lightning as L
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

# Constants
RAW_AUDIO_REPO = "speechcolab/gigaspeech"
ALIGN_DATASET_REPO = "hungphongtrn/speech-time-alignment"
WAVLM_FEATURES_REPO = "hungphongtrn/wavlm-features"
LLLM_HIDDEN_STATES_REPO = "hungphongtrn/llm-hidden-states"

TARGET_SAMPLE_RATE = 24000
CODEC_FRAME_RATE = 12.5
HOP_LENGTH = int(TARGET_SAMPLE_RATE / CODEC_FRAME_RATE)  # 1920

class CompressorDataset(Dataset):
    def __init__(self, audio_ds, align_ds, wavlm_ds, llm_ds):
        self.audio_ds = audio_ds
        self.align_ds = align_ds
        self.wavlm_ds = wavlm_ds
        self.llm_ds = llm_ds
        
    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        # 1. Load Data Items
        audio_item = self.audio_ds[idx]
        align_item = self.align_ds[idx]
        wavlm_item = self.wavlm_ds[idx]
        llm_item = self.llm_ds[idx]

        # 2. Process Audio
        # Assuming audio_item['audio'] is a dict with 'array' and 'sampling_rate'
        audio_array = audio_item['audio']['array']
        sr = audio_item['audio']['sampling_rate']
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)

        # Resample if necessary
        if sr != TARGET_SAMPLE_RATE:
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, TARGET_SAMPLE_RATE)
        
        # Ensure audio is mono [1, T] or [T]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0) # [1, T]

        num_frames = int(audio_tensor.shape[-1] / HOP_LENGTH)
        
        # 3. Process WavLM Features
        # shape: [T_wavlm, D] -> [1, D, T_wavlm] for interpolate
        wavlm_feat = torch.tensor(wavlm_item['wavlm_feat'], dtype=torch.float32)
        if wavlm_feat.ndim == 2:
            wavlm_feat = wavlm_feat.transpose(0, 1).unsqueeze(0) # [1, D, T_wavlm]
            
        # Downsample to match codec frame rate
        # We need output size equal to num_frames
        if num_frames > 0:
            wavlm_aligned = F.interpolate(wavlm_feat, size=num_frames, mode='linear', align_corners=False)
            wavlm_aligned = wavlm_aligned.squeeze(0).transpose(0, 1) # [T, D]
        else:
             wavlm_aligned = torch.zeros((0, wavlm_feat.shape[1]))

        # 4. Process Qwen Hidden States
        # shape: [N_tokens, D]
        # We need to align tokens to frames using timestamps
        qwen_feat = torch.tensor(llm_item['llm_feat'], dtype=torch.float32)
        llm_times = np.array(llm_item['llm_times']) # [N_tokens, 2] (start, end)
        
        qwen_aligned = torch.zeros((num_frames, qwen_feat.shape[1]), dtype=qwen_feat.dtype)
        
        if num_frames > 0 and len(llm_times) > 0:
            # Create frame center timestamps
            frame_centers = (np.arange(num_frames) * HOP_LENGTH / TARGET_SAMPLE_RATE) + (0.5 * HOP_LENGTH / TARGET_SAMPLE_RATE)
            
            # For each frame, find the corresponding token
            # This can be optimized, but simple loop is clear
            # Or use broadcasting/masks
            
            # Simple greedy assignment: find token where start <= t_center <= end
            # Using numpy logic
            # llm_times[:, 0] <= centers[None, :] & llm_times[:, 1] >= centers[None, :]
            
            # Iterate for safety (or vectorization if consistent)
            for t_idx, t_center in enumerate(frame_centers):
                # Find token index
                matches = np.where((llm_times[:, 0] <= t_center) & (llm_times[:, 1] >= t_center))[0]
                if len(matches) > 0:
                    token_idx = matches[0] # Take first match
                    qwen_aligned[t_idx] = qwen_feat[token_idx]
                else:
                    # Silence/No token -> Keep zero or use previous?
                    # Zero is safer for "no semantic content"
                    pass

        # 5. Process Duration/Word Alignment (Optional based on docstring)
        # Construct a duration tensor or similar. 
        # Docstring says "Duration of each word: [B, T]". 
        # I'll create a tensor indicating the word index or just 1.0/0.0 mask?
        # Let's assume it means "Seconds per word" broadcasted to frames? 
        # Or maybe "Word boundaries"?
        # Given "Using duration information... to downsample...", we already used it.
        # But it lists it as an output. 
        # I will return a dummy or the word_id per frame.
        
        # Let's return the alignment_json as processed tensor (e.g. word IDs)
        # Using alignment_json from align_item
        word_alignment = torch.zeros(num_frames, dtype=torch.float32) # placeholder
        
        # If we have word timestamps, we can fill this
        try:
            words = json.loads(align_item['alignment_json'])
            # words is list of {'word':..., 'start':..., 'end':...}
            for i, word in enumerate(words):
                # map start/end to frames
                start_frame = int(word['start'] * CODEC_FRAME_RATE)
                end_frame = int(word['end'] * CODEC_FRAME_RATE)
                # Clamp
                start_frame = max(0, min(start_frame, num_frames))
                end_frame = max(0, min(end_frame, num_frames))
                word_alignment[start_frame:end_frame] = word['end'] - word['start'] # Example: fill with duration
        except Exception:
            pass

        return {
            "audio": audio_tensor.squeeze(0), # [T]
            "wavlm": wavlm_aligned,    # [T, 768/1024]
            "qwen": qwen_aligned,      # [T, 2048]
            "duration": word_alignment # [T]
        }

def collate_fn(batch):
    # Pad all sequences to the longest in the batch
    
    # 1. Audio
    audios = [b['audio'] for b in batch]
    audio_lens = [len(a) for a in audios]
    max_audio_len = max(audio_lens)
    padded_audio = torch.zeros(len(batch), max_audio_len)
    
    # 2. WavLM
    wavlms = [b['wavlm'] for b in batch]
    wavlm_dim = wavlms[0].shape[1]
    max_frame_len = max([len(w) for w in wavlms])
    padded_wavlm = torch.zeros(len(batch), max_frame_len, wavlm_dim)

    # 3. Qwen
    qwens = [b['qwen'] for b in batch]
    qwen_dim = qwens[0].shape[1]
    padded_qwen = torch.zeros(len(batch), max_frame_len, qwen_dim)
    
    # 4. Duration
    durations = [b['duration'] for b in batch]
    padded_duration = torch.zeros(len(batch), max_frame_len)
    
    # Fill
    for i in range(len(batch)):
        # Audio
        end_audio = len(audios[i])
        padded_audio[i, :end_audio] = audios[i]
        
        # Frames
        end_frame = len(wavlms[i]) # Should be same for wavlm/qwen/duration
        padded_wavlm[i, :end_frame] = wavlms[i]
        padded_qwen[i, :end_frame] = qwens[i]
        padded_duration[i, :end_frame] = durations[i]
        
    return {
        "audio": padded_audio,
        "wavlm": padded_wavlm,
        "qwen": padded_qwen,
        "duration": padded_duration
    }

class CompressorDataLoader(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str = None):
        # Load datasets
        if stage == "fit" or stage is None:
            self.train_dataset = self._load_split("train")
            self.val_dataset = self._load_split("validation")
        elif stage == "validate":
            self.val_dataset = self._load_split("validation")

    def _load_split(self, split: str):
        try:
            raw_audio = load_dataset(RAW_AUDIO_REPO, split=split, trust_remote_code=True)
            align_ds = load_dataset(ALIGN_DATASET_REPO, split=split)
            wavlm_ds = load_dataset(WAVLM_FEATURES_REPO, split=split)
            llm_ds = load_dataset(LLLM_HIDDEN_STATES_REPO, split=split)
            return CompressorDataset(raw_audio, align_ds, wavlm_ds, llm_ds)
        except Exception as e:
            print(f"Warning: Could not load split '{split}': {e}")
            return None

    def train_dataloader(self):
        if self.train_dataset:
            return DataLoader(
                self.train_dataset, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )
        return None

    def val_dataloader(self):
        if self.val_dataset:
            return DataLoader(
                self.val_dataset, 
                batch_size=self.batch_size, 
                shuffle=False, 
                num_workers=self.num_workers,
                collate_fn=collate_fn
            )
