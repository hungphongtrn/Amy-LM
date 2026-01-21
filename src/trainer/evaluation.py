"""
Stage 1 Intrinsic Evaluation Module for Amy-LM.

Implements:
1. SI-SDR (Scale-Invariant Signal-to-Distortion Ratio)
2. Reconstruction Ablation per codebook group
3. Codebook Health Metrics (entropy, usage)
4. Probing Classifiers (phoneme/pitch)
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# =============================================================================
# SI-SDR Metric
# =============================================================================

def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio.
    
    Args:
        estimate: Estimated signal [B, 1, T] or [B, T]
        reference: Reference signal [B, 1, T] or [B, T]
        eps: Small constant for numerical stability
        
    Returns:
        SI-SDR in dB, averaged over batch
    """
    # Flatten to [B, T]
    if estimate.dim() == 3:
        estimate = estimate.squeeze(1)
    if reference.dim() == 3:
        reference = reference.squeeze(1)
    
    # Ensure same length
    min_len = min(estimate.shape[-1], reference.shape[-1])
    estimate = estimate[..., :min_len]
    reference = reference[..., :min_len]
    
    # Zero-mean normalization
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    reference = reference - reference.mean(dim=-1, keepdim=True)
    
    # SI-SDR computation
    # s_target = <s', s> * s / ||s||^2
    dot = (estimate * reference).sum(dim=-1, keepdim=True)
    s_target_energy = (reference ** 2).sum(dim=-1, keepdim=True) + eps
    s_target = (dot / s_target_energy) * reference
    
    # e_noise = s' - s_target
    e_noise = estimate - s_target
    
    # SI-SDR = 10 * log10(||s_target||^2 / ||e_noise||^2)
    si_sdr_value = 10 * torch.log10(
        (s_target ** 2).sum(dim=-1) / ((e_noise ** 2).sum(dim=-1) + eps) + eps
    )
    
    return si_sdr_value.mean()


# =============================================================================
# Reconstruction Ablation
# =============================================================================

@dataclass
class ReconstructionAblationResult:
    """Results from reconstruction ablation analysis."""
    sisdr_semantic_only: float  # Codebook 0 only
    sisdr_semantic_prosody: float  # Codebooks 0+1
    sisdr_all: float  # All codebooks


def reconstruction_ablation(
    model: nn.Module,
    audio: torch.Tensor,
) -> ReconstructionAblationResult:
    """
    Perform reconstruction ablation by decoding subsets of codebooks.
    
    Args:
        model: MimiModel instance
        audio: Input audio [B, 1, T]
        
    Returns:
        ReconstructionAblationResult with SI-SDR for each codebook group
    """
    device = audio.device
    
    # Pad for encoding
    frame_size = model.frame_size
    T_samples = audio.shape[-1]
    
    if T_samples % frame_size != 0:
        pad_len = frame_size - (T_samples % frame_size)
        audio_padded = F.pad(audio, (0, pad_len))
    else:
        audio_padded = audio
    
    # Encode to codes [B, K, T_frames]
    with torch.no_grad():
        codes = model.encode(audio_padded)
    
    B, K, T_frames = codes.shape
    n_q_semantic = getattr(model.quantizer, 'n_q_semantic', 1)
    
    results = {}
    
    # 1. Semantic only (codebook 0)
    codes_semantic = codes.clone()
    codes_semantic[:, n_q_semantic:, :] = 0  # Zero out non-semantic
    with torch.no_grad():
        audio_semantic = model.decode(codes_semantic[:, :n_q_semantic, :])
    audio_semantic = audio_semantic[..., :T_samples]
    results['sisdr_semantic_only'] = si_sdr(audio_semantic, audio).item()
    
    # 2. Semantic + Prosody (codebooks 0+1, assuming n_q_semantic covers first group)
    # The SplitResidualVectorQuantizer uses rvq_first for semantic+prosody
    n_q_first = getattr(model.quantizer, 'n_q_semantic', 2)  # Usually 2 for semantic+prosody
    if n_q_first < K:
        codes_sem_pros = codes[:, :n_q_first, :]
        with torch.no_grad():
            audio_sem_pros = model.decode(codes_sem_pros)
        audio_sem_pros = audio_sem_pros[..., :T_samples]
        results['sisdr_semantic_prosody'] = si_sdr(audio_sem_pros, audio).item()
    else:
        results['sisdr_semantic_prosody'] = results['sisdr_semantic_only']
    
    # 3. All codebooks
    with torch.no_grad():
        audio_all = model.decode(codes)
    audio_all = audio_all[..., :T_samples]
    results['sisdr_all'] = si_sdr(audio_all, audio).item()
    
    return ReconstructionAblationResult(**results)


# =============================================================================
# Codebook Health Metrics
# =============================================================================

@dataclass
class CodebookHealthMetrics:
    """Codebook health metrics."""
    entropy: Dict[int, float]  # Per-codebook entropy (optimal=1.0)
    usage_ratio: Dict[int, float]  # Per-codebook usage ratio
    avg_entropy: float
    avg_usage: float


def extract_codebook_metrics(quantizer: nn.Module) -> CodebookHealthMetrics:
    """
    Extract codebook health metrics from quantizer.
    
    Args:
        quantizer: SplitResidualVectorQuantizer or ResidualVectorQuantizer
        
    Returns:
        CodebookHealthMetrics with entropy and usage per codebook
    """
    entropy = {}
    usage_ratio = {}
    
    # Handle SplitResidualVectorQuantizer
    if hasattr(quantizer, 'rvq_first') and hasattr(quantizer, 'rvq_rest'):
        rvqs = [('first', quantizer.rvq_first)]
        if quantizer.rvq_rest is not None:
            rvqs.append(('rest', quantizer.rvq_rest))
    else:
        rvqs = [('main', quantizer)]
    
    codebook_idx = 0
    for name, rvq in rvqs:
        if hasattr(rvq, 'layers'):
            for layer in rvq.layers:
                if hasattr(layer, '_codebook'):
                    codebook = layer._codebook
                    usage = codebook.cluster_usage.detach().cpu()
                    
                    # Compute normalized entropy
                    proba = usage / usage.sum()
                    log_proba = torch.where(
                        proba == 0,
                        torch.zeros_like(proba),
                        proba * torch.log(proba)
                    )
                    ent = -log_proba.sum()
                    normalized_entropy = (ent / math.log(codebook.codebook_size)).item()
                    
                    # Compute usage ratio (fraction of codes used above threshold)
                    threshold = usage.sum() / codebook.codebook_size * 0.1
                    used_codes = (usage > threshold).float().mean().item()
                    
                    entropy[codebook_idx] = normalized_entropy
                    usage_ratio[codebook_idx] = used_codes
                    codebook_idx += 1
    
    avg_entropy = sum(entropy.values()) / len(entropy) if entropy else 0.0
    avg_usage = sum(usage_ratio.values()) / len(usage_ratio) if usage_ratio else 0.0
    
    return CodebookHealthMetrics(
        entropy=entropy,
        usage_ratio=usage_ratio,
        avg_entropy=avg_entropy,
        avg_usage=avg_usage,
    )


# =============================================================================
# Probing Classifiers
# =============================================================================

class LinearProbe(nn.Module):
    """Simple linear probe for classification/regression."""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ProbingEvaluator:
    """
    Evaluator for probing classifiers.
    
    Uses kylelovesllms/timit_asr dataset for phoneme probing
    and torchaudio pitch extraction for pitch probing.
    """
    
    def __init__(
        self,
        model_dim: int,
        num_phonemes: int = 61,  # TIMIT has 61 phonemes
        device: str = "cuda",
    ):
        self.model_dim = model_dim
        self.num_phonemes = num_phonemes
        self.device = device
        
        # Probing classifiers
        self.phoneme_probe: Optional[LinearProbe] = None
        self.pitch_probe: Optional[LinearProbe] = None
        
        # Dataset cache
        self._dataset_loaded = False
        self._train_data = None
        self._test_data = None
    
    def load_timit_dataset(self):
        """Load TIMIT dataset from HuggingFace."""
        try:
            from datasets import load_dataset
            # Use kylelovesllms/timit_asr as requested
            dataset = load_dataset("kylelovesllms/timit_asr")
            self._train_data = dataset["train"]
            self._test_data = dataset["test"]
            self._dataset_loaded = True
            
            # Map phones to IDs
            all_phones = set()
            for x in self._train_data["phonetic_detail"]:
                for p in x["utterance"]:
                    all_phones.add(p)
            self.phone_map = {p: i for i, p in enumerate(sorted(list(all_phones)))}
            self.num_phonemes = len(self.phone_map)
            
            return True
        except Exception as e:
            print(f"Warning: Could not load TIMIT dataset: {e}")
            return False
    
    def get_probing_batch(
        self, 
        dataset_split: str, 
        batch_size: int = 16, 
        sample_rate: int = 24000,
        fps: float = 12.5
    ):
        """Yield batches of (audio, phonemes, pitch) from TIMIT."""
        if not self._dataset_loaded:
            if not self.load_timit_dataset():
                return
        
        data = self._train_data if dataset_split == "train" else self._test_data
        
        # Simple generator for now
        for i in range(0, len(data), batch_size):
            batch_slice = data[i:i+batch_size]
            
            audio_tensors = []
            phoneme_labels = []
            pitch_targets = []
            
            for j in range(len(batch_slice["audio"])):
                # 1. Process Audio
                arr = torch.from_numpy(batch_slice["audio"][j]["array"]).float()
                sr = batch_slice["audio"][j]["sampling_rate"]
                if sr != sample_rate:
                    resampler = torchaudio.transforms.Resample(sr, sample_rate)
                    arr = resampler(arr)
                
                # Make it [1, T]
                if arr.dim() == 1:
                    arr = arr.unsqueeze(0)
                
                # 2. Extract Phonemes per frame
                # TIMIT phonetic_detail: {'start': [], 'stop': [], 'utterance': []}
                # Times are in samples at original SR
                detail = batch_slice["phonetic_detail"][j]
                num_frames = int(math.ceil(arr.shape[-1] / sample_rate * fps))
                frame_duration = 1.0 / fps
                
                p_labels = torch.zeros(num_frames, dtype=torch.long)
                for start, stop, phone in zip(detail["start"], detail["stop"], detail["utterance"]):
                    # Convert sample indices to frame indices
                    start_f = int(start / sr * fps)
                    stop_f = int(stop / sr * fps)
                    p_id = self.phone_map.get(phone, 0)
                    p_labels[start_f:stop_f] = p_id
                
                # 3. Extract Pitch per frame
                pitch = torchaudio.functional.detect_pitch_frequency(
                    arr, sample_rate, frame_time=frame_duration
                ).squeeze(0)
                
                # Align lengths (detect_pitch might return slightly different length)
                min_f = min(num_frames, pitch.shape[0])
                audio_tensors.append(arr)
                phoneme_labels.append(p_labels[:min_f])
                pitch_targets.append(pitch[:min_f])
            
            # Pad and stack
            # For simplicity in probing, we might just concatenate all frames
            yield audio_tensors, phoneme_labels, pitch_targets

    def run_probing_eval(
        self,
        model: nn.Module,
        encoder_fn,  # lambda audio: model.encode_latent(audio)
        max_batches: int = 10,
    ) -> Dict[str, float]:
        """
        Train and evaluate probes on the provided representations.
        
        Args:
            model: The main model (to access device/etc)
            encoder_fn: Function to extract the specific representation (e.g. Head 0)
            max_batches: How many batches to use for training probes
        """
        device = next(model.parameters()).device
        
        # We'll use two probes: one for phoneme (classification), one for pitch (regression)
        phoneme_probe = LinearProbe(self.model_dim, self.num_phonemes).to(device)
        pitch_probe = LinearProbe(self.model_dim, 1).to(device)
        
        p_optimizer = torch.optim.Adam(phoneme_probe.parameters(), lr=1e-3)
        f_optimizer = torch.optim.Adam(pitch_probe.parameters(), lr=1e-3)
        
        # 1. Training Phase
        phoneme_probe.train()
        pitch_probe.train()
        
        batch_gen = self.get_probing_batch("train", batch_size=4)
        for i, (audios, p_labels, p_targets) in enumerate(batch_gen):
            if i >= max_batches: break
            
            for audio, labels, targets in zip(audios, p_labels, p_targets):
                audio = audio.unsqueeze(0).to(device)
                labels = labels.to(device)
                targets = targets.to(device)
                
                with torch.no_grad():
                    # Extract representation [1, D, T]
                    rep = encoder_fn(audio)
                    rep = rep.transpose(1, 2).squeeze(0) # [T, D]
                
                # Match lengths
                min_t = min(rep.shape[0], labels.shape[0], targets.shape[0])
                rep, labels, targets = rep[:min_t], labels[:min_t], targets[:min_t]
                
                # Train Phoneme Probe
                p_logits = phoneme_probe(rep)
                p_loss = F.cross_entropy(p_logits, labels)
                p_optimizer.zero_grad()
                p_loss.backward()
                p_optimizer.step()
                
                # Train Pitch Probe
                f_pred = pitch_probe(rep).squeeze(-1)
                f_loss = F.mse_loss(f_pred, targets)
                f_optimizer.zero_grad()
                f_loss.backward()
                f_optimizer.step()

        # 2. Evaluation Phase
        phoneme_probe.eval()
        pitch_probe.eval()
        
        correct = 0
        total = 0
        pitch_errors = []
        
        test_gen = self.get_probing_batch("test", batch_size=4)
        for i, (audios, p_labels, p_targets) in enumerate(test_gen):
            if i >= 5: break # Evaluate on 5 batches
            
            for audio, labels, targets in zip(audios, p_labels, p_targets):
                audio = audio.unsqueeze(0).to(device)
                labels = labels.to(device)
                targets = targets.to(device)
                
                with torch.no_grad():
                    rep = encoder_fn(audio).transpose(1, 2).squeeze(0)
                    min_t = min(rep.shape[0], labels.shape[0], targets.shape[0])
                    rep, labels, targets = rep[:min_t], labels[:min_t], targets[:min_t]
                    
                    p_logits = phoneme_probe(rep)
                    preds = p_logits.argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
                    
                    f_pred = pitch_probe(rep).squeeze(-1)
                    pitch_errors.append(F.l1_loss(f_pred, targets, reduction='none'))
        
        avg_acc = correct / total if total > 0 else 0.0
        avg_pitch_mae = torch.cat(pitch_errors).mean().item() if pitch_errors else 0.0
        
        return {
            "phoneme_accuracy": avg_acc,
            "pitch_mae": avg_pitch_mae
        }
