"""
Trainer for the compressor model (Final Consolidated Version).
"""

from dataclasses import dataclass, field, asdict

import lightning as L
import torch
import torch.nn.functional as F
import wandb
import torchaudio
import os
from torch import nn

from models.mimi.modeling_mimi import get_mimi
from models.mimi.configuration_mimi import DEFAULT_MIMI_CONFIG, MimiConfig
from trainer.compressor_data_loader import BatchInputData
from trainer.mstftd import MultiScaleSTFTDiscriminator
from trainer.specloss import MultiScaleMelSpectrogramLoss
from trainer.adversarial_losses import (
    AdversarialLoss,
    FeatureMatchingLoss,
    get_adv_criterion,
    get_fake_criterion,
    get_real_criterion,
)
from trainer.evaluation import (
    reconstruction_ablation,
    extract_codebook_metrics,
    si_sdr,
    ProbingEvaluator,
)


@dataclass
class CompressorTrainerConfig:
    orignal_filename: str
    device: str
    num_codebooks: int

    # Dimensions for distillation
    wavlm_dim: int
    llm_dim: int

    # Training flags
    adversarial_only: bool

    # Loss Weights
    alpha_adv: float
    alpha_feat: float
    alpha_msspec: float
    alpha_wavlm: float
    alpha_llm: float

    lr_g: float
    lr_d: float

    mimi_config: MimiConfig = field(default_factory=lambda: DEFAULT_MIMI_CONFIG)


class CompressorTrainer(L.LightningModule):
    def __init__(self, config: CompressorTrainerConfig):
        super().__init__()
        self.config = config
        self.automatic_optimization = False  # Manual optimization for GAN
        self.save_hyperparameters(asdict(config))

        # 1. Generator (Mimi)
        self.model = get_mimi(
            config.orignal_filename,
            config.mimi_config,
            config.device,
            config.num_codebooks,
        )
        self.model.train()

        # # Freeze encoder as standard practice for this distillation phase
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        if self.model.encoder_transformer is not None:
            for param in self.model.encoder_transformer.parameters():
                param.requires_grad = False

        self.frame_rate = self.config.mimi_config.frame_rate

        # 2. Discriminator (Multi-Scale STFT)
        self.discriminator = MultiScaleSTFTDiscriminator(
            filters=32,
            norm="weight_norm",
            n_ffts=[1024, 2048, 512, 256, 128],
            hop_lengths=[256, 512, 128, 64, 32],
            win_lengths=[1024, 2048, 512, 256, 128],
            activation="LeakyReLU",
            activation_params={"negative_slope": 0.3},
        )

        # 3. Distillation Projections
        # Using Conv1d with larger kernel to allow more temporal adaptation (Kernel 5)
        # Increased capacity with wider hidden dimension
        self.wavlm_proj = nn.Sequential(
            nn.Conv1d(self.model.dimension, config.wavlm_dim * 2, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(config.wavlm_dim * 2, config.wavlm_dim, kernel_size=1),
        )

        # LLM Projection (Distill Semantic)
        if hasattr(config, "llm_dim") and config.llm_dim > 0:
            self.llm_proj = nn.Sequential(
                nn.Conv1d(
                    self.model.dimension, config.llm_dim * 2, kernel_size=5, padding=2
                ),
                nn.GELU(),
                nn.Conv1d(config.llm_dim * 2, config.llm_dim, kernel_size=1),
            )
        else:
            self.llm_proj = None

        # 4. Losses
        if not config.adversarial_only:
            self.msspec_loss = MultiScaleMelSpectrogramLoss(
                sample_rate=self.config.mimi_config.sample_rate,
                range_start=6,
                range_end=11,
                n_mels=64,
                f_min=64,
                normalized=True,
                alphas=False,
            )

        # Adversarial Loss Wrapper
        self.adv_loss_wrapper = AdversarialLoss(
            adversary=self.discriminator,
            optimizer=None,  # Will be set in configure_optimizers
            loss=get_adv_criterion("hinge"),
            loss_real=get_real_criterion("hinge"),
            loss_fake=get_fake_criterion("hinge"),
            loss_feat=FeatureMatchingLoss(),
            normalize=True,
            gradient_clip_val=1.0,
        )

        # 5. Probing Evaluator
        self.probing_evaluator = ProbingEvaluator(
            model_dim=self.model.dimension,
            device=config.device
        )

    def configure_optimizers(self):
        # 1. Generator Optimizer (Mimi Model only - Decoder/Quantizer)
        # GAN standard betas for stability
        opt_g = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.config.lr_g, 
            betas=(0.5, 0.9)
        )

        # 2. Discriminator Optimizer
        # GAN standard betas
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), 
            lr=self.config.lr_d, 
            betas=(0.5, 0.9)
        )

        # 3. Projection Optimizer ("The Rest" - Distillation Projections)
        # Default betas for regression/feature matching tasks
        proj_params = list(self.wavlm_proj.parameters())
        if self.llm_proj is not None:
            proj_params += list(self.llm_proj.parameters())
            
        opt_proj = torch.optim.AdamW(
            proj_params, 
            lr=self.config.lr_g, # Use same LR as generator for now
            betas=(0.9, 0.999)
        )

        # Inject optimizer into AdversarialLoss (needs D optimizer)
        self.adv_loss_wrapper.optimizer = opt_d

        return [opt_g, opt_d, opt_proj]

    def training_step(self, batch: BatchInputData, batch_idx):
        opt_g, opt_d, opt_proj = self.optimizers()

        # Unpack batch
        # audio: (B, 1, T_samples)
        # wavlm_feat: (B, T_frames, D_wavlm)
        audio = batch.audio
        wavlm_feat = batch.wavlm_feat
        llm_feat = batch.llm_feat

        # --- Forward Pass (Generator) ---

        # 1. Encode
        emb = self.model._encode_to_unquantized_latent(audio)

        # 2. Split RVQ Quantization
        quantizer = self.model.quantizer

        # Semantic + Prosody (First RVQ group)
        sem_pros_res = quantizer.rvq_first(emb, self.frame_rate)

        # Acoustic Residuals (Rest of codebooks)
        if quantizer.rvq_rest is not None:
            ac_res = quantizer.rvq_rest(emb, self.frame_rate)
        else:
            ac_res = None

        # Reconstruct Total Quantized Latent for Audio Decoder
        emb_quant = sem_pros_res.x
        if ac_res is not None:
            emb_quant = emb_quant + ac_res.x

        # 3. Decode (Audio Path - Uses RAW magnitudes)
        emb_dec = self.model._to_encoder_framerate(emb_quant)
        if self.model.decoder_transformer is not None:
            state = self.model._streaming_state
            if state is None:
                (emb_dec,) = self.model.decoder_transformer(emb_dec)
            else:
                (emb_dec,) = state.graphed_tr_dec(emb_dec)

        out_audio = self.model.decoder(emb_dec)

        # Trim output to match input length
        if out_audio.shape[-1] >= audio.shape[-1]:
            out_audio = out_audio[..., : audio.shape[-1]]

        # --- Generator Optimization ---
        # Note: We technically don't need to toggle for manual backward per se, 
        # but it's good practice for Lightning hooks. 
        # We'll consider opt_g as the primary context.
        # self.toggle_optimizer(opt_g) # Implicitly handled if needed, or just run manually.

        # A. Adversarial Loss
        loss_adv, loss_feat = self.adv_loss_wrapper(out_audio, audio)

        # B. Distillation Losses (Latent Path - Uses NORMALIZED magnitudes)

        # WavLM Distillation targets the first RVQ group (Semantic + Prosody)
        wavlm_feat_norm = F.layer_norm(wavlm_feat, (wavlm_feat.shape[-1],))

        sem_pros_latent_raw = sem_pros_res.x.transpose(1, 2)  # (B, T, D)
        sem_pros_norm = F.layer_norm(
            sem_pros_latent_raw, (sem_pros_latent_raw.shape[-1],)
        )

        # Transpose for Conv1d: (B, T, D) -> (B, D, T)
        sem_pros_norm_t = sem_pros_norm.transpose(1, 2)
        # Project and transpose back: (B, D_wavlm, T) -> (B, T, D_wavlm)
        sem_pros_wavlm_proj = self.wavlm_proj(sem_pros_norm_t).transpose(1, 2)

        # Ensure shapes match before cosine similarity (handle potential padding differences)
        T_wavlm = wavlm_feat_norm.shape[1]
        T_proj = sem_pros_wavlm_proj.shape[1]
        
        if T_wavlm != T_proj:
             min_t = min(T_wavlm, T_proj)
             wavlm_feat_norm = wavlm_feat_norm[:, :min_t, :]
             sem_pros_wavlm_proj = sem_pros_wavlm_proj[:, :min_t, :]

        loss_wavlm = (
            1.0
            - F.cosine_similarity(sem_pros_wavlm_proj, wavlm_feat_norm, dim=-1).mean()
        )

        # LLM Distillation (Targets Semantic Only)
        loss_llm = torch.tensor(0.0, device=self.device)
        if self.llm_proj is not None:
             # Extract codes for Semantic layer (index 0)
             # sem_pros_res.codes is shape [B, K, T]
             # We assume semantic is always the first codebook
             sem_codes = sem_pros_res.codes[:, : quantizer.n_q_semantic, :]
             
             # Decode separated semantic latent
             # [B, T, D] <- [B, K, T]  (Decode expects [B, K, T])
             sem_emb = quantizer.rvq_first.decode(sem_codes)
             
             # Prepare for projection
             sem_emb_t = sem_emb.transpose(1, 2) # (B, D, T) -> (B, T, D) for layer norm
             sem_emb_norm = F.layer_norm(sem_emb_t, (sem_emb_t.shape[-1],)).transpose(1, 2) # Back to (B, D, T)
             
             # Project
             sem_llm_proj = self.llm_proj(sem_emb_norm).transpose(1, 2) # (B, T, D_llm)
             
             llm_feat_norm = F.layer_norm(llm_feat, (llm_feat.shape[-1],))
             
             # Align shapes
             T_llm = llm_feat_norm.shape[1]
             T_proj_llm = sem_llm_proj.shape[1]
             
             if T_llm != T_proj_llm:
                  min_t_llm = min(T_llm, T_proj_llm)
                  llm_feat_norm = llm_feat_norm[:, :min_t_llm, :]
                  sem_llm_proj = sem_llm_proj[:, :min_t_llm, :]
                  
             loss_llm = (
                 1.0
                 - F.cosine_similarity(sem_llm_proj, llm_feat_norm, dim=-1).mean()
            )

        # C. Reconstruction Loss
        loss_msspec = torch.tensor(0.0, device=self.device)
        if not self.config.adversarial_only:
            loss_msspec = self.msspec_loss(out_audio, audio)

        # Total Generator Loss
        total_loss_g = (
            self.config.alpha_adv * loss_adv
            + self.config.alpha_feat * loss_feat
            + self.config.alpha_wavlm * loss_wavlm
            + self.config.alpha_msspec * loss_msspec
        )
        
        if self.llm_proj is not None:
             total_loss_g += self.config.alpha_llm * loss_llm

        self.manual_backward(total_loss_g)
        
        # Optimize Generator (Mimi)
        self.clip_gradients(
            opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_g.step()
        opt_g.zero_grad()
        
        # Optimize Projections
        self.clip_gradients(
            opt_proj, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_proj.step()
        opt_proj.zero_grad()


        out_audio_detached = out_audio.detach()
        loss_d = self.adv_loss_wrapper.train_adv(out_audio_detached, audio)

        self.untoggle_optimizer(opt_d)

        # --- Logging ---
        logs = {
                "train/loss_g": total_loss_g,
                "train/loss_d": loss_d,
                "train/adv_g": loss_adv,
                "train/feat": loss_feat,
                "train/wavlm": loss_wavlm,
                "train/msspec": loss_msspec,
            }
            
        if self.llm_proj is not None:
            logs["train/llm"] = loss_llm
            
        self.log_dict(
            logs,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

    def validation_step(self, batch: BatchInputData, batch_idx):
        # Only run on the first batch to act as a "single file" test
        if batch_idx > 0:
            return

        # Target audio path
        target_audio_path = "data/audio/POD1000000018_S0000269.wav"

        if os.path.exists(target_audio_path):
            # Load specific audio file
            audio, sr = torchaudio.load(target_audio_path)
            
            # Resample if needed
            if sr != self.config.mimi_config.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.config.mimi_config.sample_rate)
                audio = resampler(audio)
            
            # Ensure it is (1, 1, T)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0).unsqueeze(0)
            elif audio.dim() == 2:
                audio = audio.unsqueeze(0)
            
            # Move to device
            audio = audio.to(self.device)
        else:
            # Fallback to batch audio if target not found
            print(f"Warning: {target_audio_path} not found. Using batch audio for validation.")
            audio = batch.audio
            if audio.dim() == 2:
                audio = audio.unsqueeze(0)
        
        # 1. Encode to Latent
        # Pad for encoding
        frame_size = self.model.frame_size
        _, _, T_samples = audio.shape
        
        if T_samples % frame_size != 0:
            pad_len = frame_size - (T_samples % frame_size)
            audio_padded = F.pad(audio, (0, pad_len))
        else:
            audio_padded = audio

        # Encode
        latent = self.model.encode(audio_padded)
        
        # Decode
        generated_audio = self.model.decode(latent)
        
        # Trim back to original length
        if generated_audio.shape[-1] > T_samples:
            generated_audio = generated_audio[..., :T_samples]

        # === Stage 1: Intrinsic Evaluation ===
        
        # 1. Reconstruction Ablation (SI-SDR per codebook group)
        ablation_results = reconstruction_ablation(self.model, audio)
        
        # 2. Codebook Health Metrics
        codebook_metrics = extract_codebook_metrics(self.model.quantizer)
        
        # 3. Overall SI-SDR for full reconstruction
        overall_sisdr = si_sdr(generated_audio, audio)
        
        # 4. Probing Evaluation (Phoneme vs Pitch on each head)
        # We only run this on the first batch to avoid too much overhead
        # Probing Head 0 (Semantic)
        head_0_results = self.probing_evaluator.run_probing_eval(
            self.model,
            encoder_fn=lambda x: self.model.quantizer.rvq_first.decode(self.model.encode(x)[:, :1, :]),
            max_batches=5
        )
        
        # Probing Head 1 (Prosody)
        head_1_results = self.probing_evaluator.run_probing_eval(
            self.model,
            encoder_fn=lambda x: self.model.quantizer.rvq_first.decode(self.model.encode(x)[:, 1:2, :]),
            max_batches=5
        )
        
        # Build metrics dict for logging
        eval_metrics = {
            "val/sisdr_semantic_only": ablation_results.sisdr_semantic_only,
            "val/sisdr_semantic_prosody": ablation_results.sisdr_semantic_prosody,
            "val/sisdr_all": ablation_results.sisdr_all,
            "val/sisdr_overall": overall_sisdr.item(),
            "val/codebook_entropy_avg": codebook_metrics.avg_entropy,
            "val/codebook_usage_avg": codebook_metrics.avg_usage,
            # Head 0 Probing
            "val/probe_head0_phone_acc": head_0_results["phoneme_accuracy"],
            "val/probe_head0_pitch_mae": head_0_results["pitch_mae"],
            # Head 1 Probing
            "val/probe_head1_phone_acc": head_1_results["phoneme_accuracy"],
            "val/probe_head1_pitch_mae": head_1_results["pitch_mae"],
        }
        
        # Add per-codebook metrics
        for idx, entropy in codebook_metrics.entropy.items():
            eval_metrics[f"val/entropy_{idx}"] = entropy
        for idx, usage in codebook_metrics.usage_ratio.items():
            eval_metrics[f"val/usage_{idx}"] = usage
        
        # Log metrics
        self.log_dict(eval_metrics, prog_bar=False, logger=True)

        # Log audio samples
        self._log_audio_samples(audio, generated_audio, num_samples=1)

    def _log_audio_samples(self, real_audio, generated_audio, num_samples=5):
        num_samples = min(num_samples, real_audio.size(0))
        original_audios = []
        generated_audios = []

        for i in range(num_samples):
            real_sample = real_audio[i].squeeze().cpu().numpy()
            gen_sample = generated_audio[i].squeeze().cpu().numpy()

            original_audios.append(
                wandb.Audio(
                    real_sample,
                    sample_rate=self.config.mimi_config.sample_rate,
                    caption=f"Original {i}",
                )
            )
            generated_audios.append(
                wandb.Audio(
                    gen_sample,
                    sample_rate=self.config.mimi_config.sample_rate,
                    caption=f"Generated {i}",
                )
            )

        wandb.log(
            {
                "val/original_audio": original_audios,
                "val/generated_audio": generated_audios,
            }
        )
