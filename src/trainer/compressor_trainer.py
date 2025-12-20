"""
Trainer for the compressor model.
"""

from dataclasses import dataclass, field

import lightning as L
import torch
import torch.nn.functional as F
from torch import nn

from models.mimi.modeling_mimi import get_mimi_with_prosody_from_original_mimi_weights
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


@dataclass
class CompressorTrainerConfig:
    orignal_filename: str
    mimi_config: MimiConfig = field(default_factory=lambda: DEFAULT_MIMI_CONFIG)
    device: str = "cuda"
    num_codebooks: int = 8

    # Dimensions for distillation
    wavlm_dim: int = 1024
    llm_dim: int = 1024

    # Training flags
    adversarial_only: bool = False

    # Loss Weights
    alpha_adv: float = 4.0
    alpha_feat: float = 4.0
    alpha_msspec: float = 2.0
    alpha_wavlm: float = 1.0
    alpha_llm: float = 1.0


class CompressorTrainer(L.LightningModule):
    def __init__(self, config: CompressorTrainerConfig):
        super().__init__()
        self.config = config
        self.automatic_optimization = False  # Manual optimization for GAN

        # 1. Generator (Mimi)
        self.model = get_mimi_with_prosody_from_original_mimi_weights(
            config.orignal_filename,
            config.mimi_config,
            config.device,
            config.num_codebooks,
        )
        self.model.train()
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
        # Project from Mimi latent dimension to WavLM/LLM dimensions
        self.wavlm_proj = nn.Linear(self.model.dimension, config.wavlm_dim)
        self.llm_proj = nn.Linear(self.model.dimension, config.llm_dim)

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
        # This handles the logic for G and D losses properly
        self.adv_loss_wrapper = AdversarialLoss(
            adversary=self.discriminator,
            optimizer=None,  # Will be set in configure_optimizers
            loss=get_adv_criterion("hinge"),
            loss_real=get_real_criterion("hinge"),
            loss_fake=get_fake_criterion("hinge"),
            loss_feat=FeatureMatchingLoss(),
            normalize=True,
        )

    def configure_optimizers(self):
        # Generator Optimizer (Mimi + Projections)
        g_params = (
            list(self.model.parameters())
            + list(self.wavlm_proj.parameters())
            + list(self.llm_proj.parameters())
        )
        opt_g = torch.optim.Adam(g_params, lr=3e-4, betas=(0.5, 0.9))

        # Discriminator Optimizer
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=3e-4, betas=(0.5, 0.9)
        )
        
        # Inject optimizer into AdversarialLoss
        self.adv_loss_wrapper.optimizer = opt_d

        return [opt_g, opt_d]

    def training_step(self, batch: BatchInputData, batch_idx):
        opt_g, opt_d = self.optimizers()

        # Unpack batch
        # audio: (B, 1, T_samples)
        # wavlm_feat: (B, T_frames, D_wavlm)
        # llm_feat: (B, T_frames, D_llm)
        audio = batch.audio
        wavlm_feat = batch.wavlm_feat
        llm_feat = batch.llm_feat

        # Ensure audio requires grad for some implementations, but generally input doesn't need it.
        # Forward Pass Generator

        # 1. Encode
        # emb: (B, Dim, T_frames)
        emb = self.model._encode_to_unquantized_latent(audio)

        # 2. Split RVQ Quantization
        # We need to access the specific quantizer structure: SplitResidualVectorQuantizerWithProsody
        quantizer = self.model.quantizer

        # Semantic
        sem_res = quantizer.rvq_first(emb, self.frame_rate)
        # Prosody
        pros_res = quantizer.rvq_prosody(emb, self.frame_rate)
        # Acoustic (Rest)
        if quantizer.rvq_rest is not None:
            ac_res = quantizer.rvq_rest(emb, self.frame_rate)
        else:
            ac_res = None

        # Reconstruct Quantized Latent (Sum)
        emb_quant = sem_res.x + pros_res.x
        if ac_res is not None:
            emb_quant = emb_quant + ac_res.x

        # 3. Decode
        emb_dec = self.model._to_encoder_framerate(emb_quant)
        if self.model.decoder_transformer is not None:
            state = self.model._streaming_state
            if state is None:
                (emb_dec,) = self.model.decoder_transformer(emb_dec)
            else:
                # Training usually doesn't use streaming state
                (emb_dec,) = state.graphed_tr_dec(emb_dec)

        out_audio = self.model.decoder(emb_dec)

        # Trim output to match input length (handling padding)
        if out_audio.shape[-1] >= audio.shape[-1]:
            out_audio = out_audio[..., : audio.shape[-1]]

        # --- Generator Optimization ---
        # Toggle optimizer
        self.toggle_optimizer(opt_g)

        # A. Adversarial Loss (Generator Part) using AdversarialLoss wrapper
        # The wrapper computes G's adversarial loss and Feature Matching loss
        # Note: We pass `real` audio to get feature matching target
        loss_adv, loss_feat = self.adv_loss_wrapper(out_audio, audio)

        # B. Distillation Losses
        
        # 1. LLM Distillation -> Semantic Codebook
        # We want the semantic codebook to capture text/meaning, so we distill LLM features into it.
        sem_latent = sem_res.x.transpose(1, 2)  # (B, T, Dim)
        sem_llm_proj = self.llm_proj(sem_latent)
        loss_llm = (
            1.0 - F.cosine_similarity(sem_llm_proj, llm_feat, dim=-1).mean()
        )

        # 2. WavLM Distillation -> Semantic + Prosody Codebooks
        # WavLM contains both semantic and acoustic info. We want the prosody codebook
        # to capture what's missing from semantic (intonation, speaker style, etc).
        # So we distill WavLM into the sum of (Semantic + Prosody).
        sem_pros_latent = (sem_res.x + pros_res.x).transpose(1, 2) # (B, T, Dim)
        sem_pros_wavlm_proj = self.wavlm_proj(sem_pros_latent)
        loss_wavlm = (
            1.0 - F.cosine_similarity(sem_pros_wavlm_proj, wavlm_feat, dim=-1).mean()
        )

        # C. Reconstruction Loss (Optional / Not in Adv-Only)
        loss_msspec = torch.tensor(0.0, device=self.device)
        if not self.config.adversarial_only:
            loss_msspec = self.msspec_loss(out_audio, audio)

        # Total Generator Loss
        total_loss_g = (
            self.config.alpha_adv * loss_adv
            + self.config.alpha_feat * loss_feat
            + self.config.alpha_wavlm * loss_wavlm
            + self.config.alpha_llm * loss_llm
            + self.config.alpha_msspec * loss_msspec
        )

        self.manual_backward(total_loss_g)
        self.clip_gradients(opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # --- Discriminator Optimization ---
        self.toggle_optimizer(opt_d)

        # Detach generated audio so we don't backprop into generator
        out_audio_detached = out_audio.detach()

        # Train Discriminator using AdversarialLoss wrapper
        # The wrapper's train_adv method handles backward and step internally
        # Note: In manual optimization with Lightning, calling .backward() directly on loss tensor
        # or optimizer.step() inside a sub-module is fine as long as we handle toggling.
        # However, since `train_adv` calls `optimizer.step()`, we should ensure it's compatible.
        # `AdversarialLoss.train_adv` does: zero_grad -> backward -> step.
        
        # We need to compute loss_d for logging though.
        loss_d = self.adv_loss_wrapper.train_adv(out_audio_detached, audio)
        
        # We don't need to call manual_backward or step here because train_adv does it.
        # But we might want to clip gradients if needed. 
        # AdversarialLoss doesn't clip gradients.
        
        self.untoggle_optimizer(opt_d)

        # --- Logging ---
        self.log_dict(
            {
                "train/loss_g": total_loss_g,
                "train/loss_d": loss_d,
                "train/adv_g": loss_adv,
                "train/feat": loss_feat,
                "train/wavlm": loss_wavlm,
                "train/llm": loss_llm,
                "train/msspec": loss_msspec,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
