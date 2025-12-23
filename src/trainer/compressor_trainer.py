"""
Trainer for the compressor model (Final Consolidated Version).
"""

from dataclasses import dataclass, field, asdict

import lightning as L
import torch
import torch.nn.functional as F
import wandb
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

        # Freeze encoder as standard practice for this distillation phase
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
        # Using simple Linear layers as requested (relying on Normalization for stability)
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

    def configure_optimizers(self):
        # Generator Optimizer (Mimi + Projections)
        g_params = (
            list(self.model.parameters())
            + list(self.wavlm_proj.parameters())
            + list(self.llm_proj.parameters())
        )
        opt_g = torch.optim.AdamW(g_params, lr=self.config.lr_g, betas=(0.9, 0.999))

        # Discriminator Optimizer
        opt_d = torch.optim.AdamW(
            self.discriminator.parameters(), lr=self.config.lr_d, betas=(0.9, 0.999)
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
        self.toggle_optimizer(opt_g)

        # A. Adversarial Loss
        loss_adv, loss_feat = self.adv_loss_wrapper(out_audio, audio)

        # B. Distillation Losses (Latent Path - Uses NORMALIZED magnitudes)

        # --- Prepare Semantic Latents ---
        # Recover purely semantic latent (Codebook 0)
        sem_codes = sem_pros_res.codes[:, : quantizer.n_q_semantic]
        sem_latent_raw = quantizer.rvq_first.decode(sem_codes).transpose(
            1, 2
        )  # (B, T, D)

        # 1. LLM Distillation
        # Norm(Latent) -> Linear -> Cosine -> Norm(Target)
        sem_latent_norm = F.layer_norm(sem_latent_raw, (sem_latent_raw.shape[-1],))
        sem_llm_proj = self.llm_proj(sem_latent_norm)

        # Normalize LLM Target
        llm_feat_norm = F.layer_norm(llm_feat, (llm_feat.shape[-1],))

        loss_llm = 1.0 - F.cosine_similarity(sem_llm_proj, llm_feat_norm, dim=-1).mean()

        # 2. WavLM Distillation
        # This targets Semantic + Prosody, but we must block gradients to Semantic
        # to prevent fighting with loss_llm.

        # Normalize WavLM Target
        wavlm_feat_norm = F.layer_norm(wavlm_feat, (wavlm_feat.shape[-1],))

        # Reconstruct "Prosody-Only" Gradient Flow:
        # We want: Input = (Semantic_Detached + Prosody)
        # We have: sem_pros_res.x (which is Semantic + Prosody sum)

        sem_pros_latent_raw = sem_pros_res.x.transpose(1, 2)  # (B, T, D)

        # Construct the mixed latent where Semantic part is detached
        # Math: (Sem + Pros) - Sem + Sem_Detached = Pros + Sem_Detached
        mixed_latent_raw = (
            sem_pros_latent_raw - sem_latent_raw + sem_latent_raw.detach()
        )

        # Normalize the mixed input before projection
        mixed_latent_norm = F.layer_norm(
            mixed_latent_raw, (mixed_latent_raw.shape[-1],)
        )

        sem_pros_wavlm_proj = self.wavlm_proj(mixed_latent_norm)

        loss_wavlm = (
            1.0
            - F.cosine_similarity(sem_pros_wavlm_proj, wavlm_feat_norm, dim=-1).mean()
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
            + self.config.alpha_llm * loss_llm
            + self.config.alpha_msspec * loss_msspec
        )

        self.manual_backward(total_loss_g)
        self.clip_gradients(
            opt_g, gradient_clip_val=1.0, gradient_clip_algorithm="norm"
        )
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # --- Discriminator Optimization ---
        self.toggle_optimizer(opt_d)

        out_audio_detached = out_audio.detach()
        loss_d = self.adv_loss_wrapper.train_adv(out_audio_detached, audio)

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

    def validation_step(self, batch: BatchInputData, batch_idx):
        audio = batch.audio
        latent = self.model.encode(audio)
        generated_audio = self.model.decode(latent)

        if batch_idx == 0:
            self._log_audio_samples(audio, generated_audio)

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
