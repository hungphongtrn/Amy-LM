import os
import sys
from datetime import datetime

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

# Add src to python path to allow imports within src modules to work
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from models.mimi.configuration_mimi import DEFAULT_MIMI_CONFIG
from trainer.compressor_data_loader import CompressorDataLoader
from trainer.compressor_trainer import CompressorTrainer, CompressorTrainerConfig


class WavLMMLPProjection(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.fc2(x)
        return x


class CompressorTrainerWavLMCenteredMLP(CompressorTrainer):
    """
    Improved WavLM trainer:
    - centers both prediction and target for WavLM cosine loss
    - replaces the WavLM linear projection with a small MLP
    - keeps the improved RVQ warmup / loss weighting from Priority 2
    """

    def __init__(self, config: CompressorTrainerConfig):
        super().__init__(config)
        self.wavlm_proj = WavLMMLPProjection(
            input_dim=self.model.dimension,
            hidden_dim=self.model.dimension,
            output_dim=config.wavlm_dim,
        )

        if config.projections_only:
            for param in self.wavlm_proj.parameters():
                param.requires_grad = True
        if config.semantic_prosody_only:
            for param in self.wavlm_proj.parameters():
                param.requires_grad = True

        self.warmup_steps = 100
        self.base_rvq_lr = 5e-5
        self.proj_lr = 3e-4

    def configure_optimizers(self):
        opt_d = torch.optim.Adam(
            self.discriminator.parameters(), lr=3e-4, betas=(0.5, 0.9)
        )
        self.adv_loss_wrapper.optimizer = opt_d

        rvq_params = list(self.model.quantizer.rvq_first.parameters()) + list(
            self.model.quantizer.rvq_prosody.parameters()
        )
        proj_params = list(self.wavlm_proj.parameters()) + list(
            self.llm_proj.parameters()
        )

        opt_g = torch.optim.Adam(
            [
                {"params": rvq_params, "lr": self.base_rvq_lr, "name": "rvq"},
                {"params": proj_params, "lr": self.proj_lr, "name": "proj"},
            ],
            betas=(0.5, 0.9),
        )

        print("Semantic+Prosody optimizer with WARMUP:")
        print(
            f"  RVQ heads: lr={self.base_rvq_lr} (with {self.warmup_steps}-step warmup)"
        )
        print(f"  Projections: lr={self.proj_lr}")

        return [opt_g, opt_d]

    def on_train_batch_start(self, batch, batch_idx):
        opt_g, _ = self.optimizers()

        if self.global_step < self.warmup_steps:
            current_rvq_lr = (
                self.base_rvq_lr * (self.global_step + 1) / self.warmup_steps
            )
            for param_group in opt_g.param_groups:
                if param_group.get("name") == "rvq":
                    param_group["lr"] = current_rvq_lr

        return super().on_train_batch_start(batch, batch_idx)

    @staticmethod
    def _center_clipwise(x: torch.Tensor) -> torch.Tensor:
        return x - x.mean(dim=1, keepdim=True)

    def training_step(self, batch, batch_idx):
        import torch.nn.functional as F

        opt_g, opt_d = self.optimizers()

        audio = batch.audio
        wavlm_feat = batch.wavlm_feat
        llm_feat = batch.llm_feat

        emb = self.model._encode_to_unquantized_latent(audio)
        quantizer = self.model.quantizer

        sem_res = quantizer.rvq_first(emb, self.frame_rate)
        pros_res = quantizer.rvq_prosody(emb, self.frame_rate)
        if quantizer.rvq_rest is not None:
            ac_res = quantizer.rvq_rest(emb, self.frame_rate)
        else:
            ac_res = None

        emb_quant = sem_res.x + pros_res.x
        if ac_res is not None:
            emb_quant = emb_quant + ac_res.x

        emb_dec = self.model._to_encoder_framerate(emb_quant)
        if self.model.decoder_transformer is not None:
            state = self.model._streaming_state
            if state is None:
                (emb_dec,) = self.model.decoder_transformer(emb_dec)
            else:
                (emb_dec,) = state.graphed_tr_dec(emb_dec)

        out_audio = self.model.decoder(emb_dec)

        if out_audio.shape[-1] >= audio.shape[-1]:
            out_audio = out_audio[..., : audio.shape[-1]]

        self.toggle_optimizer(opt_g)

        loss_adv = torch.tensor(0.0, device=self.device)
        loss_feat = torch.tensor(0.0, device=self.device)
        if self.config.alpha_adv > 0 or self.config.alpha_feat > 0:
            loss_adv, loss_feat = self.adv_loss_wrapper(out_audio, audio)

        sem_latent = sem_res.x.transpose(1, 2)
        sem_llm_proj = self.llm_proj(sem_latent)
        loss_llm = 1.0 - F.cosine_similarity(sem_llm_proj, llm_feat, dim=-1).mean()

        sem_pros_latent = (sem_res.x + pros_res.x).transpose(1, 2)
        sem_pros_wavlm_proj = self.wavlm_proj(sem_pros_latent)
        wavlm_target_centered = self._center_clipwise(wavlm_feat)
        wavlm_pred_centered = self._center_clipwise(sem_pros_wavlm_proj)
        loss_wavlm = (
            1.0
            - F.cosine_similarity(
                wavlm_pred_centered, wavlm_target_centered, dim=-1
            ).mean()
        )

        loss_msspec = torch.tensor(0.0, device=self.device)
        if not self.config.adversarial_only:
            loss_msspec = self.msspec_loss(out_audio, audio)

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

        loss_d = torch.tensor(0.0, device=self.device)
        if self.config.alpha_adv > 0:
            self.toggle_optimizer(opt_d)
            loss_d = self.adv_loss_wrapper.train_adv(out_audio.detach(), audio)
            self.untoggle_optimizer(opt_d)

        current_rvq_lr = opt_g.param_groups[0]["lr"]
        self.log_dict(
            {
                "train/loss_g": total_loss_g,
                "train/loss_d": loss_d,
                "train/adv_g": loss_adv,
                "train/feat": loss_feat,
                "train/wavlm": loss_wavlm,
                "train/llm": loss_llm,
                "train/msspec": loss_msspec,
                "train/rvq_lr": current_rvq_lr,
            },
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            logger=True,
        )

    def validation_step(self, batch, batch_idx):
        import torch.nn.functional as F

        audio = batch.audio
        wavlm_feat = batch.wavlm_feat
        llm_feat = batch.llm_feat

        emb = self.model._encode_to_unquantized_latent(audio)
        quantizer = self.model.quantizer
        sem_res = quantizer.rvq_first(emb, self.frame_rate)
        pros_res = quantizer.rvq_prosody(emb, self.frame_rate)
        if quantizer.rvq_rest is not None:
            ac_res = quantizer.rvq_rest(emb, self.frame_rate)
        else:
            ac_res = None

        emb_quant = sem_res.x + pros_res.x
        if ac_res is not None:
            emb_quant = emb_quant + ac_res.x

        emb_dec = self.model._to_encoder_framerate(emb_quant)
        if self.model.decoder_transformer is not None:
            state = self.model._streaming_state
            if state is None:
                (emb_dec,) = self.model.decoder_transformer(emb_dec)
            else:
                (emb_dec,) = state.graphed_tr_dec(emb_dec)

        generated_audio = self.model.decoder(emb_dec)

        if generated_audio.shape[-1] >= audio.shape[-1]:
            generated_audio = generated_audio[..., : audio.shape[-1]]

        loss_msspec = torch.tensor(0.0, device=self.device)
        if not self.config.adversarial_only:
            loss_msspec = self.msspec_loss(generated_audio, audio)

        sem_latent = sem_res.x.transpose(1, 2)
        sem_llm_proj = self.llm_proj(sem_latent)
        loss_llm = 1.0 - F.cosine_similarity(sem_llm_proj, llm_feat, dim=-1).mean()

        sem_pros_latent = (sem_res.x + pros_res.x).transpose(1, 2)
        sem_pros_wavlm_proj = self.wavlm_proj(sem_pros_latent)
        wavlm_target_centered = self._center_clipwise(wavlm_feat)
        wavlm_pred_centered = self._center_clipwise(sem_pros_wavlm_proj)
        loss_wavlm = (
            1.0
            - F.cosine_similarity(
                wavlm_pred_centered, wavlm_target_centered, dim=-1
            ).mean()
        )

        self.log_dict(
            {
                "val/msspec": loss_msspec,
                "val/llm": loss_llm,
                "val/wavlm": loss_wavlm,
            },
            prog_bar=True,
            on_epoch=True,
            logger=True,
        )

        if batch_idx == 0:
            self._log_audio_samples(audio, generated_audio)


def train():
    L.seed_everything(42)

    dataset_path = "data/Amy-LM-Dataset-Aligned"
    mimi_weights_path = "data/mimi_weights/tokenizer-e351c8d8-checkpoint125.safetensors"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    mimi_config = DEFAULT_MIMI_CONFIG
    trainer_config = CompressorTrainerConfig(
        orignal_filename=mimi_weights_path,
        mimi_config=mimi_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_codebooks=9,
        wavlm_dim=1024,
        llm_dim=2048,
        adversarial_only=False,
        projections_only=False,
        semantic_prosody_only=True,
        alpha_adv=0.0,
        alpha_feat=0.0,
        alpha_msspec=5.0,
        alpha_wavlm=5.0,
        alpha_llm=1.0,
    )

    data_module = CompressorDataLoader(
        data_path=dataset_path,
        batch_size=128,
        num_workers=8,
        sample_rate=int(mimi_config.sample_rate),
        fps=mimi_config.frame_rate,
        segment_duration=1.0,
        seed=42,
        persistent_workers=True,
        prefetch_factor=4,
    )

    model = CompressorTrainerWavLMCenteredMLP(config=trainer_config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/wavlm_centered_mlp",
        filename="amy-wavlm-centered-mlp-{epoch:02d}-{step:06d}",
        save_top_k=3,
        monitor="train/msspec",
        mode="min",
        every_n_epochs=1,
        save_last=True,
    )

    logger = WandbLogger(
        project="amy_wavlm_centered_mlp",
        name=f"wavlm_centered_mlp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=trainer_config,
    )
    logger.watch(model, log="gradients", log_freq=100, log_graph=True)

    trainer = L.Trainer(
        max_steps=2000,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed",
        log_every_n_steps=10,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        enable_progress_bar=True,
    )

    print("=" * 70)
    print("PRIORITY 2B: CENTERED-BOTH-SIDES + MLP WAVLM")
    print("=" * 70)
    print("Changes over the previous centered run:")
    print("  1. Center both predicted and target WavLM features per clip")
    print("  2. Replace wavlm_proj linear layer with MLP + LayerNorm")
    print("  3. Keep RVQ LR=5e-5 with 100-step warmup")
    print("  4. Keep alpha_msspec=5.0 and alpha_wavlm=5.0")
    print("=" * 70)

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    train()
