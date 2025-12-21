import os
import sys
from datetime import datetime

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

# Add src to python path to allow imports within src modules to work
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.trainer.compressor_trainer import CompressorTrainer, CompressorTrainerConfig
from src.trainer.compressor_data_loader import CompressorDataLoader
from src.models.mimi.configuration_mimi import DEFAULT_MIMI_CONFIG


def train():
    # 1. Configuration
    L.seed_everything(42)

    # Paths
    # Assuming the script is run from the project root
    DATASET_PATH = "data/Amy-LM-Dataset"
    MIMI_WEIGHTS_PATH = "data/mimi_weights/tokenizer-e351c8d8-checkpoint125.safetensors"

    # Check if paths exist
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    if not os.path.exists(MIMI_WEIGHTS_PATH):
        print(
            f"Warning: Mimi weights not found at {MIMI_WEIGHTS_PATH}. Trainer will attempt to download or fail."
        )
        # If not found locally, we might want to pass the repo_id if we knew it,
        # but here we rely on the local file or the behavior of the model loader if it handles paths vs ids.
        # Based on the code, if it doesn't exist, it treats it as repo_id.
        # But we want to use the local file if possible.

    # Model Configuration
    # We use the default MimiConfig, but you can modify it here if needed
    mimi_config = DEFAULT_MIMI_CONFIG

    trainer_config = CompressorTrainerConfig(
        orignal_filename=MIMI_WEIGHTS_PATH,
        mimi_config=mimi_config,
        device="cuda" if torch.cuda.is_available() else "cpu",
        num_codebooks=9,
        wavlm_dim=1024,
        llm_dim=2048,
        adversarial_only=False,  # Set to False for adaptation/warmup phase
        alpha_adv=1.0,           # Lower adversarial weight for stability
        alpha_feat=4.0,
        alpha_msspec=15.0,       # High reconstruction weight to force adaptation
        alpha_wavlm=1.0,
        alpha_llm=1.0,
    )

    # 2. Data Module
    data_module = CompressorDataLoader(
        data_path=DATASET_PATH,
        batch_size=32,  # Adjust based on GPU memory
        num_workers=4,  # Adjust based on CPU cores
        sample_rate=mimi_config.sample_rate,  # 24000
        fps=mimi_config.frame_rate,  # 12.5
        segment_duration=1.0,  # Training segment duration in seconds
        seed=42,
    )

    # 3. Model
    model = CompressorTrainer(config=trainer_config)

    # 4. Trainer
    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/compressor",
        filename="amy-compressor-{epoch:02d}"
        + datetime.now().strftime("%d_%m_%Y_%H_%M"),
        save_top_k=3,
        monitor="train/msspec",  # Monitor reconstruction loss for Stage 1
        mode="min",
        every_n_epochs=1,
    )

    # Logger
    logger = WandbLogger(project="amy_compressor", config=trainer_config)
    logger.watch(model, log="gradients", log_freq=100, log_graph=True)

    trainer = L.Trainer(
        max_epochs=100,
        accelerator="auto",
        devices="auto",  # Use all available GPUs
        precision="32",  # Mixed precision for faster training
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="step"),
        ],
        logger=logger,
        # strategy="ddp_find_unused_parameters_true", # If using DDP
        enable_progress_bar=True
    )

    # 5. Fit
    print("Starting training...")
    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    train()
