import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import numpy as np
import random
import torch.utils.data
import lightning
from typing import cast

from brain_image.trainer import NICETrainerConfig, NICETrainer
from brain_image.model import NICEConfig, EEGEncoderConfig, NICEModel
from brain_image.data import EEGDatasetConfig


@pytest.fixture
def mock_nice_trainer(mock_data_directory):
    data_dir = mock_data_directory["data_dir"]
    log_dir = mock_data_directory["log_dir"]
    checkpoint_dir = mock_data_directory["checkpoint_dir"]
    config = NICEConfig(
        align_target_model="aligned_synclr_16",
        max_epochs=1,
    )
    dataset_config = EEGDatasetConfig(
        data_path=data_dir,
        batch_size=2,
        val_batch_size=2,
        subs=[1],
        num_workers=0,
    )
    trainer_config = NICETrainerConfig(
        run_name="test_nice",
        num_epochs=1,
        save_checkpoints=True,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        enable_barebones=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer = NICETrainer(
        config=trainer_config,
        model_config=config,
        dataset_config=dataset_config,
    )
    return {
        "trainer": trainer,
        "trainer_config": trainer_config,
        "nice_config": config,
        "dataset_config": dataset_config,
    }


def test_nice_trainer_runs_and_leaves_logs_and_checkpoints(mock_data_directory):
    data_dir = mock_data_directory["data_dir"]
    log_dir = mock_data_directory["log_dir"]
    checkpoint_dir = mock_data_directory["checkpoint_dir"]
    config = NICEConfig(
        align_target_model="aligned_synclr_16",
        max_epochs=1,
    )
    dataset_config = EEGDatasetConfig(
        data_path=data_dir,
        batch_size=2,
        val_batch_size=2,
        subs=[1],
        num_workers=0,
    )
    trainer_config = NICETrainerConfig(
        run_name="test_nice",
        num_epochs=1,
        save_checkpoints=True,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        enable_barebones=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer = NICETrainer(
        config=trainer_config,
        model_config=config,
        dataset_config=dataset_config,
    )

    trainer.train()
    if trainer.config.checkpoint_dir is not None:
        assert any(trainer.config.checkpoint_dir.glob("*.ckpt")) or any(
            trainer.config.checkpoint_dir.glob("*.pt")
        )
    assert trainer.config.log_dir.exists() and any(trainer.config.log_dir.iterdir())


def test_nice_trainer_loss_decreases(mock_data_directory):
    """Test that NICETrainer reduces loss over a few epochs on a small batch."""
    lightning.seed_everything(42)
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Initial model
    data_dir = mock_data_directory["data_dir"]
    log_dir = mock_data_directory["log_dir"]
    checkpoint_dir = mock_data_directory["checkpoint_dir"]
    config = NICEConfig(
        align_target_model="aligned_synclr_16",
        lr_scheduler="none",
        projector_warmup_epochs=0,
        encoder_warmup_epochs=0,
    )
    trainer_config = NICETrainerConfig(
        run_name="test_nice",
        num_epochs=3,
        save_checkpoints=True,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        enable_barebones=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        overfit_batches=1,
    )

    trainer = NICETrainer(
        config=trainer_config,
        model_config=config,
        dataset_config=EEGDatasetConfig(
            data_path=data_dir,
            batch_size=2,
            val_batch_size=2,
            subs=[1],
            num_workers=0,
            shuffle_train=False,
        ),
    )
    trainer = cast(NICETrainer, trainer)
    trainer.model = cast(NICEModel, trainer.model)

    train_dataloader = trainer.model.data_module.train_dataloader()
    batch = next(iter(train_dataloader))
    img_latent = batch["img_latent"].to(trainer.model.device, dtype=trainer.model.dtype)
    eeg_data = batch["eeg_data"].to(trainer.model.device, dtype=trainer.model.dtype)

    with torch.no_grad():
        sim = trainer.model.get_similarity(img_latent, eeg_data)
        initial_loss = trainer.model.get_loss(sim).item()

    # Train model
    trainer = NICETrainer(
        config=trainer_config,
        model_config=config,
        dataset_config=EEGDatasetConfig(
            data_path=data_dir,
            batch_size=2,
            val_batch_size=2,
            subs=[1],
            num_workers=0,
            shuffle_train=False,
        ),
    )

    # Disable validation by patching the trainer creation
    trainer.train()

    # Final model
    with torch.no_grad():
        sim = trainer.model.get_similarity(img_latent, eeg_data)
        final_loss = trainer.model.get_loss(sim).item()  # type: ignore

    assert final_loss < initial_loss

    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")
