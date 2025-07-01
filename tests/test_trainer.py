import pytest
import torch
import tempfile
import shutil
from pathlib import Path
import numpy as np
import random
import torch.utils.data

from src.trainer import NICETrainerConfig, NICETrainer
from src.model import NICEConfig, EEGEncoderConfig
from src.data import EEGDatasetConfig


@pytest.fixture
def temp_data_dir():
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "things-eeg2"
    (data_dir / "imgs" / "training_images" / "concept1").mkdir(
        parents=True, exist_ok=True
    )
    (data_dir / "imgs" / "test_images" / "concept1").mkdir(parents=True, exist_ok=True)
    (data_dir / "eeg" / "sub-01").mkdir(parents=True, exist_ok=True)
    (data_dir / "img-latents" / "synclr").mkdir(parents=True, exist_ok=True)
    (data_dir / "imgs" / "training_images" / "concept1" / "img1.jpg").touch()
    (data_dir / "imgs" / "test_images" / "concept1" / "img1.jpg").touch()
    eeg_data = {
        "preprocessed_eeg_data": torch.randn(2, 4, 17, 100).numpy(),
        "ch_names": ["Pz", "P3", "P7"],
        "times": torch.linspace(-0.2, 0.8, 100).numpy(),
    }
    eeg_obj = np.array([eeg_data], dtype=object)
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy",
        eeg_obj,
        allow_pickle=True,
    )  # type: ignore
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_test.npy",
        eeg_obj,
        allow_pickle=True,
    )  # type: ignore
    train_embeddings = torch.randn(2, 768)
    test_embeddings = torch.randn(1, 768)
    torch.save(
        train_embeddings, data_dir / "img-latents" / "synclr" / "train_embeddings.pt"
    )
    torch.save(
        test_embeddings, data_dir / "img-latents" / "synclr" / "test_embeddings.pt"
    )
    yield data_dir, temp_dir
    shutil.rmtree(temp_dir)


def test_nice_trainer_runs_minimal(temp_data_dir):
    data_dir, temp_dir = temp_data_dir
    log_dir = Path(temp_dir) / "logs"
    checkpoint_dir = Path(temp_dir) / "checkpoints"
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    # Use default EEGEncoderConfig which works with (17, 100) input
    config = NICEConfig(
        model_name="synclr",
        batch_size=2,
        eval_batch_size=2,
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
        submodel_config=config,
        dataset_config=dataset_config,
        num_epochs=1,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        save_checkpoints=True,
        log_dir=log_dir,
        checkpoint_dir=checkpoint_dir,
        enable_barebones=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
    )
    trainer = trainer_config.create_trainer()
    trainer.model.data_module.model_name = "synclr"
    trainer.train()
    assert any(checkpoint_dir.glob("*.ckpt")) or any(checkpoint_dir.glob("*.pt"))
    assert log_dir.exists() and any(log_dir.iterdir())


def test_nice_trainer_loss_decreases(nice_trainer_config):
    """Test that NICETrainer reduces loss over a few epochs on a small batch."""
    trainer = NICETrainer(nice_trainer_config)

    # Disable validation by patching the trainer creation
    trainer.pl_trainer.val_check_interval = 0.0
    trainer.pl_trainer.num_sanity_val_steps = 0

    # Track loss
    train_loader = trainer.model.train_dataloader()
    batch = next(iter(train_loader))
    img_latent = batch["img_latent"].to(trainer.model.device, dtype=trainer.model.dtype)
    eeg_data = batch["eeg_data"].to(trainer.model.device, dtype=trainer.model.dtype)
    with torch.no_grad():
        sim = trainer.model(img_latent, eeg_data)
        initial_loss = trainer.model.get_loss(sim).item()

    trainer.train()

    # Get loss after training
    with torch.no_grad():
        sim = trainer.model(img_latent, eeg_data)
        final_loss = trainer.model.get_loss(sim).item()

    print(f"Initial loss: {initial_loss}, Final loss: {final_loss}")
    assert final_loss < initial_loss, (
        f"Final loss ({final_loss}) should be less than initial loss ({initial_loss})"
    )
