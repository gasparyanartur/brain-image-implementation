"""Shared pytest fixtures for brain-image-implementation tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil
from src.model import NICEConfig
from src.data import EEGDatasetConfig
from src.trainer import NICETrainerConfig


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock data structure for testing."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "things-eeg2"

    # Create directory structure
    (data_dir / "imgs" / "training_images" / "concept1").mkdir(
        parents=True, exist_ok=True
    )
    (data_dir / "imgs" / "training_images" / "concept2").mkdir(
        parents=True, exist_ok=True
    )
    (data_dir / "imgs" / "test_images" / "concept1").mkdir(parents=True, exist_ok=True)
    (data_dir / "eeg" / "sub-01").mkdir(parents=True, exist_ok=True)
    (data_dir / "img-latents" / "synclr").mkdir(parents=True, exist_ok=True)

    # Create mock image files
    (data_dir / "imgs" / "training_images" / "concept1" / "img1.jpg").touch()
    (data_dir / "imgs" / "training_images" / "concept1" / "img2.png").touch()
    (data_dir / "imgs" / "training_images" / "concept2" / "img3.jpg").touch()
    (data_dir / "imgs" / "test_images" / "concept1" / "img4.jpg").touch()

    # Create mock EEG data
    eeg_data = {
        "preprocessed_eeg_data": np.random.randn(10, 4, 17, 100),
        "ch_names": ["Pz", "P3", "P7", "O1", "Oz", "O2"],
        "times": np.linspace(-0.2, 0.8, 100),
    }
    eeg_obj = np.array([eeg_data], dtype=object)
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy",
        eeg_obj,
        allow_pickle=True,
    )
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_test.npy",
        eeg_obj,
        allow_pickle=True,
    )

    # Create mock image latents
    train_embeddings = torch.randn(10, 768)
    test_embeddings = torch.randn(5, 768)
    torch.save(
        train_embeddings, data_dir / "img-latents" / "synclr" / "train_embeddings.pt"
    )
    torch.save(
        test_embeddings, data_dir / "img-latents" / "synclr" / "test_embeddings.pt"
    )

    yield data_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_eeg_data():
    """Create mock EEG data for testing."""
    return {
        "preprocessed_eeg_data": np.random.randn(10, 4, 17, 100),
        "ch_names": ["Pz", "P3", "P7", "O1", "Oz", "O2"],
        "times": np.linspace(-0.2, 0.8, 100),
    }


@pytest.fixture
def mock_image_latents():
    """Create mock image latents for testing."""
    return {"train": torch.randn(10, 768), "test": torch.randn(5, 768)}


@pytest.fixture
def sample_eeg_tensor():
    """Create a sample EEG tensor for testing."""
    return torch.randn(5, 4, 17, 100)  # (concepts, repetitions, channels, timesteps)


@pytest.fixture
def sample_image_tensor():
    """Create a sample image tensor for testing."""
    return torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)


@pytest.fixture
def device():
    """Get the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_mock_eeg_image_data_dir(
    tmp_path, n_concepts=2, n_reps=4, n_channels=17, n_timesteps=100, latent_dim=768
):
    data_dir = tmp_path / "things-eeg2"
    for split in ["training_images", "test_images"]:
        for concept in [f"concept{i + 1}" for i in range(n_concepts)]:
            (data_dir / "imgs" / split / concept).mkdir(parents=True, exist_ok=True)
            (data_dir / "imgs" / split / concept / "img1.jpg").touch()
    (data_dir / "img-latents" / "synclr").mkdir(parents=True, exist_ok=True)
    (data_dir / "eeg" / "sub-01").mkdir(parents=True, exist_ok=True)
    eeg_data = {
        "preprocessed_eeg_data": torch.randn(
            n_concepts, n_reps, n_channels, n_timesteps
        ).numpy(),
        "ch_names": ["Pz", "P3", "P7"],
        "times": torch.linspace(-0.2, 0.8, n_timesteps).numpy(),
    }
    eeg_obj = np.array([eeg_data], dtype=object)
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy",
        eeg_obj,
        allow_pickle=True,
    )
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_test.npy",
        eeg_obj,
        allow_pickle=True,
    )
    train_embeddings = torch.randn(n_concepts, latent_dim)
    test_embeddings = torch.randn(n_concepts, latent_dim)
    torch.save(
        train_embeddings, data_dir / "img-latents" / "synclr" / "train_embeddings.pt"
    )
    torch.save(
        test_embeddings, data_dir / "img-latents" / "synclr" / "test_embeddings.pt"
    )
    return data_dir


@pytest.fixture
def mock_eeg_image_data_dir(tmp_path):
    return create_mock_eeg_image_data_dir(tmp_path)


@pytest.fixture
def nice_config():
    return NICEConfig(
        model_name="synclr", batch_size=2, eval_batch_size=2, max_epochs=5
    )


@pytest.fixture
def eeg_dataset_config(mock_eeg_image_data_dir):
    return EEGDatasetConfig(
        data_path=mock_eeg_image_data_dir,
        batch_size=2,
        val_batch_size=2,
        subs=[1],
        num_workers=0,
    )


@pytest.fixture
def nice_trainer_config(nice_config, eeg_dataset_config, tmp_path):
    return NICETrainerConfig(
        num_epochs=5,
        learning_rate=1e-2,
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        submodel_config=nice_config,
        dataset_config=eeg_dataset_config,
        log_dir=tmp_path / "logs",
        checkpoint_dir=tmp_path / "ckpts",
        enable_progress_bar=False,
        enable_model_summary=False,
        log_every_n_steps=1,
        save_checkpoints=False,
    )
