"""Shared pytest fixtures for brain-image-implementation tests."""

import pytest
import torch
import numpy as np
from brain_image.model import NICEConfig
from brain_image.data import EEGDatasetConfig
from brain_image.trainer import NICETrainerConfig


@pytest.fixture
def mock_data_config():
    train_categories = [
        "01_ab",
        "02_cd",
        "03_ef",
        "04_gh",
        "05_ij",
        "06_kl",
        "07_mn",
        "08_op",
    ]
    test_categories = [
        "01_uv",
        "02_wx",
        "03_yz",
        "04_ab",
    ]
    num_samples_per_category_train = 10
    num_samples_per_category_test = 1
    subs = [1, 2, 3, 4, 5]
    channels = [
        "Pz",
        "P3",
        "P7",
        "O1",
        "Oz",
        "O2",
        "P4",
        "P8",
        "P1",
        "P5",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "P6",
        "P2",
    ]
    num_timesteps = 100
    latent_dim = 768
    train_batch_size = 4
    test_batch_size = 2
    return {
        "train_categories": train_categories,
        "test_categories": test_categories,
        "num_samples_per_category_train": num_samples_per_category_train,
        "num_samples_per_category_test": num_samples_per_category_test,
        "subs": subs,
        "channels": channels,
        "num_timesteps": num_timesteps,
        "latent_dim": latent_dim,
        "train_batch_size": train_batch_size,
        "test_batch_size": test_batch_size,
    }


@pytest.fixture
def mock_data_directory(tmp_path, mock_data_config):
    """Create a temporary directory with mock data structure for testing."""

    data_dir = tmp_path / "things-eeg2"
    log_dir = tmp_path / "logs"
    checkpoint_dir = tmp_path / "ckpts"
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)

    for sub in mock_data_config["subs"]:
        (data_dir / "eeg" / f"sub-{sub:02d}").mkdir(parents=True, exist_ok=True)
    (data_dir / "img-latents" / "synclr").mkdir(parents=True, exist_ok=True)

    # Create mock EEG data
    train_eeg_data = {
        "preprocessed_eeg_data": np.random.randn(
            mock_data_config["train_batch_size"]
            * len(mock_data_config["train_categories"]),
            4,
            len(mock_data_config["channels"]),
            mock_data_config["num_timesteps"],
        ),
        "ch_names": mock_data_config["channels"],
        "times": np.linspace(-0.2, 0.8, mock_data_config["num_timesteps"]),
    }
    test_eeg_data = {
        "preprocessed_eeg_data": np.random.randn(
            mock_data_config["test_batch_size"]
            * len(mock_data_config["train_categories"]),
            4,
            len(mock_data_config["channels"]),
            mock_data_config["num_timesteps"],
        ),
        "ch_names": mock_data_config["channels"],
        "times": np.linspace(-0.2, 0.8, mock_data_config["num_timesteps"]),
    }
    eeg_obj_train = np.array([train_eeg_data], dtype=object)
    eeg_obj_test = np.array([test_eeg_data], dtype=object)
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy",
        eeg_obj_train,
        allow_pickle=True,
    )
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_test.npy",
        eeg_obj_test,
        allow_pickle=True,
    )

    # Create images
    img_dir = data_dir / "imgs"
    img_dir.mkdir(parents=True, exist_ok=True)
    for category in mock_data_config["train_categories"]:
        (data_dir / "imgs" / "training_images" / category).mkdir(
            parents=True, exist_ok=True
        )
        for i in range(mock_data_config["num_samples_per_category_train"]):
            (
                data_dir / "imgs" / "training_images" / category / f"{category}{i}.jpg"
            ).touch()

    for category in mock_data_config["test_categories"]:
        (data_dir / "imgs" / "test_images" / category).mkdir(
            parents=True, exist_ok=True
        )
        for i in range(mock_data_config["num_samples_per_category_test"]):
            (
                data_dir / "imgs" / "test_images" / category / f"{category}{i}.jpg"
            ).touch()

    # Create mock image latents
    train_embeddings = torch.randn(
        mock_data_config["train_batch_size"]
        * len(mock_data_config["train_categories"]),
        mock_data_config["latent_dim"],
    )
    test_embeddings = torch.randn(
        mock_data_config["test_batch_size"] * len(mock_data_config["test_categories"]),
        mock_data_config["latent_dim"],
    )
    torch.save(
        train_embeddings, data_dir / "img-latents" / "synclr" / "train_embeddings.pt"
    )
    torch.save(
        test_embeddings, data_dir / "img-latents" / "synclr" / "test_embeddings.pt"
    )

    mock_data_directory = {
        "root_dir": tmp_path,
        "data_dir": data_dir,
        "log_dir": log_dir,
        "checkpoint_dir": checkpoint_dir,
    }

    return mock_data_directory


@pytest.fixture
def device():
    """Get the device to use for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
