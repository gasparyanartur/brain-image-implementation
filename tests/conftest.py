"""Shared pytest fixtures for brain-image-implementation tests."""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil


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
    np.save(data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy", eeg_data)
    np.save(data_dir / "eeg" / "sub-01" / "preprocessed_eeg_test.npy", eeg_data)

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
