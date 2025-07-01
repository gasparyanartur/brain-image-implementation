import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
import pickle

from src.data import (
    DataConfig,
    DataModule,
    EEGDatasetConfig,
    EEGDataModule,
    EEGDataset,
    prepare_datasets,
    load_image_from_path,
    batch_load_images,
    load_eeg_data,
    preprocess_image,
    preprocess_eeg_data,
    get_image_paths,
    load_all_eeg_data,
)
from src.configs import DEFAULT_BATCH_SIZE


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with mock data structure."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "things-eeg2"

    # Create directory structure
    (data_dir / "imgs" / "training_images" / "concept1").mkdir(
        parents=True, exist_ok=True
    )
    (data_dir / "imgs" / "test_images" / "concept1").mkdir(parents=True, exist_ok=True)
    (data_dir / "eeg" / "sub-01").mkdir(parents=True, exist_ok=True)
    (data_dir / "img-latents" / "synclr").mkdir(parents=True, exist_ok=True)

    # Create mock image files
    (data_dir / "imgs" / "training_images" / "concept1" / "img1.jpg").touch()
    (data_dir / "imgs" / "test_images" / "concept1" / "img1.jpg").touch()

    # Create mock EEG data
    eeg_data = {
        "preprocessed_eeg_data": np.random.randn(10, 4, 17, 100),
        "ch_names": ["Pz", "P3", "P7"],
        "times": np.linspace(-0.2, 0.8, 100),
    }
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy",
        eeg_data,  # type: ignore
        allow_pickle=True,
    )
    np.save(
        data_dir / "eeg" / "sub-01" / "preprocessed_eeg_test.npy",
        eeg_data,  # type: ignore
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


def test_eeg_dataset_creation(temp_data_dir):
    """Test that EEGDataset can be created and loads data correctly."""
    config = EEGDatasetConfig(data_path=temp_data_dir, subs=[1])

    # Test training dataset
    train_dataset = EEGDataset(config, split="train", model_name="synclr")
    assert len(train_dataset) > 0
    assert len(train_dataset.img_paths) > 0
    assert train_dataset.img_latents.shape[0] > 0
    assert len(train_dataset.eeg_data) > 0

    # Test test dataset
    test_dataset = EEGDataset(config, split="test", model_name="synclr")
    assert len(test_dataset) > 0
    assert len(test_dataset.img_paths) > 0
    assert test_dataset.img_latents.shape[0] > 0
    assert len(test_dataset.eeg_data) > 0


def test_eeg_dataset_getitem(temp_data_dir):
    """Test that EEGDataset returns correct data structure."""
    config = EEGDatasetConfig(data_path=temp_data_dir, subs=[1])
    dataset = EEGDataset(config, split="train", model_name="synclr")

    if len(dataset) > 0:
        item = dataset[0]
        assert isinstance(item, dict)
        assert "img_path" in item
        assert "img_latent" in item
        assert "eeg_data" in item
        assert isinstance(item["img_path"], str)
        assert isinstance(item["img_latent"], torch.Tensor)
        assert isinstance(item["eeg_data"], torch.Tensor)


def test_eeg_data_module(temp_data_dir):
    """Test that EEGDataModule creates dataloaders correctly."""
    config = EEGDatasetConfig(
        data_path=temp_data_dir,
        batch_size=4,
        val_batch_size=2,
        subs=[1],
        num_workers=0,  # Use 0 for testing
    )
    module = EEGDataModule(config, model_name="synclr")

    # Test dataloader creation
    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()
    test_loader = module.test_dataloader()

    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 2
    assert test_loader.batch_size == 2

    # Test that we can get a batch
    for batch in train_loader:
        assert "img_path" in batch
        assert "img_latent" in batch
        assert "eeg_data" in batch
        assert batch["img_latent"].shape[0] == 4  # batch_size
        assert batch["eeg_data"].shape[0] == 4  # batch_size
        break


def test_preprocess_eeg_data():
    """Test EEG data preprocessing."""
    # Create random EEG data: (concepts, repetitions, channels, timesteps)
    eeg_data = torch.randn(5, 4, 17, 100)

    processed_data = preprocess_eeg_data(eeg_data)

    assert processed_data.shape == (5, 17, 100)  # Should average over repetitions
    assert torch.allclose(processed_data, torch.mean(eeg_data, dim=1))


def test_preprocess_image():
    """Test image preprocessing."""
    # Create random image tensor
    image = torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)

    processed_image = preprocess_image(image, img_size=(224, 224))

    assert processed_image.shape == (3, 224, 224)
    assert processed_image.dtype == torch.float32
    assert torch.all(processed_image >= 0) and torch.all(processed_image <= 1)


def test_get_image_paths():
    """Test image path collection."""
    temp_dir = tempfile.mkdtemp()
    img_dir = Path(temp_dir) / "images"

    # Create directory structure
    (img_dir / "training_images" / "concept1").mkdir(parents=True, exist_ok=True)
    (img_dir / "test_images" / "concept1").mkdir(parents=True, exist_ok=True)

    # Create mock image files
    (img_dir / "training_images" / "concept1" / "img1.jpg").touch()
    (img_dir / "test_images" / "concept1" / "img2.jpg").touch()

    try:
        # Test training paths
        train_paths = get_image_paths(img_dir, split="train")
        assert len(train_paths) == 1
        assert "training_images" in str(train_paths[0])

        # Test test paths
        test_paths = get_image_paths(img_dir, split="test")
        assert len(test_paths) == 1
        assert "test_images" in str(test_paths[0])

    finally:
        shutil.rmtree(temp_dir)


def test_load_eeg_data():
    """Test EEG data loading."""
    temp_dir = tempfile.mkdtemp()
    eeg_file = Path(temp_dir) / "eeg_data.npy"

    # Create mock EEG data
    eeg_data = {
        "preprocessed_eeg_data": np.random.randn(10, 4, 17, 100),
        "ch_names": ["Pz", "P3", "P7"],
        "times": np.linspace(-0.2, 0.8, 100),
    }
    # Save as numpy file to match the expected format
    np.save(
        eeg_file,
        eeg_data,  # type: ignore
        allow_pickle=True,
    )

    try:
        loaded_data, times, ch_names = load_eeg_data(eeg_file)

        assert isinstance(loaded_data, torch.Tensor)
        assert isinstance(times, torch.Tensor)
        assert isinstance(ch_names, list)
        assert loaded_data.shape == (10, 4, 17, 100)
        assert times.shape == (100,)
        assert len(ch_names) == 3

    finally:
        shutil.rmtree(temp_dir)


def test_load_all_eeg_data():
    """Test loading multiple EEG files."""
    temp_dir = tempfile.mkdtemp()
    eeg_file1 = Path(temp_dir) / "eeg_data1.npy"
    eeg_file2 = Path(temp_dir) / "eeg_data2.npy"

    # Create mock EEG data
    eeg_data = {
        "preprocessed_eeg_data": np.random.randn(5, 4, 17, 100),
        "ch_names": ["Pz", "P3", "P7"],
        "times": np.linspace(-0.2, 0.8, 100),
    }
    # Save as numpy files to match the expected format
    np.save(eeg_file1, eeg_data, allow_pickle=True)  # type: ignore
    np.save(eeg_file2, eeg_data, allow_pickle=True)  # type: ignore

    try:
        all_data, times, ch_names = load_all_eeg_data([eeg_file1, eeg_file2])

        assert isinstance(all_data, torch.Tensor)
        assert isinstance(times, torch.Tensor)
        assert isinstance(ch_names, list)
        assert all_data.shape[0] == 10  # 5 * 2 files
        assert times.shape == (100,)
        assert len(ch_names) == 3

    finally:
        shutil.rmtree(temp_dir)
