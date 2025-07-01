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


def test_eeg_dataset_creation(mock_eeg_image_data_dir):
    config = EEGDatasetConfig(data_path=mock_eeg_image_data_dir, subs=[1])
    train_dataset = EEGDataset(config, split="train", model_name="synclr")
    assert len(train_dataset) > 0
    assert len(train_dataset.img_paths) > 0
    assert train_dataset.img_latents.shape[0] > 0
    assert len(train_dataset.eeg_data) > 0
    test_dataset = EEGDataset(config, split="test", model_name="synclr")
    assert len(test_dataset) > 0
    assert len(test_dataset.img_paths) > 0
    assert test_dataset.img_latents.shape[0] > 0
    assert len(test_dataset.eeg_data) > 0


def test_eeg_dataset_getitem(mock_eeg_image_data_dir):
    config = EEGDatasetConfig(data_path=mock_eeg_image_data_dir, subs=[1])
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


def test_eeg_data_module(mock_eeg_image_data_dir):
    config = EEGDatasetConfig(
        data_path=mock_eeg_image_data_dir,
        batch_size=4,
        val_batch_size=2,
        subs=[1],
        num_workers=0,
    )
    module = EEGDataModule(config, model_name="synclr")
    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()
    test_loader = module.test_dataloader()
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 2
    assert test_loader.batch_size == 2
    for batch in train_loader:
        assert "img_path" in batch
        assert "img_latent" in batch
        assert "eeg_data" in batch
        assert batch["img_latent"].shape[0] == 4
        assert batch["eeg_data"].shape[0] == 4
        break


def test_preprocess_eeg_data():
    eeg_data = torch.randn(5, 4, 17, 100)
    processed_data = preprocess_eeg_data(eeg_data)
    assert processed_data.shape == (5, 17, 100)
    assert torch.allclose(processed_data, torch.mean(eeg_data, dim=1))


def test_preprocess_image():
    image = torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)
    processed_image = preprocess_image(image, img_size=(224, 224))
    assert processed_image.shape == (3, 224, 224)
    assert processed_image.dtype == torch.float32
    assert torch.all(processed_image >= 0) and torch.all(processed_image <= 1)


def test_get_image_paths(mock_eeg_image_data_dir):
    img_dir = mock_eeg_image_data_dir / "imgs"
    train_paths = get_image_paths(img_dir, split="train")
    assert len(train_paths) > 0
    assert "training_images" in str(train_paths[0])
    test_paths = get_image_paths(img_dir, split="test")
    assert len(test_paths) > 0
    assert "test_images" in str(test_paths[0])


def test_load_eeg_data(mock_eeg_image_data_dir):
    eeg_file = (
        mock_eeg_image_data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy"
    )
    loaded_data, times, ch_names = load_eeg_data(eeg_file)
    assert isinstance(loaded_data, torch.Tensor)
    assert isinstance(times, torch.Tensor)
    assert isinstance(ch_names, list)
    assert loaded_data.shape[1:] == (17, 100)
    assert times.shape == (100,)
    assert len(ch_names) == 3


def test_load_all_eeg_data(mock_eeg_image_data_dir):
    eeg_file1 = (
        mock_eeg_image_data_dir / "eeg" / "sub-01" / "preprocessed_eeg_training.npy"
    )
    eeg_file2 = mock_eeg_image_data_dir / "eeg" / "sub-01" / "preprocessed_eeg_test.npy"
    all_data, times, ch_names = load_all_eeg_data([eeg_file1, eeg_file2])
    assert isinstance(all_data, torch.Tensor)
    assert isinstance(times, torch.Tensor)
    assert isinstance(ch_names, list)
    assert all_data.shape[1:] == (17, 100)
    assert times.shape == (100,)
    assert len(ch_names) == 3
