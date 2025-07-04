import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from brain_image.model import NICEModel, NICEConfig
from brain_image.data import EEGDatasetConfig


def test_nice_model_creation():
    """Test that NICE model can be created successfully."""
    config = NICEConfig(model_name="synclr")
    model = NICEModel(config=config, compile=False)

    assert model is not None
    assert hasattr(model, "eeg_encoder")
    assert hasattr(model, "eeg_projector")
    assert hasattr(model, "img_projector")
    assert hasattr(model, "temperature")


def test_nice_model_forward_pass():
    """Test that NICE model can perform a forward pass."""
    config = NICEConfig(model_name="synclr")
    model = NICEModel(config=config, compile=False)

    # Create mock input data
    batch_size = 4
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)  # channels, timesteps

    # Forward pass
    output = model(img_latent, eeg_data)

    # Check output shape
    assert output.shape == (batch_size, batch_size)  # similarity matrix
    assert torch.isfinite(output).all()


def test_nice_model_loss_computation():
    """Test that NICE model can compute loss."""
    config = NICEConfig(model_name="synclr")
    model = NICEModel(config=config, compile=False)

    # Create mock input data
    batch_size = 4
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)

    # Forward pass
    sim = model(img_latent, eeg_data)

    # Compute loss
    loss = model.get_loss(sim)

    # Check loss
    assert torch.isfinite(loss)
    assert loss > 0


def test_nice_model_accuracy_computation():
    """Test that NICE model can compute accuracy."""
    config = NICEConfig(model_name="synclr")
    model = NICEModel(config=config, compile=False)

    # Create mock input data
    batch_size = 8  # Use larger batch size for top-5 test
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)

    # Forward pass
    sim = model(img_latent, eeg_data)

    # Compute accuracies
    top1_acc = model.get_top_n_accuracy(sim, n=1)
    top3_acc = model.get_top_n_accuracy(sim, n=3)
    top5_acc = model.get_top_n_accuracy(sim, n=5)

    # Check accuracies
    assert 0 <= top1_acc <= 1
    assert 0 <= top3_acc <= 1
    assert 0 <= top5_acc <= 1
    assert top1_acc <= top3_acc <= top5_acc


def test_nice_model_with_data_module(mock_data_directory):
    """Test that NICE model works with data module."""
    config = NICEConfig(model_name="synclr")
    dataset_config = EEGDatasetConfig(
        data_path=mock_data_directory["data_dir"],
        subs=[1],
        num_workers=0,  # Use 0 for testing
    )

    model = NICEModel(
        config=config,
        dataset_config=dataset_config,
        compile=False,
    )

    # Test that dataloaders can be created
    train_loader = model.train_dataloader()
    val_loader = model.val_dataloader()
    test_loader = model.test_dataloader()

    assert train_loader is not None
    assert val_loader is not None
    assert test_loader is not None

    # Test that we can get a batch
    for batch in train_loader:
        assert "img_path" in batch
        assert "img_latent" in batch
        assert "eeg_data" in batch
        break
