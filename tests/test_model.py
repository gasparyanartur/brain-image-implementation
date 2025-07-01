import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from src.model import NICEModel, NICEConfig
from src.data import EEGDatasetConfig


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
        "preprocessed_eeg_data": torch.randn(10, 4, 17, 100).numpy(),
        "ch_names": ["Pz", "P3", "P7"],
        "times": torch.linspace(-0.2, 0.8, 100).numpy(),
    }
    import numpy as np

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


def test_nice_model_with_data_module(temp_data_dir):
    """Test that NICE model works with data module."""
    config = NICEConfig(model_name="synclr")
    dataset_config = EEGDatasetConfig(
        data_path=temp_data_dir,
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
