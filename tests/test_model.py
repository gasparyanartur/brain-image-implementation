import pytest
import torch
import tempfile
import shutil
from pathlib import Path

from brain_image.model.eeg_alignment import EEGAlignmentModel
from brain_image.model.eeg_encoder import NiceEEGEncoder, EEGEncoderConfig
from brain_image.model.model import (
    LatentProjector,
)
from brain_image.data import EEGDatasetConfig


def test_eeg_encoder_creation():
    """Test that EEGEncoder can be created successfully."""
    config = EEGEncoderConfig()
    encoder = NiceEEGEncoder(config)
    
    assert encoder is not None
    assert hasattr(encoder, "patch_embedding")
    
    # Test forward pass
    batch_size = 4
    eeg_data = torch.randn(batch_size, 17, 100)  # channels, timesteps
    output = encoder(eeg_data)
    
    # Check output shape - should be (batch_size, embed_dim)
    assert torch.isfinite(output).all()


def test_latent_projector_creation():
    """Test that LatentProjector can be created successfully."""
    projector = LatentProjector(embed_dim=1440, proj_dim=768)
    
    assert projector is not None
    assert hasattr(projector, "l_proj")
    assert hasattr(projector, "l_inner")
    assert hasattr(projector, "norm1")
    assert hasattr(projector, "l_out")
    
    # Test forward pass
    batch_size = 4
    input_data = torch.randn(batch_size, 1440)
    output = projector(input_data)
    
    assert output.shape == (batch_size, 768)
    assert torch.isfinite(output).all()


def test_nice_model_creation():
    """Test that NICE model can be created successfully."""
    config = NICEConfig(align_target_model="aligned_synclr_16")
    model = NICEModel(config=config, compile=False, preload_latents=False)

    assert model is not None
    assert hasattr(model, "eeg_encoder")
    assert hasattr(model, "eeg_projector")
    assert hasattr(model, "img_projector")
    assert hasattr(model, "temperature")
    assert hasattr(model, "data_module")
    assert model.automatic_optimization is False


def test_nice_model_forward_pass():
    """Test that NICE model can perform a forward pass."""
    config = NICEConfig(align_target_model="aligned_synclr_16")
    model = NICEModel(config=config, compile=False, preload_latents=False)

    # Create mock input data
    batch_size = 4
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)  # channels, timesteps

    # Forward pass
    output = model(img_latent, eeg_data)

    # Check output shape
    assert output.shape == (batch_size, batch_size)  # similarity matrix
    assert torch.isfinite(output).all()


def test_nice_model_get_similarity():
    """Test that NICE model can compute similarity."""
    config = NICEConfig(align_target_model="aligned_synclr_16")
    model = NICEModel(config=config, compile=False, preload_latents=False)

    # Create mock input data
    batch_size = 4
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)

    # Get similarity
    sim = model.get_similarity(img_latent, eeg_data)

    # Check similarity matrix
    assert sim.shape == (batch_size, batch_size)
    assert torch.isfinite(sim).all()


def test_nice_model_loss_computation():
    """Test that NICE model can compute loss."""
    config = NICEConfig(align_target_model="aligned_synclr_16")
    model = NICEModel(config=config, compile=False, preload_latents=False)

    # Create mock input data
    batch_size = 4
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)

    # Forward pass
    sim = model.get_similarity(img_latent, eeg_data)

    # Compute loss
    loss = model.get_align_loss(sim)

    # Check loss
    assert torch.isfinite(loss)
    assert loss > 0


def test_nice_model_accuracy_computation():
    """Test that NICE model can compute accuracy."""
    config = NICEConfig(align_target_model="aligned_synclr_16")
    model = NICEModel(config=config, compile=False, preload_latents=False)

    # Create mock input data
    batch_size = 8  # Use larger batch size for top-5 test
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)

    # Forward pass
    sim = model.get_similarity(img_latent, eeg_data)

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
    config = NICEConfig(align_target_model="aligned_synclr_16")
    dataset_config = EEGDatasetConfig(
        data_path=mock_data_directory["data_dir"],
        subs=[1],
        num_workers=0,  # Use 0 for testing
    )

    model = NICEModel(
        config=config,
        dataset_config=dataset_config,
        compile=False,
        preload_latents=False,
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
        assert "eeg_data" in batch
        break


def test_nice_model_configure_optimizers():
    """Test that NICE model can configure optimizers."""
    config = NICEConfig(align_target_model="aligned_synclr_16")
    model = NICEModel(config=config, compile=False, preload_latents=False)
    
    optimizers = model.configure_optimizers()
    
    # Should return a list of optimizer configurations
    assert isinstance(optimizers, list)
    assert len(optimizers) == 2  # encoder and projector optimizers
    
    # Check that each optimizer config has required keys
    for opt_config in optimizers:
        assert "optimizer" in opt_config
        assert "lr_scheduler" in opt_config
        assert "interval" in opt_config
        assert "frequency" in opt_config


def test_compute_cross_entropy_loss():
    """Test the compute_cross_entropy_loss function."""
    batch_size = 4
    sim = torch.randn(batch_size, batch_size)
    
    loss = compute_cross_entropy_loss(sim)
    
    assert torch.isfinite(loss)
    assert loss > 0
    assert loss.dtype == sim.dtype


def test_compute_similarity():
    """Test the compute_similarity function."""
    batch_size = 4
    eeg_latent = torch.randn(batch_size, 256)
    img_latent = torch.randn(batch_size, 256)
    temperature = torch.tensor(0.1)
    
    sim = compute_similarity(eeg_latent, img_latent, temperature)
    
    assert sim.shape == (batch_size, batch_size)
    assert torch.isfinite(sim).all()


def test_nice_model_with_image_projection():
    """Test NICE model with image projection enabled."""
    config = NICEConfig(
        align_target_model="aligned_synclr_16",
        project_image=True
    )
    model = NICEModel(config=config, compile=False, preload_latents=False)
    
    assert model.align_img_projector is not None
    
    # Test forward pass
    batch_size = 4
    img_latent = torch.randn(batch_size, config.img_latent_dim)
    eeg_data = torch.randn(batch_size, 17, 100)
    
    output = model(img_latent, eeg_data)
    assert output.shape == (batch_size, batch_size)
    assert torch.isfinite(output).all()


def test_nice_model_checkpoint_loading():
    """Test that NICE model can load checkpoints."""
    config = NICEConfig(align_target_model="aligned_synclr_16")
    model = NICEModel(config=config, compile=False, preload_latents=False)
    
    # Create a temporary checkpoint
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        checkpoint_path = tmp_file.name
    
    try:
        # Save a checkpoint
        torch.save({
            "state_dict": model.state_dict(),
            "hyperparameters": model.hparams,
            "pytorch-lightning_version": "1.10.0",
        }, checkpoint_path)
        
        # Load the checkpoint
        loaded_model = NICEModel.load_checkpoint(checkpoint_path, config=config, compile=False, preload_latents=False)
        
        assert loaded_model is not None
        assert isinstance(loaded_model, NICEModel)
        
    finally:
        # Clean up
        if Path(checkpoint_path).exists():
            Path(checkpoint_path).unlink()


