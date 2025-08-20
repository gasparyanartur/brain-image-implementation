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

from brain_image.trainer import NICETrainerConfig, NICETrainer, load_eeg_encoder_from_checkpoint
from brain_image.model import EEGEncoderConfig, EEGEncoder
from brain_image.data import EEGDatasetConfig
from brain_image.utils import state_dict_equal


def test_nice_trainer_checkpoint_saving_and_loading(mock_nice_trainer):
    """Test that NICETrainer can save and load checkpoints."""
    trainer = mock_nice_trainer["trainer"]
    trainer_config = mock_nice_trainer["trainer_config"]
    nice_config = mock_nice_trainer["nice_config"]
    dataset_config = mock_nice_trainer["dataset_config"]

    with tempfile.TemporaryDirectory() as tmp_dir:
        checkpoint_path = Path(tmp_dir) / "test.ckpt"

        # Save a checkpoint
        trainer.save_checkpoint(checkpoint_path)
        assert checkpoint_path.exists()

        # Create a new trainer and load the checkpoint
        new_trainer = NICETrainer(
            config=trainer_config,
            model_config=nice_config,
            dataset_config=dataset_config,
        )
        new_trainer.load_checkpoint(checkpoint_path)

        # Compare model state dicts
        original_state_dict = trainer.model.state_dict()
        loaded_state_dict = new_trainer.model.state_dict()

        for key in original_state_dict:
            assert torch.equal(original_state_dict[key], loaded_state_dict[key])


def test_nice_trainer_predict(mock_nice_trainer):
    """Test that NICETrainer can run predictions."""
    trainer = mock_nice_trainer["trainer"]
    
    # Get a test dataloader
    test_loader = trainer.model.test_dataloader()
    
    # Run predictions
    predictions = trainer.predict(dataloader=test_loader)
    
    assert isinstance(predictions, list)
    assert len(predictions) > 0
    
    # Check the structure of a prediction
    prediction = predictions[0]
    assert isinstance(prediction, dict)
    assert "similarity" in prediction
    assert "loss" in prediction
    assert "top_1_accuracy" in prediction
    assert "top_5_accuracy" in prediction
    assert "eeg_latent" in prediction
    assert "img_latent" in prediction
    assert "proj_eeg_latent" in prediction
    assert "proj_img_latent" in prediction
    assert "img_paths" in prediction
    assert "eeg_paths" in prediction
    
    # Check shapes
    batch_size = trainer.model.data_module.config.val_batch_size
    assert prediction["similarity"].shape == (batch_size, batch_size)
    assert prediction["eeg_latent"].shape[0] == batch_size
    assert prediction["img_latent"].shape[0] == batch_size
    assert prediction["proj_eeg_latent"].shape[0] == batch_size
    assert prediction["proj_img_latent"].shape[0] == batch_size
    assert len(prediction["img_paths"]) == batch_size
    assert len(prediction["eeg_paths"]) == batch_size

def test_load_eeg_encoder_from_checkpoint():
    """Test that EEG encoder can be loaded from a checkpoint."""
    config = EEGEncoderConfig()
    encoder = EEGEncoder(config)

    class DistractionModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

            self.x1 = torch.nn.Parameter(torch.randn(10))
            self.x2 = torch.nn.Parameter(torch.randn(10))
    
    encoder_state_dict = {
                "eeg_encoder." + key: value
                for key, value in encoder.state_dict().items()
            }
    distractor_state_dict = {
                "distractor." + key: value
                for key, value in DistractionModule().state_dict().items()
            }
    checkpoint = {
        "state_dict": {
            **encoder_state_dict,
            **distractor_state_dict
        }
    }


    
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp_file:
        checkpoint_path = Path(tmp_file.name)
        torch.save(checkpoint, checkpoint_path)

    try:
        loaded_encoder = load_eeg_encoder_from_checkpoint(config, checkpoint_path)
        
        assert loaded_encoder is not None
        assert isinstance(loaded_encoder, EEGEncoder)
        
        original_state_dict = encoder.state_dict()
        loaded_state_dict = loaded_encoder.state_dict()
        
        assert state_dict_equal(original_state_dict, loaded_state_dict)
            
    finally:
        if checkpoint_path.exists():
            checkpoint_path.unlink()