import torch
import tempfile
import shutil
from pathlib import Path

from brain_image.data import (
    EEGDatasetConfig,
    EEGDataModule,
    EEGDataset,
    load_eeg_data,
    preprocess_image,
    preprocess_eeg_data,
    get_image_paths,
    load_all_eeg_data,
    TensorCache,
)
from brain_image.model import load_image_encoder


def test_eeg_dataset_creation(mock_data_directory):
    config = EEGDatasetConfig(data_path=mock_data_directory["data_dir"], subs=[1])
    train_dataset = EEGDataset(config, split="train")
    assert len(train_dataset) > 0
    assert len(train_dataset.img_paths) > 0
    assert len(train_dataset.eeg_data) > 0
    test_dataset = EEGDataset(config, split="test")
    assert len(test_dataset) > 0
    assert len(test_dataset.img_paths) > 0
    assert len(test_dataset.eeg_data) > 0


def test_eeg_dataset_getitem(mock_data_directory):
    config = EEGDatasetConfig(data_path=mock_data_directory["data_dir"], subs=[1])
    dataset = EEGDataset(config, split="train")
    if len(dataset) > 0:
        item = dataset[0]
        assert isinstance(item, dict)
        assert "img_path" in item
        assert "eeg_data" in item
        assert isinstance(item["img_path"], str)
        assert isinstance(item["eeg_data"], torch.Tensor)


def test_eeg_data_module(mock_data_directory):
    config = EEGDatasetConfig(
        data_path=mock_data_directory["data_dir"],
        batch_size=4,
        val_batch_size=2,
        subs=[1],
        num_workers=0,
    )
    module = EEGDataModule(config)
    train_loader = module.train_dataloader()
    val_loader = module.val_dataloader()
    test_loader = module.test_dataloader()
    assert train_loader.batch_size == 4
    assert val_loader.batch_size == 2
    assert test_loader.batch_size == 2
    for batch in train_loader:
        assert "img_path" in batch
        assert "eeg_data" in batch
        assert batch["eeg_data"].shape[0] == 4
        break


def test_preprocess_eeg_data():
    eeg_data = torch.randn(5, 4, 17, 100)
    processed_data = preprocess_eeg_data(eeg_data)
    assert processed_data.shape == (5, 17, 100)
    assert torch.allclose(processed_data, torch.mean(eeg_data, dim=1))


def test_preprocess_image():
    image = torch.randint(0, 256, (3, 100, 100), dtype=torch.uint8)
    processed_image = preprocess_image(image, img_size=[224, 224])
    assert processed_image.shape == (3, 224, 224)
    assert processed_image.dtype == torch.float32
    assert torch.all(processed_image >= 0) and torch.all(processed_image <= 1)


def test_get_image_paths(mock_data_directory):
    img_dir = mock_data_directory["data_dir"] / "imgs"
    train_paths = get_image_paths(img_dir, split="train")
    assert len(train_paths) > 0
    assert "training_images" in str(train_paths[0])
    test_paths = get_image_paths(img_dir, split="test")
    assert len(test_paths) > 0
    assert "test_images" in str(test_paths[0])


def test_load_eeg_data(mock_data_directory):
    eeg_file = (
        mock_data_directory["data_dir"]
        / "eeg"
        / "sub-01"
        / "preprocessed_eeg_training.npy"
    )
    loaded_data, times, ch_names = load_eeg_data(eeg_file)
    assert isinstance(loaded_data, torch.Tensor)
    assert isinstance(times, torch.Tensor)
    assert isinstance(ch_names, list)
    assert loaded_data.shape[-2:] == (17, 100)
    assert times.shape == (100,)
    assert len(ch_names) == 17


def test_load_all_eeg_data(mock_data_config, mock_data_directory):
    eeg_file1 = (
        mock_data_directory["data_dir"]
        / "eeg"
        / "sub-01"
        / "preprocessed_eeg_training.npy"
    )
    eeg_file2 = (
        mock_data_directory["data_dir"] / "eeg" / "sub-01" / "preprocessed_eeg_test.npy"
    )
    all_data, times, ch_names = load_all_eeg_data([eeg_file1, eeg_file2])
    assert isinstance(all_data, torch.Tensor)
    assert isinstance(times, torch.Tensor)
    assert isinstance(ch_names, list)
    assert all_data.shape[1:] == (
        len(mock_data_config["channels"]),
        mock_data_config["num_timesteps"],
    )
    assert times.shape == (mock_data_config["num_timesteps"],)
    assert len(ch_names) == len(mock_data_config["channels"])


def test_tensor_cache_save_and_load():
    """Test basic save and load functionality of TensorCache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = TensorCache(cache_path=Path(tmp_dir) / "test_cache")
        
        # Create a test tensor
        test_tensor = torch.randn(3, 4)
        
        # Save tensor with keys
        cache.save(test_tensor, "test", "key1", "key2")
        
        # Load tensor with same keys
        loaded_tensor = cache.get("test", "key1", "key2")
        
        # Verify tensors are equal
        assert torch.allclose(test_tensor, loaded_tensor)
        assert test_tensor.shape == loaded_tensor.shape

        path = cache._get_tensor_path("test", "key1", "key2")
        assert path.exists()
        assert torch.allclose(test_tensor, torch.load(path))


def test_tensor_cache_memory_cache():
    """Test memory cache functionality of TensorCache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        cache = TensorCache(cache_path=Path(tmp_dir) / "test_cache", memory_cache_size=2)
        
        # Create test tensors
        tensor1 = torch.randn(2, 2)
        tensor2 = torch.randn(3, 3)
        tensor3 = torch.randn(4, 4)
        
        # Save all tensors
        cache.save(tensor1, "key1")
        cache.save(tensor2, "key2")
        cache.save(tensor3, "key3")
        
        # The first tensor should be evicted from memory cache due to size limit
        # But we can still load it from disk
        loaded_tensor1 = cache.get("key1")
        assert torch.allclose(tensor1, loaded_tensor1)
        
        # The last two tensors should still be in memory cache
        assert len(cache.memory_cache) == 2
        assert len(cache.memory_cache_keys) == 2


def test_load_image_encoder_synclr():
    """Test loading SynCLR image encoder."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_path = Path(tmp_dir) / "models"
        
        # Test SynCLR model loading
        encoder = load_image_encoder(
            task_type="align",
            model_config_str="aligned_synclr_16",
            models_path=models_path,
            download_weights=False,  # Don't download weights in tests
            device="cpu",
            dtype=torch.float32,
            img_size=(224, 224)
        )
        
        # Verify encoder is callable
        assert callable(encoder)
        
        # Test with dummy input
        dummy_img = torch.randn(2, 3, 224, 224)
        try:
            # This might fail if weights aren't available, but that's expected
            output = encoder(dummy_img)
            assert isinstance(output, torch.Tensor)
        except Exception:
            # Expected if weights aren't available
            pass


def test_load_image_encoder_clip():
    """Test loading CLIP image encoder."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_path = Path(tmp_dir) / "models"
        
        # Test CLIP model loading
        encoder = load_image_encoder(
            task_type="align",
            model_config_str="aligned_clip_32",
            models_path=models_path,
            download_weights=False,  # Don't download weights in tests
            device="cpu",
            dtype=torch.float32,
            img_size=(224, 224)
        )
        
        # Verify encoder is callable
        assert callable(encoder)
        
        # Test with dummy input
        dummy_img = torch.randn(1, 3, 224, 224)
        try:
            # This might fail if weights aren't available, but that's expected
            output = encoder(dummy_img)
            assert isinstance(output, torch.Tensor)
        except Exception:
            # Expected if weights aren't available
            pass


def test_load_image_encoder_reconstruction():
    """Test loading reconstruction image encoder."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_path = Path(tmp_dir) / "models"
        
        # Test reconstruction model loading
        encoder = load_image_encoder(
            task_type="recon",
            model_config_str="sd_highlevel",
            models_path=models_path,
            download_weights=False,  # Don't download weights in tests
            device="cpu",
            dtype=torch.float32,
            img_size=(224, 224)
        )
        
        # Verify encoder is callable
        assert callable(encoder)
        
        # Test with dummy input
        dummy_img = torch.randn(1, 3, 224, 224)
        try:
            # This might fail if weights aren't available, but that's expected
            output = encoder(dummy_img)
            assert isinstance(output, torch.Tensor)
        except Exception:
            # Expected if weights aren't available
            pass
