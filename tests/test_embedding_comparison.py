from typing import cast
import torch
import torchvision.transforms.v2 as tv2
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest

# Import the current functions
from src.brain_image.data import preprocess_image, batch_load_images, load_image_from_path
from src.brain_image.model import load_image_encoder
import dreamsim
from dreamsim.model import PerceptualModel


def create_test_image(save_path: Path, size: tuple[int, int] = (256, 256)):
    """Create a test image with values 0-255 and save it."""
    # Create a test image with values 0-255
    img_array = np.random.randint(0, 256, size + (3,), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img.save(save_path)
    return img_array


def old_embedding_method(model_config_str, models_path, img_paths, img_size=(224, 224), device="cpu", dtype=torch.float32):
    """Replicate the old working embedding method exactly."""
    device = device or "cpu"
    
    # Load the model exactly like the old version did
    try:
        # Extract model config like the old version
        from src.brain_image.model import extract_model_config
        model_config = extract_model_config("align", model_config_str)
        
        aligned_option = model_config["aligned_option"]
        model_name = model_config["model_name"]
        patch_size = model_config["patch_size"]
        normalize_option = model_config["normalize_option"]
        
        model_type = f"{model_name}_vitb{patch_size}"
        
        if aligned_option == "unaligned":
            model: PerceptualModel = PerceptualModel(
                model_type=model_type,
                normalize_embeds=normalize_option == "norm",
                stride=patch_size,
                baseline=True,
                device=device,
                load_dir=str(models_path),
            )
        else:
            model, _ = cast(tuple[PerceptualModel, PerceptualModel], dreamsim.dreamsim(
                dreamsim_type=model_type,
                normalize_embeds=normalize_option == "norm",
                device=device,
                cache_dir=str(models_path),
            ))

            model: PerceptualModel = model
    except Exception as e:
        raise Exception(f"Error loading {model_name} model: {e}")
    
    # Set up model exactly like the old version
    model.eval()
    model.to(device=device, dtype=dtype)
    model.requires_grad_(False)

    latents = []
    with torch.no_grad():
        for i in range(0, len(img_paths), 32):  # batch_size=32
            paths = img_paths[i : i + 32]
            imgs = batch_load_images(paths).to(device=device, dtype=dtype)
            imgs = preprocess_image(imgs, img_size=img_size)
            
            latent = model.embed(imgs).detach().cpu()
            latents.append(latent)

    return torch.concat(latents, dim=0)


def new_embedding_method(task_type, model_name, img_paths, img_size=(224, 224), device="cpu", dtype=torch.float32):
    """Replicate the new broken embedding method exactly."""
    # Create a temporary models path
    with tempfile.TemporaryDirectory() as temp_dir:
        models_path = Path("models")
        
        # Load the image encoder using the new method
        image_encoder = load_image_encoder(
            task_type=task_type,
            model_config_str=model_name,
            models_path=models_path,
            download_weights=False,  # Don't download for test
            device=device,
            dtype=dtype,
            img_size=img_size,
            compile=True
        )
        
        # Process images using the new method
        latents = []
        with torch.no_grad():
            for i in range(0, len(img_paths), 32):  # batch_size=32
                paths = img_paths[i : i + 32]
                imgs = batch_load_images(paths).to(device=device, dtype=dtype)
                
                # Use the new embedding function
                latent = image_encoder(imgs)
                latents.append(latent)

        return torch.concat(latents, dim=0)


def test_embedding_methods_identical():
    """Test that both embedding methods produce identical results."""
    
    models_path = Path("models")

    # Create test data
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images
        img_paths = []
        for i in range(5):  # Create 5 test images
            img_path = temp_path / f"test_img_{i}.jpg"
            create_test_image(img_path)
            img_paths.append(img_path)
        
        # Test both methods
        old_result = old_embedding_method(
            model_config_str="unaligned_synclr_16",
            models_path=models_path,
            img_paths=img_paths,
            img_size=(224, 224),
            device="cpu",
            dtype=torch.float16
        )
        
        new_result = new_embedding_method(
            task_type="align",
            model_name="unaligned_synclr_16",  # Use valid model name
            img_paths=img_paths,
            img_size=(224, 224),
            device="cpu",
            dtype=torch.float16
        )
        
        # Compare results
        assert old_result.shape == new_result.shape, f"Shape mismatch: old={old_result.shape}, new={new_result.shape}"
        
        # Check if they're identical
        assert torch.allclose(old_result, new_result, atol=1e-6), f"Results differ by max {torch.max(torch.abs(old_result - new_result))}"



def test_preprocessing_pipeline():
    """Test that preprocessing pipeline produces expected results."""
    
    # Create test image tensor with values 0-255
    test_img = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8).float()
    
    # Test preprocessing
    result = preprocess_image(test_img, [224, 224])
    
    # Assertions
    assert result.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224), got {result.shape}"
    assert result.dtype == torch.float32, f"Expected dtype torch.float32, got {result.dtype}"
    # Note: preprocessing can produce values slightly outside [0,1] due to interpolation
    assert result.min() >= -0.2, f"Expected min >= -0.2, got {result.min()}"
    assert result.max() <= 1.2, f"Expected max <= 1.2, got {result.max()}"


def test_core_differences():
    """Test that the core preprocessing sequences produce identical results."""
    
    # Create test image tensor with values 0-255
    test_img = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8).float()
    
    device = "cpu"
    dtype = torch.float32
    
    # Test the exact sequence from the old method
    old_sequence = test_img.to(device=device, dtype=dtype)
    old_sequence = preprocess_image(old_sequence, [224, 224])
    
    # Test the exact sequence from the new method
    new_sequence = preprocess_image(test_img, [224, 224])
    new_sequence = new_sequence.to(device=device, dtype=dtype)
    
    # Compare the sequences
    assert torch.allclose(old_sequence, new_sequence, atol=1e-6), \
        f"Old and new sequences produce different results! Max difference: {torch.max(torch.abs(old_sequence - new_sequence))}"


def test_image_loading_pipeline():
    """Test that image loading pipeline works correctly."""
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test images
        img_paths = []
        for i in range(3):
            img_path = temp_path / f"test_img_{i}.jpg"
            create_test_image(img_path)
            img_paths.append(img_path)
        
        # Test batch loading
        batch = batch_load_images(img_paths)
        
        # Assertions
        assert batch.shape == (3, 3, 256, 256), f"Expected shape (3, 3, 256, 256), got {batch.shape}"
        assert batch.dtype == torch.uint8, f"Expected dtype torch.uint8, got {batch.dtype}"
        assert batch.min() >= 0, f"Expected min >= 0, got {batch.min()}"
        assert batch.max() <= 255, f"Expected max <= 255, got {batch.max()}"


def test_preprocessing_consistency():
    """Test that preprocessing is consistent across different input sizes."""
    
    # Test different input sizes
    test_sizes = [(128, 128), (256, 256), (512, 512)]
    target_size = [224, 224]
    
    for input_size in test_sizes:
        test_img = torch.randint(0, 256, (1, 3) + input_size, dtype=torch.uint8).float()
        result = preprocess_image(test_img, target_size)
        
        # Assertions
        assert result.shape == (1, 3, 224, 224), f"Expected shape (1, 3, 224, 224) for input {input_size}, got {result.shape}"
        assert result.dtype == torch.float32, f"Expected dtype torch.float32, got {result.dtype}"
        # Note: preprocessing can produce values slightly outside [0,1] due to interpolation
        assert result.min() >= -0.2, f"Expected min >= -0.2, got {result.min()}"
        assert result.max() <= 1.2, f"Expected max <= 1.2, got {result.max()}" 