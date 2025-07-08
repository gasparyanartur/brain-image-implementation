# Configuration Files

This directory contains all the configuration files for the brain-image-implementation project. The configurations are organized in a modular way to promote reusability and maintainability.

## Configuration Files Overview

### Core Configuration Files

- **`data_config.yaml`** - EEG dataset configuration with all data loading parameters
- **`eeg_encoder_config.yaml`** - EEG encoder model configuration
- **`nice_config.yaml`** - NICE model configuration with training parameters
- **`trainer_config.yaml`** - Complete trainer configuration including model and dataset configs

### Script-Specific Configuration Files

- **`download_models.yaml`** - Configuration for downloading pre-trained models
- **`gen_embeddings.yaml`** - Configuration for generating image embeddings
- **`train_nice.yaml`** - Configuration for training NICE models
- **`evaluate_nice.yaml`** - Configuration for evaluating trained NICE models
- **`data.yaml`** - Hydra composition file for data configuration

# Configuration Structure

This directory contains the configuration files for the brain-image project. The configuration is structured to be flexible and avoid overlapping fields between different components.

## Configuration Hierarchy

### 1. Script-Level Configs (`train_nice.yaml`, etc.)
- **Purpose**: Component composition and script-specific settings
- **Contains**: 
  - Script name and description
  - Component composition (which trainer, model, dataset, encoder to use)
  - Script-specific behavior flags
  - **No training parameters**

### 2. Trainer Configs (`trainer/nice_trainer.yaml`, etc.)
- **Purpose**: Training-specific parameters only
- **Contains**:
  - Training parameters (epochs, batch size, etc.)
  - Model compilation settings
  - Logging and checkpointing settings
  - Device settings
  - Wandb settings
  - **No component references**

### 3. Component Configs
- **Model Configs** (`model/nice_model.yaml`): Model architecture and parameters
- **Dataset Configs** (`dataset/eeg_dataset.yaml`): Data loading and preprocessing
- **Encoder Configs** (`encoder/eeg_encoder.yaml`): Encoder-specific settings

## Key Design Principles

1. **Single Responsibility**: Each config type has one clear purpose
2. **No Duplication**: Components are referenced only in script configs
3. **Composition**: Script configs compose all components via Hydra
4. **Separation**: Training logic is separate from component configuration

## Adding New Training Scripts

To add a new training script (e.g., `train_other_model.py`):

1. **Create script config** (`train_other_model.yaml`):
```yaml
defaults:
  - dataset: other_dataset
  - model: other_model
  - trainer: other_trainer
  - encoder: other_encoder
  - _self_

script_name: train_other_model
description: "Training script for Other model"
checkpoint_path: null
resume_training: false
run_test_after_training: true
save_final_model: true
```

2. **Create trainer config** (`trainer/other_trainer.yaml`):
```yaml
run_name: other
num_epochs: 50
num_workers: 16
compile_model: true
init_weights: true
# ... other training parameters (no component references)
```

3. **Create script class** (`scripts/train_other_model.py`):
```python
class TrainOtherConfig(BaseConfig):
    script_name: str = "train_other_model"
    description: str = "Training script for Other model"
    
    dataset: Any = None
    model: Any = None
    trainer: Any = None
    encoder: Any = None
    
    checkpoint_path: str | None = None
    resume_training: bool = False
    run_test_after_training: bool = True
    save_final_model: bool = True
```

## Benefits of This Structure

1. **Clear Separation**: Script configs handle composition, trainer configs handle training
2. **No Duplication**: Components are defined once and referenced where needed
3. **Flexibility**: Easy to override specific components without duplicating config
4. **Maintainability**: Clear boundaries between different types of configuration
5. **Extensibility**: Easy to add new training scripts without modifying existing ones

## Usage Examples

### Training a NICE Model
```bash
python scripts/train_nice.py
```

### Overriding Components
```bash
# Use different dataset
python scripts/train_nice.py dataset=other_dataset

# Use different model
python scripts/train_nice.py model=other_model

# Override training parameters
python scripts/train_nice.py trainer.num_epochs=50 trainer.device=cpu
```