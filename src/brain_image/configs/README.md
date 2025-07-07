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

## Configuration Structure

### Data Configuration (`data_config.yaml`)
Contains all parameters for EEG dataset loading:
- Data paths and directories
- Batch sizes for train/val/test
- Data loading parameters (shuffle, limits, workers)
- Subject list and image counts per concept

### EEG Encoder Configuration (`eeg_encoder_config.yaml`)
Contains parameters for the EEG encoder model:
- Embedding dimensions
- Convolutional layer parameters
- Dropout settings

### NICE Model Configuration (`nice_config.yaml`)
Contains parameters for the NICE model:
- Model architecture parameters
- Learning rate settings
- Training schedule (warmup, epochs)
- Temperature and other model-specific parameters

### Trainer Configuration (`trainer_config.yaml`)
Complete training configuration that includes:
- Training settings (epochs, workers, precision)
- Logging and checkpointing settings
- Model configuration (references `nice_config.yaml`)
- Dataset configuration (references `data_config.yaml`)

## Usage Examples

### Training a NICE Model
```bash
python scripts/train_nice.py
```

### Generating Embeddings
```bash
python scripts/gen_embeddings.py
```

### Evaluating a Model
```bash
python scripts/evaluate_nice.py
```

### Downloading Models
```bash
python scripts/download_models.py
```

## Hydra Composition

The configuration system uses Hydra's composition feature to combine configurations. For example, `train_nice.yaml` uses:

```yaml
defaults:
  - trainer_config
  - _self_
```

This loads the `trainer_config.yaml` file and then applies any overrides from `train_nice.yaml`.

## Customizing Configurations

To customize configurations for your specific use case:

1. **Modify existing configs**: Edit the appropriate YAML file
2. **Create new configs**: Create a new YAML file and reference it in your script
3. **Override via command line**: Use Hydra's command-line override syntax:
   ```bash
   python scripts/train_nice.py trainer_config.num_epochs=50
   ```

## Configuration Classes

The configurations correspond to these Python classes:
- `EEGDatasetConfig` - Data loading configuration
- `EEGEncoderConfig` - EEG encoder model configuration  
- `NICEConfig` - NICE model configuration
- `NICETrainerConfig` - Complete trainer configuration
- `EmbeddingGenerationConfig` - Embedding generation configuration
- `DownloadModelsConfig` - Model download configuration
- `EvaluateNiceConfig` - Model evaluation configuration
- `TrainNICEConfig` - Training configuration wrapper 