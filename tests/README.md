# Tests for brain-image-implementation

This directory contains comprehensive unit tests for the `src/data.py` module using pytest.

## Test Structure

- `test_data.py` - Main test file containing all tests for the data module
- `conftest.py` - Shared pytest fixtures used across test files
- `__init__.py` - Makes the tests directory a Python package

## Test Coverage

The tests cover the following components:

### Classes
- `DataConfig` - Base configuration class
- `DataModule` - Base data module class
- `EEGDatasetConfig` - EEG-specific configuration
- `EEGDataModule` - EEG data module implementation
- `EEGDataset` - EEG dataset class

### Functions
- `prepare_datasets()` - Dataset preparation utility
- `load_image_from_path()` - Image loading utility
- `batch_load_images()` - Batch image loading utility
- `load_eeg_data()` - EEG data loading utility
- `preprocess_image()` - Image preprocessing utility
- `preprocess_eeg_data()` - EEG data preprocessing utility
- `get_image_paths()` - Image path collection utility
- `load_all_eeg_data()` - Multi-subject EEG data loading utility

## Running Tests

### Prerequisites

Install the test dependencies:
```bash
pip install -e ".[test]"
```

### Basic Test Execution

Run all tests:
```bash
python -m pytest tests/
```

Run with verbose output:
```bash
python -m pytest -v tests/
```

### Using the Test Runner Script

Use the provided test runner script for convenience:
```bash
# Run all tests
python run_tests.py

# Run with coverage
python run_tests.py --coverage

# Run only unit tests
python run_tests.py --unit-only

# Run only integration tests
python run_tests.py --integration-only

# Include slow tests
python run_tests.py --slow

# Run without verbose output
python run_tests.py --no-verbose
```

### Test Markers

The tests use pytest markers for organization:
- `@pytest.mark.unit` - Unit tests (fast, isolated)
- `@pytest.mark.integration` - Integration tests (slower, test interactions)
- `@pytest.mark.slow` - Slow tests (may take longer to run)

Run tests by marker:
```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Skip slow tests
pytest -m "not slow"
```

## Test Fixtures

The tests use several fixtures defined in `conftest.py`:

- `temp_data_dir` - Creates a temporary directory with mock data structure
- `mock_eeg_data` - Provides mock EEG data for testing
- `mock_image_latents` - Provides mock image latents for testing
- `sample_eeg_tensor` - Provides sample EEG tensor data
- `sample_image_tensor` - Provides sample image tensor data
- `device` - Provides the appropriate device (CPU/GPU) for testing

## Test Categories

### Unit Tests
- Configuration class instantiation and validation
- Utility function behavior with various inputs
- Data preprocessing functions
- Path handling and file operations

### Integration Tests
- Full dataset creation and loading
- Data module initialization and dataloader creation
- End-to-end data pipeline testing

### Mock Data
Tests use temporary files and mock data to avoid requiring real data files:
- Temporary directories are created and cleaned up automatically
- Mock EEG data with realistic shapes and types
- Mock image files and latents
- Simulated file system operations

## Coverage

The tests aim for high coverage of the data module, including:
- Happy path scenarios
- Edge cases and error conditions
- Different configuration options
- Various data types and shapes
- File system operations and error handling

## Contributing

When adding new functionality to `src/data.py`, please add corresponding tests to `test_data.py`. Follow the existing patterns:
- Use descriptive test names
- Add appropriate docstrings
- Use fixtures for common setup
- Test both success and failure cases
- Add markers for test categorization 