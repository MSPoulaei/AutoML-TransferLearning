# Dataset Preset Feature - Implementation Summary

## Overview
Added support for famous datasets (CIFAR-10, CIFAR-100, MNIST, Fashion-MNIST, SVHN, ImageNet) with automatic configuration and downloading capabilities.

## Changes Made

### 1. `main.py` - Command Line Interface
- **Added `--dataset` option**: Allows users to specify a famous dataset preset
- **Made custom dataset args optional**: When using a preset, `--num-classes`, `--num-samples`, and `--domain-desc` are no longer required
- **Added `get_dataset_preset()` function**: Returns predefined configurations for famous datasets
- **Added `list-datasets` command**: Display all available dataset presets in a formatted table
- **Updated examples**: Added usage examples for both preset and custom datasets

#### Available Presets:
- `cifar10`: 10 classes, 50k samples, 32x32 images (natural domain)
- `cifar100`: 100 classes, 50k samples, 32x32 images (fine-grained domain)
- `mnist`: 10 classes, 60k samples, 28x28 images (document domain)
- `fashion_mnist`: 10 classes, 60k samples, 28x28 images (natural domain)
- `svhn`: 10 classes, 73k samples, 32x32 images (document domain)
- `imagenet`: 1000 classes, 1.2M samples, 224x224 images (natural domain)

### 2. `src/models/schemas.py` - Data Models
- **Added `dataset_name` field** to `DatasetInfo`: Optional field to store the preset name for auto-loading

### 3. `src/data/dataset_loader.py` - Dataset Loading
- **Added `get_torchvision_dataset()` function**: Helper to load datasets from torchvision
- **Updated `DatasetLoader.__init__()`**: Added optional `dataset_name` parameter
- **Updated `get_dataloaders()`**: Attempts to load from torchvision first, then falls back to directory-based loading

### 4. `src/training/real_trainer.py` - Training
- **Added `_load_torchvision_dataset()` method**: Handles automatic downloading and loading of torchvision datasets
- **Updated `_create_dataloaders()`**: Checks for `dataset_name` and attempts auto-loading before falling back to directory loading

## Usage Examples

### Using a Famous Dataset Preset
```bash
# Run with CIFAR-10
python main.py run --dataset cifar10 --budget 5 --simulation

# Run with Fashion-MNIST
python main.py run --dataset fashion_mnist --budget 10 --simulation

# List all available presets
python main.py list-datasets
```

### Using a Custom Dataset (unchanged)
```bash
python main.py run --num-classes 10 --num-samples 5000 \
    --domain natural --domain-desc "My custom dataset" \
    --budget 5 --simulation --data-dir ./my_data
```

## Benefits

1. **Easier Experimentation**: Users can quickly test with well-known datasets without manual configuration
2. **Automatic Downloading**: Famous datasets are automatically downloaded via torchvision
3. **Backward Compatible**: Existing custom dataset functionality remains unchanged
4. **Reduced Errors**: Predefined configurations eliminate common configuration mistakes
5. **Better Documentation**: Built-in descriptions for each dataset help users understand the data

## Technical Details

### Auto-Loading Flow:
1. User specifies `--dataset cifar10`
2. `get_dataset_preset()` returns configuration dictionary
3. Configuration populates all dataset parameters
4. `dataset_name` is stored in `DatasetInfo`
5. During training, `RealTrainer` detects `dataset_name`
6. Calls `_load_torchvision_dataset()` which downloads/loads the dataset
7. Falls back to directory loading if auto-loading fails

### Supported Torchvision Datasets:
- `datasets.CIFAR10`
- `datasets.CIFAR100`
- `datasets.MNIST`
- `datasets.FashionMNIST`
- `datasets.SVHN`

Note: ImageNet is listed as a preset but requires manual download due to size and licensing.

## Future Enhancements

Potential additions:
- More dataset presets (STL10, Caltech101, etc.)
- Dataset-specific augmentation strategies
- Automatic train/val split recommendations per dataset
- Support for custom dataset repositories (e.g., Hugging Face datasets)
