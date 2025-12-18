# Quick Start: Using Dataset Presets

## How to Run with Famous Datasets

The system now supports famous datasets with automatic downloading and configuration!

### Step 1: View Available Datasets

```bash
python test_dataset_presets.py
```

Or once your dependencies are fixed:

```bash
python main.py list-datasets
```

### Step 2: Run with a Dataset Preset

#### CIFAR-10 (recommended for quick testing)
```bash
python main.py run --dataset cifar10 --budget 5 --simulation
```

#### Fashion-MNIST
```bash
python main.py run --dataset fashion_mnist --budget 10 --simulation
```

#### MNIST
```bash
python main.py run --dataset mnist --budget 5 --simulation
```

#### CIFAR-100 (more challenging)
```bash
python main.py run --dataset cifar100 --budget 10 --simulation
```

### Available Dataset Presets

| Name          | Classes | Samples   | Image Size | Auto-Download |
| ------------- | ------- | --------- | ---------- | ------------- |
| cifar10       | 10      | 50,000    | 32x32      | ✅ Yes         |
| cifar100      | 100     | 50,000    | 32x32      | ✅ Yes         |
| mnist         | 10      | 60,000    | 28x28      | ✅ Yes         |
| fashion_mnist | 10      | 60,000    | 28x28      | ✅ Yes         |
| svhn          | 10      | 73,257    | 32x32      | ✅ Yes         |
| imagenet      | 1000    | 1,281,167 | 224x224    | ❌ Manual      |

### Benefits

✅ **No Manual Configuration** - All parameters are preset
✅ **Automatic Download** - Datasets download automatically (except ImageNet)
✅ **No Data Directory Needed** - Downloads to `./data` by default
✅ **Standard Splits** - Train/test splits are handled automatically

### Advanced Options

You can still override some parameters:

```bash
# Change the budget
python main.py run --dataset cifar10 --budget 20 --simulation

# Change memory limit
python main.py run --dataset cifar10 --budget 5 --simulation --memory-limit 10.0

# Use real training (if GPU available)
python main.py run --dataset cifar10 --budget 5
```

### Custom Datasets (Original Functionality)

You can still use custom datasets as before:

```bash
python main.py run \
    --num-classes 10 \
    --num-samples 5000 \
    --domain natural \
    --domain-desc "My custom image dataset" \
    --data-dir /path/to/your/data \
    --budget 5 \
    --simulation
```

## What Changed?

### New CLI Arguments
- `--dataset`: Use a famous dataset preset (e.g., `cifar10`, `mnist`)

### Modified Arguments
- `--num-classes`: Now **optional** when using `--dataset`
- `--num-samples`: Now **optional** when using `--dataset`  
- `--domain-desc`: Now **optional** when using `--dataset`

### New Commands
- `list-datasets`: Show all available presets

## Technical Implementation

The system now:
1. Detects when a dataset preset is used
2. Automatically configures all dataset parameters
3. Downloads the dataset via torchvision (on first use)
4. Caches downloaded data in `./data` directory
5. Reuses cached data on subsequent runs

### File Changes
- `main.py`: Added preset logic and new command
- `src/models/schemas.py`: Added `dataset_name` field
- `src/data/dataset_loader.py`: Added torchvision loading
- `src/training/real_trainer.py`: Added auto-download support
- `docs/DATASET_PRESETS.md`: Full documentation
