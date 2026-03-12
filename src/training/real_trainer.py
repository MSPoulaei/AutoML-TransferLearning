import asyncio
import os
import time
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from torchvision import transforms, datasets
import timm
from tqdm import tqdm

try:
    from peft import get_peft_model, LoraConfig

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table

    FVCORE_AVAILABLE = True
except ImportError:
    FVCORE_AVAILABLE = False
    logger = None  # Will be set later

from src.models import (
    DatasetInfo,
    TrainingConfig,
    TrainingResult,
    FinetuningType,
)
from src.utils import (
    get_logger,
    get_gpu_memory_info,
    estimate_flops,
    estimate_inference_flops,
)
from .trainer import Trainer

logger = get_logger(__name__)

# LoRA target modules per backbone family.
# peft matches these against the *last* component of each module's dotted name.
_LORA_TARGET_MODULES: dict[str, list[str]] = {
    # Transformer families — target attention projections + MLP linear layers
    "vit": ["qkv", "proj", "fc1", "fc2"],
    "deit": ["qkv", "proj", "fc1", "fc2"],
    "swin_transformer": ["qkv", "proj", "fc1", "fc2"],
    # ConvNeXt uses nn.Linear for its MLP blocks (pwconv1/2)
    "convnext": ["pwconv1", "pwconv2"],
    # Pure CNN families — only the classification head is Linear
    "resnet": ["fc"],
    "efficientnet": ["classifier"],
    "mobilenet": ["classifier"],
    "regnet": ["fc"],
    "densenet": ["classifier"],
}

# Head module names to keep fully trainable alongside LoRA weights (modules_to_save)
_HEAD_MODULE_NAMES = ["head", "fc", "classifier"]


class BottleneckAdapter(nn.Module):
    """Residual bottleneck adapter: LayerNorm → down → GELU → drop → up → residual.

    Initialised near-zero so the adapter starts as an approximate identity,
    preserving the pretrained backbone's representations at the start of training.
    """

    def __init__(self, in_dim: int, bottleneck_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.down = nn.Linear(in_dim, bottleneck_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, in_dim)

        # Near-zero init so adapter ≈ identity at t=0
        nn.init.normal_(self.down.weight, std=1e-3)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(self.drop(self.act(self.down(self.norm(x)))))


class _AdapterModel(nn.Module):
    """Wraps a feature-only timm backbone with a bottleneck adapter and a new head.

    The backbone is frozen; only adapter + head are trained.
    Works for both CNN (B, C, H, W output) and transformer (B, N, C output) backbones.
    """

    def __init__(
        self,
        backbone: nn.Module,
        adapter: BottleneckAdapter,
        head: nn.Linear,
    ):
        super().__init__()
        self.backbone = backbone
        self.adapter = adapter
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)  # backbone reset to num_classes=0 returns pooled features
        features = self.adapter(features)
        return self.head(features)


class RealTrainer(Trainer):
    """
    Real PyTorch trainer for actual model training.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        device: Optional[str] = None,
        num_workers: int = 4,
    ):
        self.data_dir = data_dir
        self.num_workers = num_workers

        # Set device
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        logger.info(f"RealTrainer initialized with device: {self.device}")

        # Mixed-precision (AMP) — only effective on CUDA; no-op on CPU/MPS
        self._use_amp = self.device.type == "cuda"
        self._scaler = GradScaler(enabled=self._use_amp)
        if self._use_amp:
            logger.info("AMP (mixed precision) enabled")

        if not FVCORE_AVAILABLE:
            logger.warning(
                "fvcore library not available. FLOPS profiling will use estimation. "
                "Install with: pip install fvcore"
            )

        self.model = None
        self.optimizer = None
        self.scheduler = None

        # Cache for FLOPs profiling
        self._flops_per_sample_forward = None
        self._flops_profiled = False

    async def train(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
    ) -> TrainingResult:
        """Execute real training with PyTorch."""
        logger.info(f"Starting real training with {config.backbone.full_name}")

        start_time = time.time()

        # Log GPU memory before training
        gpu_info = get_gpu_memory_info()
        if gpu_info["available"]:
            logger.info(f"GPU Memory: {gpu_info['free_gb']:.2f}GB free")

        # Create data loaders
        train_loader, val_loader = self._create_dataloaders(config, dataset_info)

        # Create model
        self.model = self._create_model(config, dataset_info)
        self.model = self.model.to(self.device)

        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model total params: {total_params:,}")

        # Profile FLOPs once before training (cached for efficiency)
        self._profile_flops_once(train_loader, config)

        # Apply fine-tuning strategy (may replace self.model with a wrapper)
        self._apply_finetuning_strategy(config)

        # Re-move to device in case _apply_finetuning_strategy replaced self.model
        # (e.g., _AdapterModel wraps the backbone with new adapter + head on CPU)
        self.model = self.model.to(self.device)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"Trainable params after {config.strategy.strategy_type.value}: "
            f"{trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / max(total_params, 1):.1f}%)"
        )

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer(config)
        self.scheduler = self._create_scheduler(config, len(train_loader))

        # Loss function — use class-weighted loss for highly imbalanced datasets
        if dataset_info.class_balance == "highly_imbalanced":
            try:
                targets = [int(train_loader.dataset[i][1]) for i in range(len(train_loader.dataset))]
                target_tensor = torch.tensor(targets)
                num_classes = dataset_info.num_classes
                counts = torch.bincount(target_tensor, minlength=num_classes).float().clamp(min=1)
                class_weights = (counts.sum() / (num_classes * counts)).to(self.device)
                criterion = nn.CrossEntropyLoss(
                    weight=class_weights, label_smoothing=config.label_smoothing
                )
                logger.info("Using class-weighted CrossEntropyLoss for highly_imbalanced dataset")
            except Exception as e:
                logger.warning(f"Could not compute class weights: {e}. Using standard loss.")
                criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        else:
            criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        # Training history
        train_loss_history = []
        val_loss_history = []
        metric_history = []

        best_metric = 0.0
        best_epoch = 0
        no_improvement = 0
        epochs_trained = 0
        stopped_early = False

        # Training loop
        for epoch in range(1, config.epochs + 1):
            epochs_trained = epoch
            epoch_start = time.time()

            # Train epoch
            train_loss = await self._train_epoch(
                train_loader, criterion, epoch, config.epochs
            )

            # Validate
            val_loss, metric = await self._validate_epoch(
                val_loader, criterion, dataset_info
            )

            epoch_elapsed = time.time() - epoch_start
            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            metric_history.append(metric)

            gpu_mem_str = ""
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(self.device) / 1024**3
                reserved = torch.cuda.memory_reserved(self.device) / 1024**3
                gpu_mem_str = f", GPU mem: {allocated:.2f}/{reserved:.2f}GB alloc/reserved"

            logger.info(
                f"Epoch {epoch}/{config.epochs} - "
                f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
                f"{dataset_info.primary_metric.value}={metric:.4f}, "
                f"time={epoch_elapsed:.1f}s{gpu_mem_str}"
            )

            # Track best
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch
                no_improvement = 0
                # Save best model
                self._save_checkpoint(epoch, metric, config)
            else:
                no_improvement += 1

            # Early stopping
            if config.early_stopping and no_improvement >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                stopped_early = True
                break

            # Step scheduler
            if self.scheduler:
                self.scheduler.step()

            # Handle gradual unfreezing
            if config.strategy.strategy_type == FinetuningType.GRADUAL_UNFREEZING:
                unfreeze_epochs = config.strategy.gradual_unfreeze_epochs or 5
                if epoch % unfreeze_epochs == 0:
                    self._unfreeze_next_layer()

        training_time = time.time() - start_time

        # Get model size
        model_size = self._get_model_size()

        # Get model FLOPs using cached profiling results
        # Training FLOPs: forward + backward (≈3x forward pass)
        flops_per_epoch = self._get_training_flops(len(train_loader.dataset))
        total_flops = flops_per_epoch * epochs_trained

        # Inference FLOPs: forward pass only
        inference_flops_per_epoch = self._get_inference_flops(len(val_loader.dataset))
        inference_flops = inference_flops_per_epoch * epochs_trained

        inference_flops_per_sample = (
            inference_flops / (len(val_loader.dataset) * epochs_trained)
            if epochs_trained > 0 and len(val_loader.dataset) > 0
            else 0
        )

        result = TrainingResult(
            config=config,
            train_loss=train_loss_history[-1],
            val_loss=val_loss_history[-1],
            primary_metric_value=best_metric,
            primary_metric_name=dataset_info.primary_metric,
            secondary_metrics={},
            epochs_trained=epochs_trained,
            best_epoch=best_epoch,
            training_time_seconds=training_time,
            stopped_early=stopped_early,
            total_flops=total_flops,
            flops_per_epoch=flops_per_epoch,
            inference_flops=inference_flops,
            inference_flops_per_sample=inference_flops_per_sample,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            metric_history=metric_history,
            model_size_mb=model_size,
        )

        logger.info(
            f"Training completed: {dataset_info.primary_metric.value}={best_metric:.4f}, "
            f"Time: {training_time:.1f}s, Training FLOPs: {total_flops:.2e}, "
            f"Inference FLOPs: {inference_flops:.2e}"
        )

        return result

    def _create_dataloaders(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
    ) -> tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""

        # Define transforms
        train_transform = transforms.Compose(
            [
                transforms.Resize(config.backbone.input_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.Resize(config.backbone.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # Try loading from torchvision datasets first if dataset_name is provided
        if dataset_info.dataset_name:
            train_dataset = self._load_torchvision_dataset(
                dataset_info.dataset_name, train=True, transform=train_transform
            )
            val_dataset = self._load_torchvision_dataset(
                dataset_info.dataset_name, train=False, transform=val_transform
            )

            if train_dataset is not None and val_dataset is not None:
                logger.info(
                    f"Loaded {dataset_info.dataset_name}: {len(train_dataset)} train, {len(val_dataset)} val"
                )
            else:
                logger.warning(
                    f"Failed to load {dataset_info.dataset_name}, falling back to directory loading"
                )
                train_dataset = None
                val_dataset = None
        else:
            train_dataset = None
            val_dataset = None

        # Fall back to directory-based loading
        if train_dataset is None or val_dataset is None:
            if self.data_dir and os.path.exists(self.data_dir):
                # Load from directory
                full_dataset = datasets.ImageFolder(
                    root=self.data_dir, transform=train_transform
                )

                # Split into train/val
                val_size = int(len(full_dataset) * dataset_info.validation_split)
                train_size = len(full_dataset) - val_size

                train_dataset, val_dataset = random_split(
                    full_dataset, [train_size, val_size]
                )

                # Apply val transform to validation set
                val_dataset.dataset.transform = val_transform
            else:
                # Create synthetic dataset for testing
                logger.warning("No data directory provided, using synthetic data")
                train_dataset = self._create_synthetic_dataset(
                    dataset_info, train_transform, is_train=True
                )
                val_dataset = self._create_synthetic_dataset(
                    dataset_info, val_transform, is_train=False
                )

        pin = self.device.type == "cuda"

        # Weighted sampler for imbalanced datasets
        sampler = None
        if dataset_info.class_balance in ("slightly_imbalanced", "highly_imbalanced"):
            sampler = self._build_weighted_sampler(train_dataset)
            if sampler is not None:
                logger.info(
                    f"Using WeightedRandomSampler for {dataset_info.class_balance} dataset"
                )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(sampler is None),  # mutually exclusive with sampler
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=pin,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=pin,
        )

        return train_loader, val_loader

    def _build_weighted_sampler(self, dataset) -> Optional[WeightedRandomSampler]:
        """Build a WeightedRandomSampler that up-samples minority classes."""
        try:
            targets = [int(dataset[i][1]) for i in range(len(dataset))]
            target_tensor = torch.tensor(targets)
            num_classes = int(target_tensor.max().item()) + 1
            class_counts = torch.bincount(target_tensor, minlength=num_classes).float()
            # Avoid division by zero for missing classes
            class_counts = class_counts.clamp(min=1)
            class_weights = 1.0 / class_counts
            sample_weights = class_weights[target_tensor]
            return WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True,
            )
        except Exception as e:
            logger.warning(f"Could not build weighted sampler: {e}. Using uniform shuffle.")
            return None

    def _load_torchvision_dataset(
        self,
        dataset_name: str,
        train: bool,
        transform: transforms.Compose,
    ):
        """
        Load a torchvision dataset by name.

        Args:
            dataset_name: Name of the dataset (e.g., 'cifar10', 'mnist')
            train: Whether to load training or test set
            transform: Transform to apply to images

        Returns:
            Dataset instance or None if not found
        """
        dataset_classes = {
            "cifar10": datasets.CIFAR10,
            "cifar100": datasets.CIFAR100,
            "mnist": datasets.MNIST,
            "fashion_mnist": datasets.FashionMNIST,
            "svhn": datasets.SVHN,
        }

        dataset_name_lower = dataset_name.lower()

        if dataset_name_lower not in dataset_classes:
            return None

        try:
            dataset_class = dataset_classes[dataset_name_lower]
            data_root = self.data_dir if self.data_dir else "./data"

            # SVHN has different parameter names
            if dataset_name_lower == "svhn":
                split = "train" if train else "test"
                dataset = dataset_class(
                    root=data_root, split=split, transform=transform, download=True
                )
            else:
                dataset = dataset_class(
                    root=data_root, train=train, transform=transform, download=True
                )

            return dataset

        except Exception as e:
            logger.error(f"Failed to load {dataset_name}: {e}")
            return None

    def _create_synthetic_dataset(
        self,
        dataset_info: DatasetInfo,
        transform: transforms.Compose,
        is_train: bool = True,
    ):
        """Create a synthetic dataset for testing."""
        from torch.utils.data import TensorDataset

        num_samples = int(
            dataset_info.num_samples
            * (
                1 - dataset_info.validation_split
                if is_train
                else dataset_info.validation_split
            )
        )

        # Create random images and labels
        images = torch.randn(
            num_samples,
            dataset_info.num_channels,
            dataset_info.image_size[0],
            dataset_info.image_size[1],
        )
        labels = torch.randint(0, dataset_info.num_classes, (num_samples,))

        return TensorDataset(images, labels)

    def _create_model(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
    ) -> nn.Module:
        """Create model using timm library."""

        model = timm.create_model(
            config.backbone.variant,
            pretrained=config.backbone.pretrained,
            num_classes=dataset_info.num_classes,
            drop_rate=config.dropout,
        )

        logger.info(
            f"Created model: {config.backbone.variant}, "
            f"classes={dataset_info.num_classes}"
        )

        return model

    def _apply_finetuning_strategy(self, config: TrainingConfig):
        """Apply fine-tuning strategy to model.

        For LORA and ADAPTER, self.model may be replaced with a wrapped version.
        All other code (optimizer, training loop) operates on self.model after this call.
        """
        if config.strategy.strategy_type == FinetuningType.LORA:
            self._apply_lora(config)
            return
        elif config.strategy.strategy_type == FinetuningType.ADAPTER:
            self._apply_adapter(config)
            return

        if config.strategy.strategy_type == FinetuningType.HEAD_ONLY:
            self._apply_head_only()

        elif config.strategy.strategy_type == FinetuningType.FULL_FINETUNING:
            # All parameters trainable
            for param in self.model.parameters():
                param.requires_grad = True

            logger.info("Applied FULL_FINETUNING strategy: all layers trainable")

        elif config.strategy.strategy_type == FinetuningType.GRADUAL_UNFREEZING:
            # Start with only head trainable
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "fc" not in name and "head" not in name:
                    param.requires_grad = False

            # Store layer groups for unfreezing
            self._layer_groups = self._get_layer_groups()
            self._current_unfrozen = 0

            logger.info("Applied GRADUAL_UNFREEZING strategy: starting with head only")

        elif config.strategy.strategy_type == FinetuningType.DISCRIMINATIVE_LR:
            # All trainable, but different LRs applied in optimizer
            for param in self.model.parameters():
                param.requires_grad = True

            logger.info(
                "Applied DISCRIMINATIVE_LR strategy: all layers with decaying LR"
            )

        # Optionally freeze batch norm
        if config.strategy.freeze_bn:
            for module in self.model.modules():
                if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False

            logger.info("Batch normalization layers frozen")

    def _get_layer_groups(self) -> list[list[nn.Parameter]]:
        """Get parameter groups for gradual unfreezing."""
        groups = []
        current_group = []

        for name, param in self.model.named_parameters():
            current_group.append(param)

            # Start new group at layer boundaries
            if "layer" in name and "0.conv" in name:
                if current_group:
                    groups.append(current_group)
                    current_group = []

        if current_group:
            groups.append(current_group)

        return groups

    # ------------------------------------------------------------------
    # PEFT strategies
    # ------------------------------------------------------------------

    def _apply_lora(self, config: TrainingConfig) -> None:
        """Apply LoRA via the peft library.

        Target modules are selected per backbone family so LoRA is applied to
        the most semantically rich linear projections available. Falls back to
        HEAD_ONLY if peft is not installed.
        """
        if not PEFT_AVAILABLE:
            logger.warning(
                "peft not installed — falling back to head_only. "
                "Install with: pip install peft"
            )
            self._apply_head_only()
            return

        family = config.backbone.family
        target_modules = _LORA_TARGET_MODULES.get(family, ["fc", "classifier", "head"])

        lora_cfg = LoraConfig(
            r=config.strategy.lora_rank,
            lora_alpha=config.strategy.lora_alpha,
            target_modules=target_modules,
            lora_dropout=config.strategy.lora_dropout,
            bias="none",
            # Keep head/classifier fully trainable (not LoRA-adapted)
            modules_to_save=_HEAD_MODULE_NAMES,
        )

        try:
            self.model = get_peft_model(self.model, lora_cfg)
            trainable, total = self.model.get_nb_trainable_parameters()
            pct = 100.0 * trainable / max(total, 1)
            logger.info(
                f"LoRA applied (rank={config.strategy.lora_rank}, "
                f"alpha={config.strategy.lora_alpha}, "
                f"targets={target_modules}): "
                f"{trainable:,}/{total:,} trainable params ({pct:.2f}%)"
            )
        except Exception as e:
            logger.warning(f"LoRA application failed ({e}) — falling back to head_only")
            self._apply_head_only()

    def _apply_adapter(self, config: TrainingConfig) -> None:
        """Replace model with a frozen backbone + BottleneckAdapter + new head.

        Works universally on both CNN and transformer timm models.
        Only the adapter and head weights are trained.
        """
        num_features = getattr(self.model, "num_features", None)
        num_classes = getattr(self.model, "num_classes", None)

        if num_features is None or num_classes is None:
            logger.warning(
                "Model does not expose num_features/num_classes — "
                "falling back to head_only strategy"
            )
            self._apply_head_only()
            return

        # Remove the existing head; backbone.forward() will now return pooled features
        self.model.reset_classifier(0)

        # Freeze backbone
        for param in self.model.parameters():
            param.requires_grad = False

        adapter = BottleneckAdapter(
            in_dim=num_features,
            bottleneck_dim=config.strategy.adapter_size,
            dropout=config.dropout,
        )
        head = nn.Linear(num_features, num_classes)

        # Replace self.model with the wrapper (moves to device in train())
        self.model = _AdapterModel(self.model, adapter, head)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        pct = 100.0 * trainable / max(total, 1)
        logger.info(
            f"Adapter applied (size={config.strategy.adapter_size}): "
            f"{trainable:,}/{total:,} trainable params ({pct:.2f}%)"
        )

    def _apply_head_only(self) -> None:
        """Freeze all layers except the classification head (shared fallback)."""
        for name, param in self.model.named_parameters():
            if not any(h in name for h in ("classifier", "fc", "head")):
                param.requires_grad = False
        logger.info("Applied HEAD_ONLY strategy: backbone frozen")

    # ------------------------------------------------------------------

    def _unfreeze_next_layer(self):
        """Unfreeze the next layer group."""
        if hasattr(self, "_layer_groups") and self._current_unfrozen < len(
            self._layer_groups
        ):
            group_idx = len(self._layer_groups) - 1 - self._current_unfrozen
            for param in self._layer_groups[group_idx]:
                param.requires_grad = True

            self._current_unfrozen += 1
            logger.info(f"Unfroze layer group {self._current_unfrozen}")

    def _create_optimizer(self, config: TrainingConfig) -> optim.Optimizer:
        """Create optimizer based on configuration."""

        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        if config.strategy.strategy_type == FinetuningType.DISCRIMINATIVE_LR:
            # Create parameter groups with decaying learning rates
            param_groups = self._create_discriminative_param_groups(config)
        else:
            param_groups = [{"params": trainable_params}]

        if config.optimizer.lower() == "adamw":
            optimizer = optim.AdamW(
                param_groups,
                lr=config.strategy.learning_rate,
                weight_decay=config.strategy.weight_decay,
            )
        elif config.optimizer.lower() == "sgd":
            optimizer = optim.SGD(
                param_groups,
                lr=config.strategy.learning_rate,
                momentum=0.9,
                weight_decay=config.strategy.weight_decay,
            )
        elif config.optimizer.lower() == "adam":
            optimizer = optim.Adam(
                param_groups,
                lr=config.strategy.learning_rate,
                weight_decay=config.strategy.weight_decay,
            )
        else:
            optimizer = optim.AdamW(
                param_groups,
                lr=config.strategy.learning_rate,
                weight_decay=config.strategy.weight_decay,
            )

        return optimizer

    def _create_discriminative_param_groups(self, config: TrainingConfig) -> list[dict]:
        """Create parameter groups with discriminative learning rates."""
        param_groups = []
        lr = config.strategy.learning_rate
        decay = config.strategy.layer_lr_decay or 0.9

        # Get all named parameters
        all_params = list(self.model.named_parameters())

        # Group by layer
        current_layer = None
        current_params = []

        for name, param in all_params:
            if not param.requires_grad:
                continue

            # Detect layer change
            layer_match = None
            for i in range(10):
                if f"layer{i}" in name or f"blocks.{i}" in name:
                    layer_match = i
                    break

            if layer_match != current_layer:
                if current_params:
                    param_groups.append(
                        {
                            "params": current_params,
                            "lr": lr * (decay ** len(param_groups)),
                        }
                    )
                current_params = [param]
                current_layer = layer_match
            else:
                current_params.append(param)

        # Add remaining params (usually classifier)
        if current_params:
            param_groups.append(
                {
                    "params": current_params,
                    "lr": lr,  # Full learning rate for classifier
                }
            )

        return param_groups

    def _create_scheduler(
        self,
        config: TrainingConfig,
        steps_per_epoch: int,
    ) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""

        total_steps = config.epochs * steps_per_epoch
        warmup_steps = config.warmup_epochs * steps_per_epoch

        if config.scheduler.lower() == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.epochs,
                eta_min=config.strategy.learning_rate * 0.01,
            )
        elif config.scheduler.lower() == "step":
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.epochs // 3,
                gamma=0.1,
            )
        elif config.scheduler.lower() == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=0.5,
                patience=5,
            )
        else:
            scheduler = None

        return scheduler

    async def _train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=False,
        )

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()

            with autocast(enabled=self._use_amp):
                output = self.model(data)
                loss = criterion(output, target)

            self._scaler.scale(loss).backward()
            # Unscale before clipping so clip operates on true gradients
            self._scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self._scaler.step(self.optimizer)
            self._scaler.update()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            # Allow async operations
            if batch_idx % 10 == 0:
                await asyncio.sleep(0)

        return total_loss / num_batches

    async def _validate_epoch(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        dataset_info: DatasetInfo,
    ) -> tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                loss = criterion(output, target)

                total_loss += loss.item()

                # Calculate accuracy
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total

        # For now, return accuracy as the metric
        # Can be extended to support other metrics
        return avg_loss, accuracy

    def _save_checkpoint(
        self,
        epoch: int,
        metric: float,
        config: TrainingConfig,
    ):
        """Save model checkpoint."""
        checkpoint_dir = Path("checkpoints")
        checkpoint_dir.mkdir(exist_ok=True)

        checkpoint_path = checkpoint_dir / f"best_model_{config.backbone.variant}.pt"

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metric": metric,
                "config": config.model_dump(),
            },
            checkpoint_path,
        )

        logger.debug(f"Saved checkpoint to {checkpoint_path}")

    def _get_model_size(self) -> float:
        """Get model size in MB."""
        if self.model is None:
            return 0

        param_size = sum(
            p.nelement() * p.element_size() for p in self.model.parameters()
        )
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())

        return (param_size + buffer_size) / (1024 * 1024)

    def _profile_flops_once(
        self,
        data_loader: DataLoader,
        config: TrainingConfig,
    ) -> None:
        """
        Profile FLOPs once at the start of training and cache the result.
        This minimizes overhead by avoiding repeated profiling.
        """
        if self._flops_profiled:
            return  # Already profiled

        logger.info("Profiling model FLOPs (one-time overhead)...")
        start_time = time.time()

        if not FVCORE_AVAILABLE:
            logger.warning("fvcore not available, will use estimation")
            self._flops_per_sample_forward = None
            self._flops_profiled = True
            return

        try:
            # Get a sample batch
            sample_batch = next(iter(data_loader))[0]
            sample_input = sample_batch[:1].to(self.device)  # Single sample

            # Profile forward pass
            self.model.eval()
            with torch.no_grad():
                flops = FlopCountAnalysis(self.model, sample_input)
                self._flops_per_sample_forward = flops.total()

            self._flops_profiled = True
            elapsed = time.time() - start_time

            logger.info(
                f"FLOPS profiling completed in {elapsed:.2f}s. "
                f"Forward pass: {self._flops_per_sample_forward:,} FLOPs/sample"
            )

        except Exception as e:
            logger.warning(f"FLOPS profiling failed: {e}, will use estimation")
            self._flops_per_sample_forward = None
            self._flops_profiled = True

    def _get_training_flops(self, num_samples: int) -> float:
        """
        Get total training FLOPs using cached profiling.
        Training = forward + backward ≈ 3x forward pass.
        """
        if self._flops_per_sample_forward is not None:
            # Use profiled value (3x for forward + backward)
            return float(self._flops_per_sample_forward * num_samples * 3)
        else:
            # Fallback to estimation
            model_params = sum(p.numel() for p in self.model.parameters())
            return estimate_flops(
                num_samples=num_samples,
                batch_size=1,  # Already per-sample calculation
                epochs=1,
                model_params=model_params,
                image_size=(
                    self.model.default_cfg.get("input_size", (3, 224, 224))[1:]
                    if hasattr(self.model, "default_cfg")
                    else (224, 224)
                ),
            )

    def _get_inference_flops(self, num_samples: int) -> float:
        """
        Get total inference FLOPs using cached profiling.
        Inference = forward pass only.
        """
        if self._flops_per_sample_forward is not None:
            # Use profiled value (1x for forward only)
            return float(self._flops_per_sample_forward * num_samples)
        else:
            # Fallback to estimation
            model_params = sum(p.numel() for p in self.model.parameters())
            return estimate_inference_flops(
                num_samples=num_samples,
                batch_size=1,  # Already per-sample calculation
                model_params=model_params,
                image_size=(
                    self.model.default_cfg.get("input_size", (3, 224, 224))[1:]
                    if hasattr(self.model, "default_cfg")
                    else (224, 224)
                ),
            )

    def cleanup(self):
        """Clean up GPU memory."""
        if self.model is not None:
            del self.model
            self.model = None

        if self.optimizer is not None:
            del self.optimizer
            self.optimizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Trainer resources cleaned up")
