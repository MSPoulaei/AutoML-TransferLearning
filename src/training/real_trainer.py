import asyncio
import os
import time
from pathlib import Path
from typing import Optional, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import timm
from tqdm import tqdm

from src.models import (
    DatasetInfo,
    TrainingConfig,
    TrainingResult,
    FinetuningType,
)
from src.utils import get_logger, get_gpu_memory_info
from .trainer import Trainer

logger = get_logger(__name__)


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

        self.model = None
        self.optimizer = None
        self.scheduler = None

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

        # Apply fine-tuning strategy
        self._apply_finetuning_strategy(config)

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer(config)
        self.scheduler = self._create_scheduler(config, len(train_loader))

        # Loss function
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

            # Train epoch
            train_loss = await self._train_epoch(
                train_loader, criterion, epoch, config.epochs
            )

            # Validate
            val_loss, metric = await self._validate_epoch(
                val_loader, criterion, dataset_info
            )

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            metric_history.append(metric)

            logger.info(
                f"Epoch {epoch}/{config.epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                f"{dataset_info.primary_metric.value}: {metric:.4f}"
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
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            metric_history=metric_history,
            model_size_mb=model_size,
        )

        logger.info(
            f"Training completed: {dataset_info.primary_metric.value}={best_metric:.4f}, "
            f"Time: {training_time:.1f}s"
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

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True if self.device.type == "cuda" else False,
        )

        return train_loader, val_loader

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
        """Apply fine-tuning strategy to model."""

        if config.strategy.strategy_type == FinetuningType.HEAD_ONLY:
            # Freeze all layers except classifier
            for name, param in self.model.named_parameters():
                if "classifier" not in name and "fc" not in name and "head" not in name:
                    param.requires_grad = False

            logger.info("Applied HEAD_ONLY strategy: backbone frozen")

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
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

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
