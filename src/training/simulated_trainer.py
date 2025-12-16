import asyncio
import random
import time
from typing import Optional

import numpy as np

from src.models import (
    DatasetInfo,
    TrainingConfig,
    TrainingResult,
    FinetuningType,
)
from src.utils import get_logger
from .trainer import Trainer

logger = get_logger(__name__)


class SimulatedTrainer(Trainer):
    """
    Simulated trainer for testing and demonstration.

    Generates realistic training curves and metrics without actual training.
    """

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed
        if seed:
            random.seed(seed)
            np.random.seed(seed)

    async def train(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
    ) -> TrainingResult:
        """Simulate a training run."""
        logger.info(f"[SIMULATION] Starting training with {config.backbone.full_name}")

        start_time = time.time()

        # Calculate base accuracy based on configuration
        base_accuracy = self._calculate_base_accuracy(config, dataset_info)

        # Simulate training curves
        epochs = config.epochs
        train_loss_history = []
        val_loss_history = []
        metric_history = []

        current_train_loss = 2.5
        current_val_loss = 2.5
        current_metric = 0.1
        best_metric = 0
        best_epoch = 1

        no_improvement_count = 0
        stopped_early = False

        for epoch in range(1, epochs + 1):
            # Simulate training progress
            progress = epoch / epochs

            # Learning curve simulation
            lr_factor = self._get_lr_factor(config, progress)

            # Train loss decreases with noise
            train_noise = np.random.normal(0, 0.02)
            current_train_loss *= (0.95 - 0.1 * lr_factor) + train_noise
            current_train_loss = max(0.01, current_train_loss)

            # Val loss with potential overfitting
            overfit_factor = self._calculate_overfit_factor(
                config, dataset_info, progress
            )
            val_noise = np.random.normal(0, 0.03)
            current_val_loss = current_train_loss * (1 + overfit_factor) + val_noise
            current_val_loss = max(0.01, current_val_loss)

            # Metric improvement
            metric_improvement = (base_accuracy - current_metric) * 0.15 * lr_factor
            metric_noise = np.random.normal(0, 0.01)
            current_metric = min(
                base_accuracy, current_metric + metric_improvement + metric_noise
            )
            current_metric = max(0, min(1, current_metric))

            train_loss_history.append(current_train_loss)
            val_loss_history.append(current_val_loss)
            metric_history.append(current_metric)

            # Track best
            if current_metric > best_metric:
                best_metric = current_metric
                best_epoch = epoch
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            # Early stopping
            if config.early_stopping and no_improvement_count >= config.patience:
                stopped_early = True
                logger.info(f"[SIMULATION] Early stopping at epoch {epoch}")
                break

            # Simulate training time
            await asyncio.sleep(0.01)  # Minimal delay for async

        training_time = time.time() - start_time

        # Add simulated training overhead
        simulated_time = self._estimate_training_time(
            config, dataset_info, len(train_loss_history)
        )

        result = TrainingResult(
            config=config,
            train_loss=train_loss_history[-1],
            val_loss=val_loss_history[-1],
            primary_metric_value=best_metric,
            primary_metric_name=dataset_info.primary_metric,
            secondary_metrics={},
            epochs_trained=len(train_loss_history),
            best_epoch=best_epoch,
            training_time_seconds=simulated_time,
            stopped_early=stopped_early,
            train_loss_history=train_loss_history,
            val_loss_history=val_loss_history,
            metric_history=metric_history,
            model_size_mb=config.backbone.estimated_memory_mb,
        )

        logger.info(
            f"[SIMULATION] Training completed: "
            f"{dataset_info.primary_metric.value}={best_metric:.4f}"
        )

        return result

    def _calculate_base_accuracy(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
    ) -> float:
        """Calculate expected base accuracy for the configuration."""
        # Base accuracy depends on backbone capacity
        backbone_scores = {
            "resnet18": 0.75,
            "resnet34": 0.78,
            "resnet50": 0.82,
            "resnet101": 0.84,
            "resnet152": 0.85,
            "efficientnet_b0": 0.77,
            "efficientnet_b1": 0.79,
            "efficientnet_b2": 0.81,
            "efficientnet_b3": 0.83,
            "efficientnet_b4": 0.85,
            "efficientnet_b5": 0.86,
            "convnext_tiny": 0.80,
            "convnext_small": 0.82,
            "convnext_base": 0.84,
            "mobilenetv3_small_100": 0.70,
            "mobilenetv3_large_100": 0.75,
        }

        base = backbone_scores.get(config.backbone.variant, 0.78)

        # Adjust for dataset size
        if dataset_info.num_samples < 500:
            base *= 0.85
        elif dataset_info.num_samples < 1000:
            base *= 0.92
        elif dataset_info.num_samples > 10000:
            base *= 1.05

        # Adjust for strategy
        if config.strategy.strategy_type == FinetuningType.HEAD_ONLY:
            if dataset_info.num_samples < 1000:
                base *= 1.02  # Better for small datasets
            else:
                base *= 0.95  # Worse for large datasets
        elif config.strategy.strategy_type == FinetuningType.FULL_FINETUNING:
            if dataset_info.num_samples < 1000:
                base *= 0.90  # Risk of overfitting
            else:
                base *= 1.03  # Better for large datasets

        # Adjust for number of classes
        if dataset_info.num_classes > 100:
            base *= 0.92

        # Add randomness
        base += np.random.normal(0, 0.03)

        return min(0.98, max(0.5, base))

    def _get_lr_factor(self, config: TrainingConfig, progress: float) -> float:
        """Get learning rate factor based on scheduler."""
        if config.scheduler == "cosine":
            return 0.5 * (1 + np.cos(np.pi * progress))
        elif config.scheduler == "step":
            return 0.1 ** int(progress * 3)
        return 1.0

    def _calculate_overfit_factor(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
        progress: float,
    ) -> float:
        """Calculate overfitting factor."""
        factor = 0.0

        # More overfitting with small datasets
        if dataset_info.num_samples < 1000:
            factor += 0.1 * progress

        # Full finetuning more prone to overfitting
        if config.strategy.strategy_type == FinetuningType.FULL_FINETUNING:
            factor += 0.05 * progress

        # Regularization helps
        factor -= config.dropout * 0.1
        factor -= config.label_smoothing * 0.05

        return max(0, factor)

    def _estimate_training_time(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
        epochs_trained: int,
    ) -> float:
        """Estimate realistic training time in seconds."""
        # Base time per epoch in seconds
        base_time = 30

        # Adjust for dataset size
        batches_per_epoch = dataset_info.num_samples // config.batch_size
        base_time = batches_per_epoch * 0.1  # 100ms per batch

        # Adjust for model size
        model_factor = config.backbone.estimated_memory_mb / 1024
        base_time *= model_factor

        return base_time * epochs_trained

    def cleanup(self):
        """No cleanup needed for simulation."""
        pass
