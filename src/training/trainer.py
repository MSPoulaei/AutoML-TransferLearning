from abc import ABC, abstractmethod

from src.models import DatasetInfo, TrainingConfig, TrainingResult


class Trainer(ABC):
    """Abstract base class for trainers."""

    @abstractmethod
    async def train(
        self,
        config: TrainingConfig,
        dataset_info: DatasetInfo,
    ) -> TrainingResult:
        """
        Train a model with the given configuration.

        Args:
            config: Training configuration
            dataset_info: Dataset information

        Returns:
            TrainingResult with metrics and history
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass
