from typing import Optional
import yaml
from pathlib import Path

from .schemas import FinetuningStrategy, FinetuningType


class StrategyRegistry:
    """Registry for fine-tuning strategies."""

    _instance = None
    _strategies: list[dict] = []

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_strategies()
        return cls._instance

    def _load_strategies(self):
        """Load strategies from YAML config."""
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self._strategies = config.get("finetuning_strategies", [])

    def get_all_strategies(self) -> list[str]:
        """Get all available strategy names."""
        return [s["name"] for s in self._strategies]

    def get_strategy_info(self, name: str) -> Optional[dict]:
        """Get information about a strategy."""
        for s in self._strategies:
            if s["name"] == name:
                return s
        return None

    def get_memory_multiplier(self, name: str) -> float:
        """Get memory multiplier for a strategy."""
        info = self.get_strategy_info(name)
        return info.get("memory_multiplier", 2.0) if info else 2.0

    def create_strategy(
        self,
        strategy_type: FinetuningType,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        **kwargs
    ) -> FinetuningStrategy:
        """Create a FinetuningStrategy with appropriate defaults."""
        memory_mult = self.get_memory_multiplier(strategy_type.value)

        strategy = FinetuningStrategy(
            strategy_type=strategy_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            memory_multiplier=memory_mult,
            **kwargs
        )

        # Set strategy-specific defaults
        if strategy_type == FinetuningType.GRADUAL_UNFREEZING:
            if strategy.gradual_unfreeze_epochs is None:
                strategy.gradual_unfreeze_epochs = 5
        elif strategy_type == FinetuningType.DISCRIMINATIVE_LR:
            if strategy.layer_lr_decay is None:
                strategy.layer_lr_decay = 0.9

        return strategy

    def get_recommended_for_dataset(
        self, num_samples: int, num_classes: int, domain: str
    ) -> list[tuple[FinetuningType, float]]:
        """Get recommended strategies with scores."""
        recommendations = []

        # Score each strategy
        for strategy_type in FinetuningType:
            score = self._calculate_strategy_score(
                strategy_type, num_samples, num_classes, domain
            )
            recommendations.append((strategy_type, score))

        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations

    def _calculate_strategy_score(
        self,
        strategy_type: FinetuningType,
        num_samples: int,
        num_classes: int,
        domain: str,
    ) -> float:
        """Calculate suitability score for a strategy."""
        score = 0.5

        if strategy_type == FinetuningType.HEAD_ONLY:
            # Best for small datasets or when pretrained features are highly relevant
            if num_samples < 1000:
                score += 0.3
            if domain in ["natural"]:
                score += 0.1

        elif strategy_type == FinetuningType.FULL_FINETUNING:
            # Best for large datasets or specialized domains
            if num_samples > 5000:
                score += 0.2
            if domain in ["medical", "satellite", "industrial"]:
                score += 0.2

        elif strategy_type == FinetuningType.GRADUAL_UNFREEZING:
            # Good balance for medium datasets
            if 1000 <= num_samples <= 10000:
                score += 0.25

        elif strategy_type == FinetuningType.DISCRIMINATIVE_LR:
            # Good for fine-grained or when features need adaptation
            if domain == "fine_grained":
                score += 0.2
            if num_classes > 50:
                score += 0.1

        return min(score, 1.0)
