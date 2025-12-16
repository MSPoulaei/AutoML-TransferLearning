from typing import Any
import yaml
from pathlib import Path

from .schemas import BackboneConfig


class BackboneRegistry:
    """Registry for available backbone models."""

    _instance = None
    _backbones: dict[str, dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_backbones()
        return cls._instance

    def _load_backbones(self):
        """Load backbone configurations from YAML."""
        config_path = (
            Path(__file__).parent.parent.parent / "config" / "default_config.yaml"
        )

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self._backbones = config.get("backbones", {})

    def get_all_families(self) -> list[str]:
        """Get all backbone families."""
        return list(self._backbones.keys())

    def get_variants(self, family: str) -> list[str]:
        """Get all variants for a backbone family."""
        return self._backbones.get(family, {}).get("variants", [])

    def get_memory_estimate(self, family: str, variant: str) -> float:
        """Get memory estimate for a backbone variant."""
        family_config = self._backbones.get(family, {})
        memory_estimates = family_config.get("memory_estimates_mb", {})
        return memory_estimates.get(variant, 1024)  # Default 1GB

    def get_backbone_config(
        self,
        family: str,
        variant: str,
        pretrained: bool = True,
        input_size: tuple[int, int] = (224, 224),
    ) -> BackboneConfig:
        """Create a BackboneConfig for a backbone."""
        memory = self.get_memory_estimate(family, variant)

        # Estimate number of features based on backbone
        feature_map = {
            "resnet18": 512,
            "resnet34": 512,
            "resnet50": 2048,
            "resnet101": 2048,
            "resnet152": 2048,
            "efficientnet_b0": 1280,
            "efficientnet_b1": 1280,
            "efficientnet_b2": 1408,
            "efficientnet_b3": 1536,
            "efficientnet_b4": 1792,
            "efficientnet_b5": 2048,
            "efficientnet_b6": 2304,
            "efficientnet_b7": 2560,
            "vit_tiny_patch16_224": 192,
            "vit_small_patch16_224": 384,
            "vit_base_patch16_224": 768,
            "vit_large_patch16_224": 1024,
            "convnext_tiny": 768,
            "convnext_small": 768,
            "convnext_base": 1024,
            "convnext_large": 1536,
            "mobilenetv3_small_100": 576,
            "mobilenetv3_large_100": 960,
        }

        return BackboneConfig(
            family=family,
            variant=variant,
            pretrained=pretrained,
            input_size=input_size,
            estimated_memory_mb=memory,
            num_features=feature_map.get(variant, 2048),
        )

    def filter_by_memory(self, max_memory_mb: float) -> list[tuple[str, str]]:
        """Get all backbones that fit within memory limit."""
        valid = []
        for family, config in self._backbones.items():
            memory_estimates = config.get("memory_estimates_mb", {})
            for variant, memory in memory_estimates.items():
                if memory <= max_memory_mb:
                    valid.append((family, variant))
        return valid

    def get_recommended_for_dataset(
        self, num_classes: int, num_samples: int, domain: str, max_memory_mb: float
    ) -> list[tuple[str, str, float]]:
        """Get recommended backbones with scores based on dataset."""
        recommendations = []
        valid_backbones = self.filter_by_memory(max_memory_mb)

        for family, variant in valid_backbones:
            score = self._calculate_backbone_score(
                family, variant, num_classes, num_samples, domain
            )
            recommendations.append((family, variant, score))

        # Sort by score descending
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations

    def _calculate_backbone_score(
        self, family: str, variant: str, num_classes: int, num_samples: int, domain: str
    ) -> float:
        """Calculate suitability score for a backbone."""
        score = 0.5  # Base score

        # Dataset size considerations
        if num_samples < 1000:
            # Small dataset - prefer smaller models
            if (
                "small" in variant
                or "tiny" in variant
                or "18" in variant
                or "b0" in variant
            ):
                score += 0.2
        elif num_samples > 10000:
            # Large dataset - can use larger models
            if (
                "large" in variant
                or "101" in variant
                or "152" in variant
                or "b5" in variant
            ):
                score += 0.2

        # Class count considerations
        if num_classes > 100:
            # Many classes - prefer higher capacity
            if (
                "50" in variant
                or "101" in variant
                or "b3" in variant
                or "base" in variant
            ):
                score += 0.15

        # Domain considerations
        if domain in ["medical", "satellite"]:
            # Specialized domains - ResNet often works well
            if family == "resnet":
                score += 0.1
        elif domain == "fine_grained":
            # Fine-grained - EfficientNet often better
            if family == "efficientnet":
                score += 0.15

        return min(score, 1.0)
