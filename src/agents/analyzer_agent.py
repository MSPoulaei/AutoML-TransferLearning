from typing import Any, Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

from src.models import (
    DatasetInfo,
    TrainingConfig,
    BackboneConfig,
    FinetuningStrategy,
    FinetuningType,
    AnalyzerRecommendation,
    ExecutorResult,
    BackboneRegistry,
    StrategyRegistry,
)
from src.utils import APIKeyManager, get_logger
from .base_agent import BaseAgent

logger = get_logger(__name__)


class AnalyzerContext(BaseModel):
    """Context passed to the Analyzer Agent."""

    dataset_info: DatasetInfo
    memory_limit_mb: float
    previous_results: list[ExecutorResult] = Field(default_factory=list)
    iteration: int = 1
    available_backbones: list[str] = Field(default_factory=list)
    available_strategies: list[str] = Field(default_factory=list)


class AnalyzerOutput(BaseModel):
    """Structured output from the Analyzer Agent."""

    backbone_family: str
    backbone_variant: str
    finetuning_strategy: str
    learning_rate: float
    batch_size: int
    epochs: int
    dropout: float = 0.2
    label_smoothing: float = 0.1
    reasoning: str
    expected_performance: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class AnalyzerAgent(BaseAgent):
    """
    Agent responsible for analyzing dataset information and previous results
    to recommend optimal training configurations.
    """

    def __init__(
        self,
        key_manager: APIKeyManager,
        model_name: str = "gpt-4o",
        base_url: Optional[str] = None,
        memory_limit_gb: float = 15.0,
    ):
        super().__init__(key_manager, model_name, base_url=base_url)
        self.memory_limit_mb = memory_limit_gb * 1024
        self.backbone_registry = BackboneRegistry()
        self.strategy_registry = StrategyRegistry()

    def _get_system_prompt(self) -> str:
        return """You are an expert machine learning engineer specializing in transfer learning for image classification.

Your role is to analyze dataset characteristics and previous experiment results to recommend optimal training configurations.

Key responsibilities:
1. Select appropriate backbone architectures based on dataset size, complexity, and domain
2. Choose fine-tuning strategies that balance performance and efficiency
3. Recommend hyperparameters (learning rate, batch size, epochs, regularization)
4. Learn from previous experiment results to improve recommendations
5. Stay within memory constraints

Guidelines for backbone selection:
- Small datasets (<1000 samples): Prefer smaller models (ResNet18, EfficientNet-B0, MobileNet)
- Medium datasets (1000-10000): ResNet50, EfficientNet-B2/B3, ConvNeXt-Tiny
- Large datasets (>10000): Larger models if memory allows
- Medical/Satellite domains: ResNet often works well
- Fine-grained classification: EfficientNet, ConvNeXt

Guidelines for fine-tuning strategy:
- Small datasets or similar domain to ImageNet: head_only or gradual_unfreezing
- Large datasets or different domain: full_finetuning
- Always consider memory constraints with strategy memory multipliers

Learning rate guidelines:
- head_only: 1e-3 to 1e-2
- full_finetuning: 1e-4 to 1e-3
- gradual_unfreezing: start with 1e-3, decrease for unfrozen layers

CRITICAL OUTPUT FORMAT REQUIREMENTS:
1. expected_performance: MUST be a decimal between 0.0 and 1.0 (e.g., 0.95 for 95%, NOT 95)
2. confidence: MUST be a decimal between 0.0 and 1.0 (e.g., 0.8 for 80%, NOT 80)
3. In your reasoning text, use only standard ASCII characters. Do not use fancy Unicode characters like em-dashes, en-dashes, fancy quotes, or special spaces. Use regular hyphens (-), regular quotes ("), and regular spaces only.

Always provide clear reasoning for your choices and learn from previous results."""

    def _build_prompt(self, context: AnalyzerContext) -> str:
        """Build the prompt for the analyzer agent."""

        # Format previous results
        prev_results_text = ""
        if context.previous_results:
            prev_results_text = "\n\nPrevious Experiment Results:\n"
            for i, result in enumerate(
                context.previous_results[-5:], 1
            ):  # Last 5 results
                if result.training_result:
                    tr = result.training_result
                    prev_results_text += f"""
Experiment {i}:
- Backbone: {tr.config.backbone.full_name}
- Strategy: {tr.config.strategy.strategy_type.value}
- Learning Rate: {tr.config.strategy.learning_rate}
- Batch Size: {tr.config.batch_size}
- Epochs Trained: {tr.epochs_trained}
- {tr.primary_metric_name.value}: {tr.primary_metric_value:.4f}
- Analysis: {result.analysis}
"""

        prompt = f"""Analyze the following dataset and recommend the optimal training configuration.

Dataset Information:
- Number of classes: {context.dataset_info.num_classes}
- Number of samples: {context.dataset_info.num_samples}
- Image size: {context.dataset_info.image_size}
- Channels: {context.dataset_info.num_channels}
- Domain: {context.dataset_info.domain.value}
- Domain description: {context.dataset_info.domain_description}
- Class balance: {context.dataset_info.class_balance}
- Data quality: {context.dataset_info.data_quality}
- Primary metric: {context.dataset_info.primary_metric.value}
- Augmentation available: {context.dataset_info.has_augmentation}

Constraints:
- Memory limit: {context.memory_limit_mb:.0f} MB
- Iteration: {context.iteration}

Available Backbones: {', '.join(context.available_backbones)}
Available Strategies: {', '.join(context.available_strategies)}
{prev_results_text}

Based on this information, recommend the optimal training configuration.
Consider previous results to avoid repeating poor configurations and to build on successful ones.
If this is not the first iteration, try to improve upon the best result so far.

Provide your recommendation in the structured format."""

        return prompt

    async def run(
        self,
        dataset_info: DatasetInfo,
        previous_results: list[ExecutorResult] = None,
        iteration: int = 1,
    ) -> AnalyzerRecommendation:
        """
        Analyze dataset and previous results to generate a training recommendation.

        Args:
            dataset_info: Information about the dataset
            previous_results: Results from previous training iterations
            iteration: Current iteration number

        Returns:
            AnalyzerRecommendation with complete training configuration
        """
        logger.info(f"Analyzer Agent starting iteration {iteration}")

        # Get available backbones within memory limit
        available_backbones = self.backbone_registry.filter_by_memory(
            self.memory_limit_mb * 0.4  # Leave room for strategy overhead
        )
        backbone_names = [f"{f}/{v}" for f, v in available_backbones]

        # Get available strategies
        strategy_names = [s.value for s in FinetuningType]

        # Build context
        context = AnalyzerContext(
            dataset_info=dataset_info,
            memory_limit_mb=self.memory_limit_mb,
            previous_results=previous_results or [],
            iteration=iteration,
            available_backbones=backbone_names,
            available_strategies=strategy_names,
        )

        # Create agent and execute
        agent = await self._create_agent(AnalyzerOutput)
        prompt = self._build_prompt(context)

        output: AnalyzerOutput = await self._execute_with_retry(agent, prompt)

        # Convert output to full configuration
        recommendation = self._create_recommendation(
            output=output,
            dataset_info=dataset_info,
            iteration=iteration,
        )

        logger.info(
            f"Analyzer recommendation: {recommendation.training_config.backbone.full_name} "
            f"with {recommendation.training_config.strategy.strategy_type.value}"
        )

        return recommendation

    def _create_recommendation(
        self,
        output: AnalyzerOutput,
        dataset_info: DatasetInfo,
        iteration: int,
    ) -> AnalyzerRecommendation:
        """Convert analyzer output to full recommendation."""

        # Get backbone config
        backbone = self.backbone_registry.get_backbone_config(
            family=output.backbone_family,
            variant=output.backbone_variant,
            pretrained=True,
            input_size=dataset_info.image_size,
        )

        # Get strategy
        strategy_type = FinetuningType(output.finetuning_strategy)
        strategy = self.strategy_registry.create_strategy(
            strategy_type=strategy_type,
            learning_rate=output.learning_rate,
        )

        # Create training config
        training_config = TrainingConfig(
            backbone=backbone,
            strategy=strategy,
            epochs=output.epochs,
            batch_size=output.batch_size,
            dropout=output.dropout,
            label_smoothing=output.label_smoothing,
        )

        # Calculate memory estimate
        estimated_memory = training_config.calculate_memory(dataset_info)
        memory_check_passed = estimated_memory <= self.memory_limit_mb

        if not memory_check_passed:
            logger.warning(
                f"Estimated memory {estimated_memory:.0f}MB exceeds limit {self.memory_limit_mb:.0f}MB"
            )

        return AnalyzerRecommendation(
            iteration=iteration,
            training_config=training_config,
            reasoning=output.reasoning,
            expected_performance=output.expected_performance,
            confidence=output.confidence,
            memory_check_passed=memory_check_passed,
            estimated_memory_gb=estimated_memory / 1024,
        )
