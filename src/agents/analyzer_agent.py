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
    total_budget: int = 5
    available_backbones: list[str] = Field(default_factory=list)
    available_strategies: list[str] = Field(default_factory=list)


class AnalyzerOutput(BaseModel):
    """Structured output from the Analyzer Agent."""

    reasoning: str = Field(
        ...,
        description="Detailed explanation of your analysis and decision-making process. Start with this field first. Include: dataset analysis, backbone selection rationale, strategy justification, hyperparameter reasoning, and how previous results influenced your recommendations.",
    )
    backbone_family: str = Field(
        ...,
        description="The backbone family to use (e.g., 'resnet', 'efficientnet', 'convnext')",
    )
    backbone_variant: str = Field(
        ...,
        description="The specific variant of the backbone (e.g., 'resnet50', 'efficientnet_b2')",
    )
    finetuning_strategy: str = Field(
        ...,
        description="The fine-tuning strategy to use (e.g., 'head_only', 'full_finetuning', 'gradual_unfreezing')",
    )
    learning_rate: float = Field(
        ..., description="The learning rate for training (e.g., 0.001, 0.0001)"
    )
    batch_size: int = Field(
        ..., description="The batch size for training (must be positive integer)"
    )
    epochs: int = Field(
        ...,
        ge=1,
        le=20,
        description="Number of training epochs (must be between 1 and 20)",
    )
    dropout: float = Field(
        default=0.2,
        ge=0,
        le=0.9,
        description="Dropout rate for regularization (0.0-0.9, default 0.2)",
    )
    label_smoothing: float = Field(
        default=0.1,
        ge=0,
        le=0.3,
        description="Label smoothing factor for regularization (0.0-0.3, default 0.1)",
    )
    expected_performance: float = Field(
        ...,
        ge=0,
        le=1,
        description="Expected performance metric as a decimal between 0.0 and 1.0 (e.g., 0.95 for 95%)",
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Confidence in this recommendation as a decimal between 0.0 and 1.0 (e.g., 0.8 for 80%)",
    )


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

STRATEGIC DECISION MAKING:
- Early iterations (1-3): Explore diverse approaches (different backbones, strategies)
- Mid iterations: Exploit successful patterns, make incremental improvements
- Final iterations: Fine-tune the best configuration found so far
- PAY ATTENTION to Executor's "recommended_changes" and "convergence_assessment"
- If convergence_assessment is "overfitting": increase regularization (dropout, label_smoothing)
- If convergence_assessment is "not_converging": try different backbone or increase learning rate
- If convergence_assessment is "converged": try more complex model or different strategy

LEARNING FROM PREVIOUS RESULTS:
- Prioritize suggestions from Executor's analysis
- If a configuration showed "overfitting", avoid similar configs or increase regularization
- If improvement is marginal (<0.01), consider more significant changes
- Don't repeat failed experiments unless addressing the identified issue

Use your expertise to make informed decisions based on:
- Dataset characteristics (size, domain, image dimensions, number of classes)
- Available computational resources (memory limits)
- Previous experiment results (learn from what worked and what didn't)
- Executor's specific recommendations from last iteration
- Transfer learning best practices from your training data

CRITICAL OUTPUT FORMAT REQUIREMENTS:
1. You MUST output your response as structured JSON following the exact schema provided
2. The FIRST field in your JSON output MUST be "reasoning" - start your JSON with "reasoning" as the very first key
3. Your "reasoning" field should contain a detailed explanation (200-500 words) covering:
   - Analysis of dataset characteristics (size, domain, complexity)
   - Reference to iteration number and strategy (exploration vs exploitation)
   - How Executor's suggestions influenced your decision
   - Rationale for backbone selection
   - Justification for fine-tuning strategy choice
   - Explanation of hyperparameter decisions (learning rate, batch size, epochs, regularization)
   - How previous results influenced your recommendations (if applicable)
4. expected_performance: MUST be a decimal between 0.0 and 1.0 (e.g., 0.95 for 95%, NOT 95)
5. confidence: MUST be a decimal between 0.0 and 1.0 (e.g., 0.8 for 80%, NOT 80)
6. epochs: MUST be between 1 and 20 (inclusive)
7. In your reasoning text, use ONLY standard ASCII characters. Do NOT use fancy Unicode characters like em-dashes (—), en-dashes (–), fancy quotes (" "), curly quotes, or special spaces. Use regular hyphens (-), regular quotes ("), and regular spaces only.

EXAMPLE REASONING (Iteration 2 after overfitting):
"Based on iteration 1 results (ResNet18, head_only, accuracy=0.826), the Executor identified overfitting with train-val gap of 0.278 and recommended full_finetuning with increased dropout. I'm implementing these suggestions: switching to full_finetuning to allow better feature adaptation, increasing dropout from 0.2 to 0.3, and reducing learning rate to 0.0005 for more stable training. Keeping ResNet18 since the architecture isn't the issue. Reducing epochs to 12 since previous training degraded after epoch 12. Expected performance: 0.88 based on addressing the overfitting issue."

REMEMBER: Start your JSON output with "reasoning" as the first field!"""

    def _build_prompt(self, context: AnalyzerContext) -> str:
        """Build the prompt for the analyzer agent."""

        # Add iteration strategy context
        iteration_context = ""
        if context.iteration == 1:
            iteration_context = (
                "\nStrategy: EXPLORE - Try a reasonable baseline configuration."
            )
        elif context.iteration <= 3:
            iteration_context = f"\nStrategy: EXPLORE - Try diverse approaches. Iteration {context.iteration} of exploration phase."
        elif context.iteration > context.total_budget * 0.7:
            iteration_context = f"\nStrategy: EXPLOIT - Fine-tune best configuration. Final iterations (iteration {context.iteration}/{context.total_budget})."
        else:
            iteration_context = f"\nStrategy: EXPLOIT - Build on successful patterns from previous results."

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

            # Add last iteration's specific recommendations
            last_result = context.previous_results[-1]
            if last_result.suggestions:
                prev_results_text += f"\n\nLast Iteration Key Recommendations:\n"
                for i, suggestion in enumerate(last_result.suggestions[:3], 1):
                    prev_results_text += f"{i}. {suggestion}\n"

        # Build dataset description without revealing the dataset name
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
- Iteration: {context.iteration}/{context.total_budget}
{iteration_context}

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
        total_budget: int = 5,
    ) -> AnalyzerRecommendation:
        """
        Analyze dataset and previous results to generate a training recommendation.

        Args:
            dataset_info: Information about the dataset
            previous_results: Results from previous training iterations
            iteration: Current iteration number
            total_budget: Total number of iterations planned

        Returns:
            AnalyzerRecommendation with complete training configuration
        """
        logger.info(f"Analyzer Agent starting iteration {iteration}/{total_budget}")

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
            total_budget=total_budget,
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

        # Add token usage and cost tracking from the API call
        usage = self.get_last_call_usage()
        recommendation.input_tokens = usage.get("input_tokens", 0)
        recommendation.output_tokens = usage.get("output_tokens", 0)
        recommendation.total_tokens = usage.get("total_tokens", 0)
        recommendation.api_cost = usage.get("cost", 0.0)

        logger.info(
            f"Analyzer recommendation: {recommendation.training_config.backbone.full_name} "
            f"with {recommendation.training_config.strategy.strategy_type.value} "
            f"(tokens: {recommendation.total_tokens}, cost: ${recommendation.api_cost:.4f})"
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
