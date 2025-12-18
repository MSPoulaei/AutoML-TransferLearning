from typing import Optional

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from src.models import (
    DatasetInfo,
    TrainingConfig,
    TrainingResult,
    ExecutorResult,
    AnalyzerRecommendation,
)
from src.training import Trainer, SimulatedTrainer, RealTrainer
from src.utils import APIKeyManager, get_logger
from .base_agent import BaseAgent

logger = get_logger(__name__)


class ExecutorAnalysis(BaseModel):
    """Structured analysis from the Executor Agent."""

    analysis: str = Field(..., description="Detailed analysis of training results")
    key_observations: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(
        default_factory=list, description="Suggestions for next iteration"
    )
    convergence_assessment: str = Field(
        default="unknown",
        description="Assessment of model convergence: converged, converging, not_converging, overfitting",
    )
    recommended_changes: dict = Field(
        default_factory=dict, description="Specific parameter changes recommended"
    )


class ExecutorAgent(BaseAgent):
    """
    Agent responsible for executing training and analyzing results.
    """

    def __init__(
        self,
        key_manager: APIKeyManager,
        model_name: str = "gpt-4o",
        base_url: Optional[str] = None,
        simulation_mode: bool = False,
        data_dir: Optional[str] = None,
    ):
        super().__init__(key_manager, model_name, base_url=base_url)
        self.simulation_mode = simulation_mode
        self.data_dir = data_dir

        # Initialize appropriate trainer
        if simulation_mode:
            self.trainer: Trainer = SimulatedTrainer()
        else:
            self.trainer: Trainer = RealTrainer(data_dir=data_dir)

    def _get_system_prompt(self) -> str:
        return """You are an expert machine learning engineer analyzing training results.

Your role is to:
1. Analyze training metrics and learning curves
2. Identify issues (overfitting, underfitting, convergence problems)
3. Provide actionable suggestions for the next iteration
4. Compare current results to previous experiments

Key indicators to watch:
- Train/Val loss gap: Large gap indicates overfitting
- Metric plateau: May indicate learning rate too low or model capacity reached
- Oscillating loss: Learning rate may be too high
- No improvement: May need different architecture or strategy

Provide specific, actionable feedback that can guide the Analyzer Agent."""

    def _build_analysis_prompt(
        self,
        result: TrainingResult,
        recommendation: AnalyzerRecommendation,
        previous_best: Optional[float] = None,
    ) -> str:
        """Build prompt for result analysis."""

        improvement_text = ""
        if previous_best is not None:
            diff = result.primary_metric_value - previous_best
            improvement_text = f"\nPrevious best {result.primary_metric_name.value}: {previous_best:.4f}"
            improvement_text += (
                f"\nChange: {diff:+.4f} ({'improved' if diff > 0 else 'decreased'})"
            )

        # Format training history
        history_text = ""
        if result.train_loss_history and result.val_loss_history:
            history_text = "\n\nTraining History (last 10 epochs):\n"
            for i, (tl, vl, m) in enumerate(
                zip(
                    result.train_loss_history[-10:],
                    result.val_loss_history[-10:],
                    result.metric_history[-10:],
                )
            ):
                history_text += f"Epoch {result.epochs_trained - 10 + i + 1}: train_loss={tl:.4f}, val_loss={vl:.4f}, metric={m:.4f}\n"

        prompt = f"""Analyze the following training results:

Configuration:
- Backbone: {result.config.backbone.full_name}
- Fine-tuning Strategy: {result.config.strategy.strategy_type.value}
- Learning Rate: {result.config.strategy.learning_rate}
- Batch Size: {result.config.batch_size}
- Epochs: {result.epochs_trained} (stopped early: {result.stopped_early})

Results:
- Training Loss: {result.train_loss:.4f}
- Validation Loss: {result.val_loss:.4f}
- {result.primary_metric_name.value}: {result.primary_metric_value:.4f}
- Best Epoch: {result.best_epoch}
- Training Time: {result.training_time_seconds:.1f} seconds
{improvement_text}
{history_text}

Original Reasoning: {recommendation.reasoning}
Expected Performance: {recommendation.expected_performance:.4f}

Analyze these results and provide:
1. Assessment of what worked and what didn't
2. Identification of any issues (overfitting, underfitting, etc.)
3. Specific suggestions for improvement in the next iteration
4. Assessment of convergence state"""

        return prompt

    async def run(
        self,
        dataset_info: DatasetInfo,
        recommendation: AnalyzerRecommendation,
        previous_results: list[ExecutorResult] = None,
        iteration: int = 1,
    ) -> ExecutorResult:
        """
        Execute training with given configuration and analyze results.

        Args:
            dataset_info: Information about the dataset
            recommendation: Training configuration from Analyzer
            previous_results: Results from previous iterations
            iteration: Current iteration number

        Returns:
            ExecutorResult with training results and analysis
        """
        logger.info(f"Executor Agent starting iteration {iteration}")

        # Check memory before training
        if not recommendation.memory_check_passed:
            logger.warning("Memory check failed, but proceeding with training")

        # Execute training
        try:
            training_result = await self.trainer.train(
                config=recommendation.training_config,
                dataset_info=dataset_info,
            )
            success = True
            error_message = None

        except Exception as e:
            logger.error(f"Training failed: {e}")
            success = False
            error_message = str(e)
            training_result = None

        # Analyze results if training succeeded
        analysis = ""
        suggestions = []
        improvement = None
        is_best = False

        if success and training_result:
            # Find previous best
            previous_best = None
            if previous_results:
                valid_results = [
                    r.training_result.primary_metric_value
                    for r in previous_results
                    if r.training_result is not None
                ]
                if valid_results:
                    previous_best = max(valid_results)

            # Calculate improvement
            if previous_best is not None:
                improvement = training_result.primary_metric_value - previous_best
                is_best = improvement > 0
            else:
                is_best = True

            # Get LLM analysis
            agent = await self._create_agent(ExecutorAnalysis)
            prompt = self._build_analysis_prompt(
                result=training_result,
                recommendation=recommendation,
                previous_best=previous_best,
            )

            analysis_output: ExecutorAnalysis = await self._execute_with_retry(
                agent, prompt
            )

            analysis = analysis_output.analysis
            suggestions = analysis_output.suggestions

            logger.info(
                f"Training completed: {training_result.primary_metric_name.value}="
                f"{training_result.primary_metric_value:.4f}, "
                f"improvement={improvement:.4f if improvement else 'N/A'}"
            )

        return ExecutorResult(
            iteration=iteration,
            success=success,
            training_result=training_result,
            error_message=error_message,
            analysis=analysis,
            suggestions=suggestions,
            improvement=improvement,
            is_best_so_far=is_best,
        )
