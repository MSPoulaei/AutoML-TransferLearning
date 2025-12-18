import asyncio
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from src.agents import AnalyzerAgent, ExecutorAgent
from src.models import (
    DatasetInfo,
    TrainingConfig,
    ExperimentRecord,
    ExecutorResult,
    OrchestratorState,
)
from src.utils import (
    APIKeyManager,
    get_logger,
    generate_experiment_id,
    ensure_directories,
)
from .budget_manager import BudgetManager
from .experiment_tracker import ExperimentTracker
from .checkpoint_manager import CheckpointManager

logger = get_logger(__name__)
console = Console()


class Orchestrator:
    """
    Main orchestrator for the multi-agent transfer learning system.

    Coordinates the Analyzer and Executor agents in a loop until
    budget is exhausted or convergence criteria are met.
    """

    def __init__(
        self,
        api_keys: list[str],
        model_name: str = "gpt-4o",
        base_url: str = "https://api.openai.com/v1",
        memory_limit_gb: float = 15.0,
        simulation_mode: bool = False,
        data_dir: Optional[str] = None,
        checkpoint_dir: str = "checkpoints",
        database_url: str = "sqlite:///experiments/experiments.db",
        max_retries: int = 3,
        rate_limit_cooldown: int = 60,
        # Per-agent settings
        analyzer_api_keys: Optional[list[str]] = None,
        analyzer_model: Optional[str] = None,
        analyzer_base_url: Optional[str] = None,
        executor_api_keys: Optional[list[str]] = None,
        executor_model: Optional[str] = None,
        executor_base_url: Optional[str] = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            api_keys: List of OpenAI API keys for rotation (default for all agents)
            model_name: OpenAI model to use (default for all agents)
            base_url: OpenAI API base URL (default for all agents)
            memory_limit_gb: GPU memory limit in GB
            simulation_mode: If True, use simulated training
            data_dir: Directory containing dataset (for real training)
            checkpoint_dir: Directory for checkpoints
            database_url: SQLAlchemy database URL
            max_retries: Max retries per API key
            rate_limit_cooldown: Cooldown seconds after rate limit
            analyzer_api_keys: API keys for Analyzer agent (overrides default)
            analyzer_model: Model for Analyzer agent (overrides default)
            analyzer_base_url: Base URL for Analyzer agent (overrides default)
            executor_api_keys: API keys for Executor agent (overrides default)
            executor_model: Model for Executor agent (overrides default)
            executor_base_url: Base URL for Executor agent (overrides default)
        """
        # Initialize per-agent API key managers
        analyzer_keys = analyzer_api_keys if analyzer_api_keys else api_keys
        executor_keys = executor_api_keys if executor_api_keys else api_keys

        self.analyzer_key_manager = APIKeyManager(
            api_keys=analyzer_keys,
            max_failures_per_key=max_retries,
            rate_limit_cooldown=rate_limit_cooldown,
        )

        self.executor_key_manager = APIKeyManager(
            api_keys=executor_keys,
            max_failures_per_key=max_retries,
            rate_limit_cooldown=rate_limit_cooldown,
        )

        # Initialize agents with per-agent settings
        self.analyzer = AnalyzerAgent(
            key_manager=self.analyzer_key_manager,
            model_name=analyzer_model if analyzer_model else model_name,
            base_url=analyzer_base_url if analyzer_base_url else base_url,
            memory_limit_gb=memory_limit_gb,
        )

        self.executor = ExecutorAgent(
            key_manager=self.executor_key_manager,
            model_name=executor_model if executor_model else model_name,
            base_url=executor_base_url if executor_base_url else base_url,
            simulation_mode=simulation_mode,
            data_dir=data_dir,
        )

        # Initialize managers
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        self.experiment_tracker = ExperimentTracker(database_url)

        # Configuration
        self.memory_limit_gb = memory_limit_gb
        self.simulation_mode = simulation_mode

        # State
        self._state: Optional[OrchestratorState] = None
        self._budget_manager: Optional[BudgetManager] = None

        # Ensure directories exist
        ensure_directories(checkpoint_dir, "experiments", "logs")

        logger.info(
            f"Orchestrator initialized: "
            f"analyzer_model={self.analyzer.model_name}, "
            f"executor_model={self.executor.model_name}, "
            f"memory={memory_limit_gb}GB, simulation={simulation_mode}"
        )

    async def run(
        self,
        dataset_info: DatasetInfo,
        budget: int,
        experiment_id: Optional[str] = None,
        resume: bool = False,
        early_stopping_patience: int = 3,
        improvement_threshold: float = 0.01,
    ) -> OrchestratorState:
        """
        Run the orchestration loop.

        Args:
            dataset_info: Information about the dataset
            budget: Number of iterations (analyzer-executor loops)
            experiment_id: Optional experiment ID (auto-generated if not provided)
            resume: If True, resume from checkpoint
            early_stopping_patience: Stop if no improvement for this many iterations
            improvement_threshold: Minimum improvement to reset patience

        Returns:
            Final OrchestratorState
        """
        # Setup experiment
        experiment_id = experiment_id or generate_experiment_id()

        # Check for resume
        if resume and self.checkpoint_manager.exists(experiment_id):
            self._state = self.checkpoint_manager.load(experiment_id)
            self._budget_manager = BudgetManager(
                total_budget=self._state.total_budget,
            )
            self._budget_manager.restore(
                self._budget_manager.get_state()._replace(
                    remaining_budget=self._state.remaining_budget
                )
            )
            console.print(
                f"[green]Resumed experiment {experiment_id} from iteration "
                f"{self._state.current_iteration}[/green]"
            )
        else:
            # Initialize new state
            self._state = OrchestratorState(
                experiment_id=experiment_id,
                dataset_info=dataset_info,
                total_budget=budget,
                remaining_budget=budget,
                current_iteration=0,
                total_iterations=budget,
                status="running",
            )
            self._budget_manager = BudgetManager(total_budget=budget)

        # Display experiment info
        self._display_experiment_info()

        # Get previous results for context
        previous_results: list[ExecutorResult] = [
            r.result for r in self._state.experiment_history
        ]

        no_improvement_count = 0

        # Main orchestration loop
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:

            task = progress.add_task(
                f"Running experiment {experiment_id}...", total=budget
            )

            while self._budget_manager.has_budget:
                self._state.current_iteration += 1
                iteration = self._state.current_iteration

                progress.update(task, description=f"Iteration {iteration}/{budget}...")

                try:
                    # Run analyzer
                    console.print(f"\n[bold blue]Iteration {iteration}[/bold blue]")
                    console.print("[yellow]Analyzer Agent thinking...[/yellow]")

                    recommendation = await self.analyzer.run(
                        dataset_info=dataset_info,
                        previous_results=previous_results,
                        iteration=iteration,
                    )

                    console.print(
                        f"  → Backbone: {recommendation.training_config.backbone.full_name}"
                    )
                    console.print(
                        f"  → Strategy: {recommendation.training_config.strategy.strategy_type.value}"
                    )
                    console.print(
                        f"  → Expected: {recommendation.expected_performance:.4f}"
                    )

                    # Run executor
                    console.print("[yellow]Executor Agent training...[/yellow]")

                    result = await self.executor.run(
                        dataset_info=dataset_info,
                        recommendation=recommendation,
                        previous_results=previous_results,
                        iteration=iteration,
                    )

                    if result.success and result.training_result:
                        metric_value = result.training_result.primary_metric_value
                        console.print(
                            f"  → Result: {dataset_info.primary_metric.value}="
                            f"{metric_value:.4f}"
                        )

                        if result.is_best_so_far:
                            console.print("[green]  → New best result![/green]")
                            self._state.best_result = result
                            self._state.best_config = recommendation.training_config
                            self._state.best_metric_value = metric_value
                            no_improvement_count = 0
                        else:
                            improvement = result.improvement or 0
                            if improvement < improvement_threshold:
                                no_improvement_count += 1
                            else:
                                no_improvement_count = 0
                    else:
                        console.print(
                            f"[red]  → Training failed: {result.error_message}[/red]"
                        )

                    # Record experiment
                    record = ExperimentRecord(
                        experiment_id=experiment_id,
                        iteration=iteration,
                        dataset_info=dataset_info,
                        recommendation=recommendation,
                        result=result,
                        api_calls_used=self.analyzer.call_count
                        + self.executor.call_count,
                        compute_time_seconds=(
                            result.training_result.training_time_seconds
                            if result.training_result
                            else 0
                        ),
                    )

                    self._state.experiment_history.append(record)
                    previous_results.append(result)

                    # Save to database
                    self.experiment_tracker.record_experiment(record)

                    # Consume budget
                    self._budget_manager.consume()
                    self._state.remaining_budget = self._budget_manager.remaining

                    # Checkpoint
                    self.checkpoint_manager.save(self._state)

                    # Check early stopping
                    if no_improvement_count >= early_stopping_patience:
                        console.print(
                            f"\n[yellow]Early stopping: no improvement for "
                            f"{early_stopping_patience} iterations[/yellow]"
                        )
                        break

                    progress.advance(task)

                except Exception as e:
                    logger.error(f"Error in iteration {iteration}: {e}")
                    console.print(f"[red]Error: {e}[/red]")

                    # Save state and continue
                    self._state.status = "error"
                    self._state.error_message = str(e)
                    self.checkpoint_manager.save(self._state)

                    # Continue to next iteration if budget remains
                    self._budget_manager.consume()

        # Finalize
        self._state.status = "completed"
        self.checkpoint_manager.save(self._state)

        # Display final results
        self._display_final_results()

        return self._state

    def _display_experiment_info(self):
        """Display experiment information."""
        console.print("\n" + "=" * 60)
        console.print(f"[bold green]Transfer Learning Orchestrator[/bold green]")
        console.print("=" * 60)

        table = Table(title="Experiment Configuration")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Experiment ID", self._state.experiment_id)
        table.add_row("Mode", "Simulation" if self.simulation_mode else "Real Training")
        table.add_row("Budget", str(self._state.total_budget))
        table.add_row("Memory Limit", f"{self.memory_limit_gb} GB")
        table.add_row("Num Classes", str(self._state.dataset_info.num_classes))
        table.add_row("Num Samples", str(self._state.dataset_info.num_samples))
        table.add_row("Image Size", str(self._state.dataset_info.image_size))
        table.add_row("Domain", self._state.dataset_info.domain.value)
        table.add_row("Primary Metric", self._state.dataset_info.primary_metric.value)

        console.print(table)
        console.print()

    def _display_final_results(self):
        """Display final experiment results."""
        console.print("\n" + "=" * 60)
        console.print("[bold green]Experiment Completed[/bold green]")
        console.print("=" * 60)

        # Summary table
        table = Table(title="Results Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")

        table.add_row("Total Iterations", str(len(self._state.experiment_history)))
        table.add_row(
            "Best Metric",
            (
                f"{self._state.best_metric_value:.4f}"
                if self._state.best_metric_value
                else "N/A"
            ),
        )

        if self._state.best_config:
            table.add_row("Best Backbone", self._state.best_config.backbone.full_name)
            table.add_row(
                "Best Strategy", self._state.best_config.strategy.strategy_type.value
            )

        budget_summary = self._budget_manager.get_summary()
        table.add_row(
            "API Calls Used", str(self.analyzer.call_count + self.executor.call_count)
        )
        table.add_row(
            "Total Compute Time", f"{budget_summary['compute_time_seconds']:.1f}s"
        )

        console.print(table)

        # API key stats from both agents
        analyzer_stats = self.analyzer_key_manager.get_stats()
        executor_stats = self.executor_key_manager.get_stats()
        total_calls = analyzer_stats["total_calls"] + executor_stats["total_calls"]
        rate_limited = (
            analyzer_stats["rate_limited_keys"] + executor_stats["rate_limited_keys"]
        )

        console.print(
            f"\nAPI Key Stats: {total_calls} total calls, "
            f"{rate_limited} rate limited keys"
        )

        console.print(f"\nCheckpoint saved: {self._state.experiment_id}")
        console.print()
