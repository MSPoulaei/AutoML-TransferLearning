import asyncio
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from dotenv import load_dotenv

from config import get_settings, Settings
from src.models import DatasetInfo, DatasetDomain, MetricType
from src.orchestrator import Orchestrator
from src.utils import setup_logging, ensure_directories

# Load environment variables
load_dotenv()

# Initialize
app = typer.Typer(
    name="tl-orchestrator", help="Multi-Agent Transfer Learning Orchestration System"
)
console = Console()


@app.command()
def run(
    # Dataset information
    num_classes: int = typer.Option(
        ..., "--num-classes", "-c", help="Number of classification classes"
    ),
    num_samples: int = typer.Option(
        ..., "--num-samples", "-n", help="Number of training samples"
    ),
    image_height: int = typer.Option(224, "--image-height", help="Image height"),
    image_width: int = typer.Option(224, "--image-width", help="Image width"),
    domain: str = typer.Option(
        "natural",
        "--domain",
        "-d",
        help="Dataset domain: natural, medical, satellite, document, fine_grained, industrial, artistic, other",
    ),
    domain_description: str = typer.Option(
        ..., "--domain-desc", help="Detailed description of the dataset domain"
    ),
    primary_metric: str = typer.Option(
        "accuracy",
        "--metric",
        "-m",
        help="Primary metric: accuracy, f1_score, precision, recall, auc_roc",
    ),
    class_balance: str = typer.Option(
        "balanced",
        "--class-balance",
        help="Class distribution: balanced, slightly_imbalanced, highly_imbalanced",
    ),
    # Orchestration settings
    budget: int = typer.Option(
        10, "--budget", "-b", help="Number of optimization iterations"
    ),
    experiment_id: Optional[str] = typer.Option(
        None,
        "--experiment-id",
        "-e",
        help="Experiment ID (auto-generated if not provided)",
    ),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume from checkpoint"),
    # Training settings
    simulation: bool = typer.Option(
        False,
        "--simulation",
        "-s",
        help="Use simulated training instead of real training",
    ),
    data_dir: Optional[str] = typer.Option(
        None, "--data-dir", help="Directory containing the dataset"
    ),
    memory_limit: float = typer.Option(
        15.0, "--memory-limit", help="GPU memory limit in GB"
    ),
    # Early stopping
    early_stopping_patience: int = typer.Option(
        3, "--patience", help="Early stopping patience (iterations without improvement)"
    ),
    improvement_threshold: float = typer.Option(
        0.01, "--threshold", help="Minimum improvement threshold"
    ),
    # Logging
    log_level: str = typer.Option(
        "INFO", "--log-level", help="Logging level: DEBUG, INFO, WARNING, ERROR"
    ),
):
    """
    Run the transfer learning orchestration.
    
    Example:
        python main.py run --num-classes 10 --num-samples 5000 \\
            --domain natural --domain-desc "Natural scene images" \\
            --budget 5 --simulation
    """
    # Setup logging
    settings = get_settings()
    setup_logging(log_level=log_level, log_file=settings.log_file)

    # Validate API keys
    if not settings.api_keys_list:
        console.print(
            "[red]Error: No API keys found. Set OPENAI_API_KEYS in .env[/red]"
        )
        raise typer.Exit(1)

    # Create dataset info
    try:
        dataset_info = DatasetInfo(
            num_classes=num_classes,
            num_samples=num_samples,
            image_size=(image_height, image_width),
            domain=DatasetDomain(domain),
            domain_description=domain_description,
            primary_metric=MetricType(primary_metric),
            class_balance=class_balance,
        )
    except ValueError as e:
        console.print(f"[red]Invalid dataset configuration: {e}[/red]")
        raise typer.Exit(1)

    # Initialize orchestrator
    orchestrator = Orchestrator(
        api_keys=settings.api_keys_list,
        model_name=settings.openai_model,
        memory_limit_gb=memory_limit,
        simulation_mode=simulation,
        data_dir=data_dir,
        checkpoint_dir=settings.checkpoint_dir,
        database_url=settings.database_url,
        max_retries=settings.max_retries_per_key,
        rate_limit_cooldown=settings.rate_limit_retry_delay,
    )

    # Run orchestration
    try:
        final_state = asyncio.run(
            orchestrator.run(
                dataset_info=dataset_info,
                budget=budget,
                experiment_id=experiment_id,
                resume=resume,
                early_stopping_patience=early_stopping_patience,
                improvement_threshold=improvement_threshold,
            )
        )

        if final_state.best_metric_value:
            console.print(
                f"\n[bold green]Best result: "
                f"{final_state.best_metric_value:.4f}[/bold green]"
            )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user. State saved.[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_experiments():
    """List all experiments with checkpoints."""
    settings = get_settings()
    setup_logging(log_level="WARNING")

    from src.orchestrator import CheckpointManager, ExperimentTracker

    checkpoint_manager = CheckpointManager(settings.checkpoint_dir)
    tracker = ExperimentTracker(settings.database_url)

    checkpoints = checkpoint_manager.list_checkpoints()

    if not checkpoints:
        console.print("[yellow]No experiments found.[/yellow]")
        return

    from rich.table import Table

    table = Table(title="Experiments")
    table.add_column("Experiment ID", style="cyan")
    table.add_column("Status", style="magenta")
    table.add_column("Iterations", style="green")
    table.add_column("Best Metric", style="yellow")

    for exp_id in checkpoints:
        info = checkpoint_manager.get_checkpoint_info(exp_id)
        if info:
            table.add_row(
                exp_id,
                info.get("status", "unknown"),
                f"{info.get('current_iteration', 0)}/{info.get('total_iterations', 0)}",
                (
                    f"{info.get('best_metric_value', 0):.4f}"
                    if info.get("best_metric_value")
                    else "N/A"
                ),
            )

    console.print(table)


@app.command()
def show_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID to show")
):
    """Show detailed information about an experiment."""
    settings = get_settings()
    setup_logging(log_level="WARNING")

    from src.orchestrator import ExperimentTracker

    tracker = ExperimentTracker(settings.database_url)
    summary = tracker.get_experiment_summary(experiment_id)

    if not summary:
        console.print(f"[red]Experiment {experiment_id} not found.[/red]")
        raise typer.Exit(1)

    from rich.table import Table

    table = Table(title=f"Experiment: {experiment_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    for key, value in summary.items():
        if isinstance(value, float):
            table.add_row(key, f"{value:.4f}")
        elif isinstance(value, list):
            table.add_row(key, ", ".join(str(v) for v in value))
        else:
            table.add_row(key, str(value))

    console.print(table)


@app.command()
def resume_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID to resume"),
    additional_budget: int = typer.Option(
        5, "--budget", "-b", help="Additional iterations to run"
    ),
    simulation: bool = typer.Option(
        False, "--simulation", "-s", help="Use simulated training"
    ),
):
    """Resume a paused or failed experiment."""
    settings = get_settings()
    setup_logging(log_level="INFO", log_file=settings.log_file)

    from src.orchestrator import CheckpointManager

    checkpoint_manager = CheckpointManager(settings.checkpoint_dir)

    if not checkpoint_manager.exists(experiment_id):
        console.print(f"[red]No checkpoint found for {experiment_id}[/red]")
        raise typer.Exit(1)

    state = checkpoint_manager.load(experiment_id)

    # Add additional budget
    state.total_budget += additional_budget
    state.remaining_budget += additional_budget
    state.total_iterations = state.total_budget

    # Save updated state
    checkpoint_manager.save(state)

    # Initialize orchestrator
    orchestrator = Orchestrator(
        api_keys=settings.api_keys_list,
        model_name=settings.openai_model,
        memory_limit_gb=settings.memory_limit_gb,
        simulation_mode=simulation,
        checkpoint_dir=settings.checkpoint_dir,
        database_url=settings.database_url,
    )

    # Run with resume
    try:
        final_state = asyncio.run(
            orchestrator.run(
                dataset_info=state.dataset_info,
                budget=state.total_budget,
                experiment_id=experiment_id,
                resume=True,
            )
        )

        console.print(
            f"\n[bold green]Resumed experiment completed. "
            f"Best: {final_state.best_metric_value:.4f}[/bold green]"
        )

    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Progress saved.[/yellow]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def delete_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID to delete"),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """Delete an experiment checkpoint."""
    settings = get_settings()

    from src.orchestrator import CheckpointManager

    checkpoint_manager = CheckpointManager(settings.checkpoint_dir)

    if not checkpoint_manager.exists(experiment_id):
        console.print(f"[yellow]No checkpoint found for {experiment_id}[/yellow]")
        return

    if not force:
        confirm = typer.confirm(f"Delete experiment {experiment_id}?")
        if not confirm:
            console.print("[yellow]Cancelled.[/yellow]")
            return

    checkpoint_manager.delete(experiment_id)
    console.print(f"[green]Deleted experiment {experiment_id}[/green]")


@app.command()
def export_results(
    experiment_id: str = typer.Argument(..., help="Experiment ID to export"),
    output: str = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file path (default: {experiment_id}_results.json)",
    ),
):
    """Export experiment results to JSON."""
    import json

    settings = get_settings()

    from src.orchestrator import ExperimentTracker

    tracker = ExperimentTracker(settings.database_url)
    history = tracker.get_experiment_history(experiment_id)

    if not history:
        console.print(f"[red]No results found for {experiment_id}[/red]")
        raise typer.Exit(1)

    output_path = output or f"{experiment_id}_results.json"

    results = {
        "experiment_id": experiment_id,
        "summary": tracker.get_experiment_summary(experiment_id),
        "iterations": [
            {
                "iteration": r.iteration,
                "backbone": r.recommendation.training_config.backbone.full_name,
                "strategy": r.recommendation.training_config.strategy.strategy_type.value,
                "metric": (
                    r.result.training_result.primary_metric_value
                    if r.result.training_result
                    else None
                ),
                "success": r.result.success,
            }
            for r in history
        ],
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    console.print(f"[green]Results exported to {output_path}[/green]")


if __name__ == "__main__":
    app()
