import asyncio
import sys
from pathlib import Path
from typing import Optional
import zipfile
import json
from datetime import datetime

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


def get_dataset_preset(preset_name: str) -> dict:
    """
    Get predefined configuration for famous datasets.

    Args:
        preset_name: Name of the dataset preset

    Returns:
        Dictionary with dataset configuration
    """
    presets = {
        "cifar10": {
            "num_classes": 10,
            "num_samples": 50000,
            "image_height": 32,
            "image_width": 32,
            "domain": "natural",
            "domain_description": "60,000 32x32 color images in 10 balanced classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck",
            "class_balance": "balanced",
            "data_quality": "high",
        },
        "cifar100": {
            "num_classes": 100,
            "num_samples": 50000,
            "image_height": 32,
            "image_width": 32,
            "domain": "fine_grained",
            "domain_description": "60,000 32x32 color images in 100 fine-grained classes grouped into 20 superclasses",
            "class_balance": "balanced",
            "data_quality": "high",
        },
        "mnist": {
            "num_classes": 10,
            "num_samples": 60000,
            "image_height": 28,
            "image_width": 28,
            "domain": "document",
            "domain_description": "70,000 28x28 grayscale images of handwritten digits (0-9)",
            "class_balance": "balanced",
            "data_quality": "high",
        },
        "fashion_mnist": {
            "num_classes": 10,
            "num_samples": 60000,
            "image_height": 28,
            "image_width": 28,
            "domain": "natural",
            "domain_description": "70,000 28x28 grayscale images of fashion items: T-shirt, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag, ankle boot",
            "class_balance": "balanced",
            "data_quality": "high",
        },
        "svhn": {
            "num_classes": 10,
            "num_samples": 73257,
            "image_height": 32,
            "image_width": 32,
            "domain": "document",
            "domain_description": "Real-world images of street view house numbers - digits extracted from Google Street View",
            "class_balance": "slightly_imbalanced",
            "data_quality": "medium",
        },
        "imagenet": {
            "num_classes": 1000,
            "num_samples": 1281167,
            "image_height": 224,
            "image_width": 224,
            "domain": "natural",
            "domain_description": "Large-scale dataset with 1000 diverse object categories including animals, plants, objects, and scenes",
            "class_balance": "balanced",
            "data_quality": "high",
        },
    }

    preset = presets.get(preset_name.lower())
    if not preset:
        available = ", ".join(presets.keys())
        raise ValueError(
            f"Unknown dataset preset '{preset_name}'. Available presets: {available}"
        )

    return preset


def zip_experiment_results(
    experiment_id: str,
    settings: Settings,
    output_dir: str = ".",
) -> str:
    """
    Zip experiment results including checkpoints, logs, and metrics.

    Args:
        experiment_id: The experiment ID
        settings: Settings object
        output_dir: Directory to save the zip file

    Returns:
        Path to the created zip file
    """
    from src.orchestrator import CheckpointManager, ExperimentTracker

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = f"{experiment_id}_results_{timestamp}.zip"
    zip_path = Path(output_dir) / zip_filename

    checkpoint_manager = CheckpointManager(settings.checkpoint_dir)
    tracker = ExperimentTracker(settings.database_url)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Add checkpoint file
        checkpoint_file = (
            Path(settings.checkpoint_dir) / f"{experiment_id}_checkpoint.json"
        )
        if checkpoint_file.exists():
            zipf.write(checkpoint_file, f"checkpoint/{checkpoint_file.name}")

        # Add experiment summary and history as JSON
        summary = tracker.get_experiment_summary(experiment_id)
        if summary:
            summary_json = json.dumps(summary, indent=2, default=str)
            zipf.writestr(f"{experiment_id}_summary.json", summary_json)

        history = tracker.get_experiment_history(experiment_id)
        if history:
            history_data = {
                "experiment_id": experiment_id,
                "total_iterations": len(history),
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
                        "error": (
                            r.result.error_message if not r.result.success else None
                        ),
                    }
                    for r in history
                ],
            }
            history_json = json.dumps(history_data, indent=2, default=str)
            zipf.writestr(f"{experiment_id}_history.json", history_json)

        # Add logs if they exist
        log_file = Path(settings.log_file)
        if log_file.exists():
            zipf.write(log_file, f"logs/{log_file.name}")

        # Add a README with experiment info
        readme_content = f"""# Experiment Results: {experiment_id}

## Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Contents:
- checkpoint/{experiment_id}_checkpoint.json: Full experiment state and configuration
- {experiment_id}_summary.json: Experiment summary statistics
- {experiment_id}_history.json: Detailed iteration history
- logs/: Training logs (if available)

## Best Result:
{summary.get('best_metric_value', 'N/A') if summary else 'N/A'}

## Total Iterations:
{summary.get('total_iterations', 'N/A') if summary else 'N/A'}
"""
        zipf.writestr("README.md", readme_content)

    return str(zip_path)


@app.command()
def run(
    # Dataset preset or custom configuration
    dataset_preset: Optional[str] = typer.Option(
        None,
        "--dataset",
        help="Use a famous dataset preset: cifar10, cifar100, mnist, fashion_mnist, svhn, imagenet (overrides other dataset options)",
    ),
    # Dataset information
    num_classes: Optional[int] = typer.Option(
        None, "--num-classes", "-c", help="Number of classification classes"
    ),
    num_samples: Optional[int] = typer.Option(
        None, "--num-samples", "-n", help="Number of training samples"
    ),
    image_height: int = typer.Option(224, "--image-height", help="Image height"),
    image_width: int = typer.Option(224, "--image-width", help="Image width"),
    domain: str = typer.Option(
        "natural",
        "--domain",
        "-d",
        help="Dataset domain: natural, medical, satellite, document, fine_grained, industrial, artistic, other",
    ),
    domain_description: Optional[str] = typer.Option(
        None, "--domain-desc", help="Detailed description of the dataset domain"
    ),
    primary_metric: str = typer.Option(
        "f1_score",
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
    # Output options
    zip_results: bool = typer.Option(
        False, "--zip-results", "-z", help="Zip experiment results after completion"
    ),
    output_dir: str = typer.Option(
        ".",
        "--output-dir",
        "-o",
        help="Directory to save zip file (default: current directory)",
    ),
):
    """
    Run the transfer learning orchestration.
    
    Examples:
        # Using a famous dataset preset
        python main.py run --dataset cifar10 --budget 5 --simulation
        
        # Using a custom dataset
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

    # Determine dataset configuration
    if dataset_preset:
        # Use preset configuration
        try:
            preset_config = get_dataset_preset(dataset_preset)
            console.print(f"[cyan]Using dataset preset: {dataset_preset}[/cyan]")

            # Override with preset values
            num_classes = preset_config["num_classes"]
            num_samples = preset_config["num_samples"]
            image_height = preset_config["image_height"]
            image_width = preset_config["image_width"]
            domain = preset_config["domain"]
            domain_description = preset_config["domain_description"]
            class_balance = preset_config["class_balance"]

            # Use preset data quality if available
            data_quality = preset_config.get("data_quality", "high")

        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    else:
        # Custom dataset - validate required fields
        if num_classes is None:
            console.print(
                "[red]Error: --num-classes is required when not using a dataset preset[/red]"
            )
            raise typer.Exit(1)
        if num_samples is None:
            console.print(
                "[red]Error: --num-samples is required when not using a dataset preset[/red]"
            )
            raise typer.Exit(1)
        if domain_description is None:
            console.print(
                "[red]Error: --domain-desc is required when not using a dataset preset[/red]"
            )
            raise typer.Exit(1)
        data_quality = "high"

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
            data_quality=data_quality,
            dataset_name=dataset_preset,  # Store the preset name for auto-loading
        )
    except ValueError as e:
        console.print(f"[red]Invalid dataset configuration: {e}[/red]")
        raise typer.Exit(1)

    # Initialize orchestrator with per-agent settings
    orchestrator = Orchestrator(
        api_keys=settings.api_keys_list,
        model_name=settings.openai_model,
        base_url=settings.openai_base_url,
        memory_limit_gb=memory_limit,
        simulation_mode=simulation,
        data_dir=data_dir,
        checkpoint_dir=settings.checkpoint_dir,
        database_url=settings.database_url,
        max_retries=settings.max_retries_per_key,
        rate_limit_cooldown=settings.rate_limit_retry_delay,
        # Per-agent settings
        analyzer_api_keys=(
            settings.get_agent_api_keys("analyzer")
            if settings.analyzer_api_keys
            else None
        ),
        analyzer_model=settings.get_agent_model("analyzer"),
        analyzer_base_url=settings.get_agent_base_url("analyzer"),
        executor_api_keys=(
            settings.get_agent_api_keys("executor")
            if settings.executor_api_keys
            else None
        ),
        executor_model=settings.get_agent_model("executor"),
        executor_base_url=settings.get_agent_base_url("executor"),
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

        # Zip results if requested
        if zip_results:
            console.print("\n[cyan]Creating results archive...[/cyan]")
            try:
                zip_path = zip_experiment_results(
                    experiment_id=final_state.experiment_id,
                    settings=settings,
                    output_dir=output_dir,
                )
                console.print(
                    f"[bold green]Results archived to: {zip_path}[/bold green]"
                )
                console.print(
                    "[cyan]You can download this file from Kaggle output.[/cyan]"
                )
            except Exception as zip_error:
                console.print(
                    f"[yellow]Warning: Failed to create zip archive: {zip_error}[/yellow]"
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
        3, "--budget", "-b", help="Additional iterations to run"
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


@app.command()
def zip_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID to zip"),
    output_dir: str = typer.Option(
        ".",
        "--output-dir",
        "-o",
        help="Directory to save zip file (default: current directory)",
    ),
):
    """Create a zip archive of experiment results."""
    settings = get_settings()

    from src.orchestrator import CheckpointManager

    checkpoint_manager = CheckpointManager(settings.checkpoint_dir)

    if not checkpoint_manager.exists(experiment_id):
        console.print(f"[red]No checkpoint found for {experiment_id}[/red]")
        raise typer.Exit(1)

    console.print(f"[cyan]Creating archive for experiment {experiment_id}...[/cyan]")

    try:
        zip_path = zip_experiment_results(
            experiment_id=experiment_id,
            settings=settings,
            output_dir=output_dir,
        )
        console.print(f"[bold green]Results archived to: {zip_path}[/bold green]")

        # Show zip file size
        zip_size = Path(zip_path).stat().st_size / (1024 * 1024)  # MB
        console.print(f"[cyan]Archive size: {zip_size:.2f} MB[/cyan]")

    except Exception as e:
        console.print(f"[red]Error creating archive: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_datasets():
    """List available dataset presets."""
    from rich.table import Table

    table = Table(title="Available Dataset Presets")
    table.add_column("Name", style="cyan")
    table.add_column("Classes", style="green")
    table.add_column("Samples", style="yellow")
    table.add_column("Image Size", style="magenta")
    table.add_column("Domain", style="blue")

    presets = {
        "cifar10": {
            "num_classes": 10,
            "num_samples": 50000,
            "image_size": "32x32",
            "domain": "natural",
        },
        "cifar100": {
            "num_classes": 100,
            "num_samples": 50000,
            "image_size": "32x32",
            "domain": "fine_grained",
        },
        "mnist": {
            "num_classes": 10,
            "num_samples": 60000,
            "image_size": "28x28",
            "domain": "document",
        },
        "fashion_mnist": {
            "num_classes": 10,
            "num_samples": 60000,
            "image_size": "28x28",
            "domain": "natural",
        },
        "svhn": {
            "num_classes": 10,
            "num_samples": 73257,
            "image_size": "32x32",
            "domain": "document",
        },
        "imagenet": {
            "num_classes": 1000,
            "num_samples": 1281167,
            "image_size": "224x224",
            "domain": "natural",
        },
    }

    for name, info in presets.items():
        table.add_row(
            name,
            str(info["num_classes"]),
            str(info["num_samples"]),
            info["image_size"],
            info["domain"],
        )

    console.print(table)
    console.print(
        "\n[cyan]Usage:[/cyan] python main.py run --dataset <preset_name> --budget 5 --simulation"
    )


if __name__ == "__main__":
    app()
