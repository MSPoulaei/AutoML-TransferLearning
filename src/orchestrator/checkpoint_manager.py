import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models import OrchestratorState
from src.utils import get_logger

logger = get_logger(__name__)


class CheckpointManager:
    """
    Manages checkpointing and restoration of orchestrator state.
    """

    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"CheckpointManager initialized: {checkpoint_dir}")

    def save(self, state: OrchestratorState) -> str:
        """
        Save orchestrator state to checkpoint.

        Returns:
            Path to checkpoint file
        """
        state.last_checkpoint_time = datetime.utcnow()

        checkpoint_path = self.checkpoint_dir / f"{state.experiment_id}_checkpoint.json"

        with open(checkpoint_path, "w") as f:
            f.write(state.model_dump_json(indent=2))

        logger.info(
            f"Checkpoint saved: {checkpoint_path}, "
            f"iteration {state.current_iteration}/{state.total_iterations}"
        )

        return str(checkpoint_path)

    def load(self, experiment_id: str) -> Optional[OrchestratorState]:
        """
        Load orchestrator state from checkpoint.

        Args:
            experiment_id: Experiment ID to load

        Returns:
            OrchestratorState if found, None otherwise
        """
        checkpoint_path = self.checkpoint_dir / f"{experiment_id}_checkpoint.json"

        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found for experiment: {experiment_id}")
            return None

        with open(checkpoint_path, "r") as f:
            state = OrchestratorState.model_validate_json(f.read())

        logger.info(
            f"Checkpoint loaded: {experiment_id}, "
            f"iteration {state.current_iteration}/{state.total_iterations}"
        )

        return state

    def exists(self, experiment_id: str) -> bool:
        """Check if checkpoint exists for experiment."""
        checkpoint_path = self.checkpoint_dir / f"{experiment_id}_checkpoint.json"
        return checkpoint_path.exists()

    def list_checkpoints(self) -> list[str]:
        """List all available checkpoints."""
        checkpoints = []
        for path in self.checkpoint_dir.glob("*_checkpoint.json"):
            experiment_id = path.stem.replace("_checkpoint", "")
            checkpoints.append(experiment_id)
        return checkpoints

    def delete(self, experiment_id: str) -> bool:
        """Delete checkpoint for experiment."""
        checkpoint_path = self.checkpoint_dir / f"{experiment_id}_checkpoint.json"

        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint deleted: {experiment_id}")
            return True

        return False

    def get_checkpoint_info(self, experiment_id: str) -> Optional[dict]:
        """Get information about a checkpoint without fully loading."""
        checkpoint_path = self.checkpoint_dir / f"{experiment_id}_checkpoint.json"

        if not checkpoint_path.exists():
            return None

        with open(checkpoint_path, "r") as f:
            data = json.load(f)

        return {
            "experiment_id": data.get("experiment_id"),
            "current_iteration": data.get("current_iteration"),
            "total_iterations": data.get("total_iterations"),
            "remaining_budget": data.get("remaining_budget"),
            "status": data.get("status"),
            "best_metric_value": data.get("best_metric_value"),
            "last_checkpoint_time": data.get("last_checkpoint_time"),
        }
