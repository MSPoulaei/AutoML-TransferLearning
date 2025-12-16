from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class BudgetState:
    """Current state of budget tracking."""

    total_budget: int
    remaining_budget: int
    api_calls_used: int = 0
    compute_time_seconds: float = 0.0
    iterations_completed: int = 0
    start_time: datetime = field(default_factory=datetime.utcnow)


class BudgetManager:
    """
    Manages the budget for orchestration iterations.

    Budget is defined as the number of analyzer-executor loop iterations.
    """

    def __init__(
        self,
        total_budget: int,
        cost_per_iteration: int = 1,
        reserved_budget: int = 0,
    ):
        """
        Initialize budget manager.

        Args:
            total_budget: Total number of iterations allowed
            cost_per_iteration: Cost per iteration (default 1)
            reserved_budget: Budget reserved for final operations
        """
        self.total_budget = total_budget
        self.cost_per_iteration = cost_per_iteration
        self.reserved_budget = reserved_budget

        self._state = BudgetState(
            total_budget=total_budget,
            remaining_budget=total_budget - reserved_budget,
        )

        logger.info(
            f"BudgetManager initialized: total={total_budget}, "
            f"available={self._state.remaining_budget}"
        )

    @property
    def remaining(self) -> int:
        """Get remaining budget."""
        return self._state.remaining_budget

    @property
    def used(self) -> int:
        """Get used budget."""
        return self.total_budget - self._state.remaining_budget - self.reserved_budget

    @property
    def has_budget(self) -> bool:
        """Check if budget is available."""
        return self._state.remaining_budget >= self.cost_per_iteration

    def consume(self, amount: int = None) -> bool:
        """
        Consume budget for an iteration.

        Args:
            amount: Amount to consume (default: cost_per_iteration)

        Returns:
            True if budget was consumed, False if insufficient
        """
        amount = amount or self.cost_per_iteration

        if self._state.remaining_budget < amount:
            logger.warning(
                f"Insufficient budget: {self._state.remaining_budget} < {amount}"
            )
            return False

        self._state.remaining_budget -= amount
        self._state.iterations_completed += 1

        logger.debug(
            f"Budget consumed: {amount}, remaining: {self._state.remaining_budget}"
        )

        return True

    def record_api_call(self, calls: int = 1):
        """Record API calls made."""
        self._state.api_calls_used += calls

    def record_compute_time(self, seconds: float):
        """Record compute time used."""
        self._state.compute_time_seconds += seconds

    def get_state(self) -> BudgetState:
        """Get current budget state."""
        return self._state

    def get_summary(self) -> dict:
        """Get budget summary."""
        elapsed = (datetime.utcnow() - self._state.start_time).total_seconds()

        return {
            "total_budget": self.total_budget,
            "remaining_budget": self._state.remaining_budget,
            "used_budget": self.used,
            "iterations_completed": self._state.iterations_completed,
            "api_calls_used": self._state.api_calls_used,
            "compute_time_seconds": self._state.compute_time_seconds,
            "elapsed_time_seconds": elapsed,
            "budget_utilization": (
                self.used / self.total_budget if self.total_budget > 0 else 0
            ),
        }

    def restore(self, state: BudgetState):
        """Restore budget state from checkpoint."""
        self._state = state
        logger.info(f"Budget restored: remaining={self._state.remaining_budget}")
