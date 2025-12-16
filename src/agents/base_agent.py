from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import RateLimitError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from src.utils import APIKeyManager, get_logger

logger = get_logger(__name__)


class BaseAgent(ABC):
    """Base class for all agents with common functionality."""

    def __init__(
        self,
        key_manager: APIKeyManager,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.key_manager = key_manager
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self._agent: Optional[Agent] = None
        self._current_key: Optional[str] = None
        self._call_count = 0

    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Run the agent with given inputs."""
        pass

    async def _create_agent(self, result_type: type) -> Agent:
        """Create a pydantic-ai agent with current API key."""
        self._current_key = await self.key_manager.get_key()

        model = OpenAIModel(
            self.model_name,
            api_key=self._current_key,
        )

        agent = Agent(
            model,
            result_type=result_type,
            system_prompt=self._get_system_prompt(),
        )

        return agent

    async def _execute_with_retry(
        self,
        agent: Agent,
        prompt: str,
        deps: Any = None,
    ) -> Any:
        """Execute agent with retry logic and key rotation."""
        last_exception = None

        for attempt in range(self.max_retries):
            try:
                self._call_count += 1

                if deps:
                    result = await agent.run(prompt, deps=deps)
                else:
                    result = await agent.run(prompt)

                # Report success
                await self.key_manager.report_success(self._current_key)

                return result.data

            except RateLimitError as e:
                logger.warning(f"Rate limit hit on attempt {attempt + 1}")

                # Extract retry-after if available
                retry_after = getattr(e, "retry_after", None)
                await self.key_manager.report_rate_limit(self._current_key, retry_after)

                # Get new key and recreate agent
                self._current_key = await self.key_manager.get_key()
                agent = await self._create_agent(type(agent.result_type))

                last_exception = e

            except Exception as e:
                logger.error(f"Agent error on attempt {attempt + 1}: {e}")
                await self.key_manager.report_error(self._current_key, e)
                last_exception = e

                if attempt < self.max_retries - 1:
                    # Try with different key
                    self._current_key = await self.key_manager.get_key()
                    agent = await self._create_agent(type(agent.result_type))

        raise last_exception or RuntimeError("Agent execution failed")

    @property
    def call_count(self) -> int:
        """Get total API calls made by this agent."""
        return self._call_count
