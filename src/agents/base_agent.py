from abc import ABC, abstractmethod
from typing import Any, Optional
import os

from openai import AsyncOpenAI, RateLimitError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
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
        base_url: Optional[str] = None,
        temperature: float = 0.7,
        max_retries: int = 3,
    ):
        self.key_manager = key_manager
        self.model_name = model_name
        self.base_url = base_url or "https://api.openai.com/v1"
        self.temperature = temperature
        self.max_retries = max_retries
        self._agent: Optional[Agent] = None
        self._current_key: Optional[str] = None
        self._call_count = 0
        self._current_result_type: Optional[type] = None

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
        self._current_result_type = result_type  # Store for recreating agent

        # Set API key in environment for pydantic-ai to use
        # This is a temporary workaround - pydantic-ai will pick it up automatically
        os.environ["OPENAI_API_KEY"] = self._current_key
        if self.base_url != "https://api.openai.com/v1":
            os.environ["OPENAI_BASE_URL"] = self.base_url

        # Create the model - it will use the environment variables
        model = OpenAIChatModel(
            self.model_name,
        )

        agent = Agent(
            model,
            output_type=result_type,
            instructions=self._get_system_prompt(),
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
                agent = await self._create_agent(self._current_result_type)

                last_exception = e

            except Exception as e:
                logger.error(f"Agent error on attempt {attempt + 1}: {e}")
                await self.key_manager.report_error(self._current_key, e)
                last_exception = e

                if attempt < self.max_retries - 1:
                    # Try with different key
                    self._current_key = await self.key_manager.get_key()
                    agent = await self._create_agent(self._current_result_type)

        raise last_exception or RuntimeError("Agent execution failed")

    @property
    def call_count(self) -> int:
        """Get total API calls made by this agent."""
        return self._call_count
