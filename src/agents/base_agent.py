from abc import ABC, abstractmethod
from typing import Any, Optional
import os
import json

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


# Monkey patch to handle non-standard service_tier values from OpenAI-compatible APIs
def patch_openai_models():
    """
    Patch OpenAI response models to handle non-standard values.
    Some providers return values like 'on_demand' for service_tier which aren't in the standard enum.
    """
    try:
        import openai.types.chat.chat_completion as cc_module

        # Store original ChatCompletion class
        OriginalChatCompletion = cc_module.ChatCompletion

        # Check if already patched
        if hasattr(OriginalChatCompletion, "_service_tier_patched"):
            return True

        # Get the original __init__ or model_validate method
        original_model_validate = OriginalChatCompletion.model_validate

        @classmethod
        def patched_model_validate(cls, obj, **kwargs):
            """Patched model_validate that cleans service_tier before validation."""
            # Clean the service_tier field if present
            if isinstance(obj, dict) and "service_tier" in obj:
                valid_tiers = ["auto", "default", "flex", "scale", "priority"]
                if (
                    obj["service_tier"] not in valid_tiers
                    and obj["service_tier"] is not None
                ):
                    logger.debug(
                        f"Removing non-standard service_tier: {obj['service_tier']}"
                    )
                    obj["service_tier"] = None

            return original_model_validate(obj, **kwargs)

        # Apply the patch
        cc_module.ChatCompletion.model_validate = patched_model_validate
        cc_module.ChatCompletion._service_tier_patched = True

        logger.info(
            "Successfully patched OpenAI ChatCompletion to handle non-standard service_tier values"
        )
        return True

    except Exception as e:
        logger.warning(f"Could not patch OpenAI models: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return False


# Apply patches on module import
patch_openai_models()


def sanitize_json_string(text: str) -> str:
    """
    Sanitize a string that might contain problematic Unicode characters for JSON parsing.
    Replaces narrow no-break spaces and other problematic characters with regular spaces.
    """
    # Replace various problematic Unicode characters
    replacements = {
        "\u202f": " ",  # Narrow no-break space
        "\u00a0": " ",  # Non-breaking space
        "\u2009": " ",  # Thin space
        "\u200b": "",  # Zero-width space
        "\u2013": "-",  # En dash
        "\u2014": "-",  # Em dash
        "\u2018": "'",  # Left single quotation mark
        "\u2019": "'",  # Right single quotation mark
        "\u201c": '"',  # Left double quotation mark
        "\u201d": '"',  # Right double quotation mark
    }

    for unicode_char, replacement in replacements.items():
        text = text.replace(unicode_char, replacement)

    return text


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
        # Configure model settings to avoid issues with non-standard API responses
        model = OpenAIChatModel(
            self.model_name,
            # Don't request service_tier in responses to avoid validation errors
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

                # In pydantic-ai >= 1.33, the return value is AgentRunResult with .output attribute
                return result.output

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
                error_msg = str(e)
                logger.error(f"Agent error on attempt {attempt + 1}: {e}")

                # Check if it's a JSON parsing error with Unicode characters
                if "Failed to parse tool call arguments as JSON" in error_msg:
                    logger.warning(
                        "JSON parsing error detected - likely due to Unicode characters in LLM response"
                    )
                    # This is a model output issue, not a key issue - continue with same key
                    if attempt < self.max_retries - 1:
                        logger.info("Retrying with same key (model output issue)...")
                        last_exception = e
                        continue

                # Check if it's a validation error (like service_tier)
                if (
                    "validation error" in error_msg.lower()
                    and "service_tier" in error_msg
                ):
                    logger.warning(
                        "Service tier validation error - this is an API compatibility issue"
                    )
                    # This is an API compatibility issue, trying another key might help
                    await self.key_manager.report_error(self._current_key, e)
                else:
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
