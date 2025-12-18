from functools import lru_cache
from pathlib import Path
from typing import Optional
import yaml

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI settings (default/fallback)
    openai_api_keys: str = Field(
        default="",
        description="Comma-separated API keys for rotation (default for all agents)",
    )
    openai_model: str = Field(
        default="gpt-4o", description="OpenAI model to use (default for all agents)"
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="OpenAI API base URL (default for all agents)",
    )

    # Analyzer Agent settings
    analyzer_api_keys: str = Field(
        default="",
        description="Comma-separated API keys for Analyzer agent (overrides default)",
    )
    analyzer_model: str = Field(
        default="", description="Model for Analyzer agent (overrides default)"
    )
    analyzer_base_url: str = Field(
        default="", description="Base URL for Analyzer agent (overrides default)"
    )

    # Executor Agent settings
    executor_api_keys: str = Field(
        default="",
        description="Comma-separated API keys for Executor agent (overrides default)",
    )
    executor_model: str = Field(
        default="", description="Model for Executor agent (overrides default)"
    )
    executor_base_url: str = Field(
        default="", description="Base URL for Executor agent (overrides default)"
    )

    # Logging
    log_level: str = Field(default="INFO")
    log_file: str = Field(default="logs/orchestrator.log")

    # Directories
    experiment_dir: str = Field(default="experiments")
    checkpoint_dir: str = Field(default="checkpoints")

    # Memory
    memory_limit_gb: float = Field(default=15.0)

    # Rate limiting
    rate_limit_retry_delay: int = Field(default=60)
    max_retries_per_key: int = Field(default=3)

    # Database
    database_url: str = Field(default="sqlite:///experiments/experiments.db")

    @property
    def api_keys_list(self) -> list[str]:
        """Parse comma-separated API keys into list."""
        return [k.strip() for k in self.openai_api_keys.split(",") if k.strip()]

    @property
    def memory_limit_bytes(self) -> int:
        """Memory limit in bytes."""
        return int(self.memory_limit_gb * 1024 * 1024 * 1024)

    def get_agent_api_keys(self, agent_name: str) -> list[str]:
        """Get API keys for a specific agent, falling back to default."""
        agent_keys_attr = f"{agent_name.lower()}_api_keys"
        agent_keys = getattr(self, agent_keys_attr, "")
        if agent_keys:
            return [k.strip() for k in agent_keys.split(",") if k.strip()]
        return self.api_keys_list

    def get_agent_model(self, agent_name: str) -> str:
        """Get model name for a specific agent, falling back to default."""
        agent_model_attr = f"{agent_name.lower()}_model"
        agent_model = getattr(self, agent_model_attr, "")
        return agent_model if agent_model else self.openai_model

    def get_agent_base_url(self, agent_name: str) -> str:
        """Get base URL for a specific agent, falling back to default."""
        agent_url_attr = f"{agent_name.lower()}_base_url"
        agent_url = getattr(self, agent_url_attr, "")
        return agent_url if agent_url else self.openai_base_url

    @classmethod
    def load_yaml_config(cls, config_path: Optional[str] = None) -> dict:
        """Load YAML configuration file."""
        if config_path is None:
            config_path = Path(__file__).parent / "default_config.yaml"

        with open(config_path, "r") as f:
            return yaml.safe_load(f)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
