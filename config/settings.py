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

    # OpenAI settings
    openai_api_keys: str = Field(
        default="", description="Comma-separated API keys for rotation"
    )
    openai_model: str = Field(default="gpt-4o", description="OpenAI model to use")

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
