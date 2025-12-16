import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional

from loguru import logger


@dataclass
class KeyState:
    """State tracking for a single API key."""

    key: str
    failures: int = 0
    last_failure_time: Optional[float] = None
    is_rate_limited: bool = False
    rate_limit_reset_time: Optional[float] = None
    total_calls: int = 0


class APIKeyManager:
    """
    Manages multiple API keys with automatic rotation on rate limits.

    Features:
    - Round-robin rotation
    - Automatic cooldown on rate limit errors
    - Failure tracking per key
    - Async-safe with locks
    """

    def __init__(
        self,
        api_keys: list[str],
        max_failures_per_key: int = 3,
        rate_limit_cooldown: int = 60,
    ):
        if not api_keys:
            raise ValueError("At least one API key is required")

        self._keys: list[KeyState] = [KeyState(key=k) for k in api_keys]
        self._current_index: int = 0
        self._max_failures = max_failures_per_key
        self._cooldown = rate_limit_cooldown
        self._lock = asyncio.Lock()

        logger.info(f"APIKeyManager initialized with {len(api_keys)} keys")

    async def get_key(self) -> str:
        """Get the next available API key."""
        async with self._lock:
            return self._get_available_key()

    def _get_available_key(self) -> str:
        """Internal method to find an available key."""
        current_time = time.time()
        attempts = 0

        while attempts < len(self._keys):
            key_state = self._keys[self._current_index]

            # Check if rate limit has expired
            if key_state.is_rate_limited:
                if (
                    key_state.rate_limit_reset_time
                    and current_time >= key_state.rate_limit_reset_time
                ):
                    key_state.is_rate_limited = False
                    key_state.failures = 0
                    logger.info(
                        f"Key {self._mask_key(key_state.key)} rate limit expired"
                    )

            # Check if key is available
            if (
                not key_state.is_rate_limited
                and key_state.failures < self._max_failures
            ):
                key_state.total_calls += 1
                self._rotate_index()
                return key_state.key

            self._rotate_index()
            attempts += 1

        # All keys exhausted - find the one with shortest wait time
        min_wait_key = min(
            self._keys, key=lambda k: k.rate_limit_reset_time or float("inf")
        )

        if min_wait_key.rate_limit_reset_time:
            wait_time = max(0, min_wait_key.rate_limit_reset_time - current_time)
            logger.warning(f"All keys rate limited. Shortest wait: {wait_time:.1f}s")

        # Return the key with least failures
        return min(self._keys, key=lambda k: k.failures).key

    def _rotate_index(self):
        """Move to next key index."""
        self._current_index = (self._current_index + 1) % len(self._keys)

    async def report_rate_limit(self, key: str, retry_after: Optional[int] = None):
        """Report a rate limit error for a key."""
        async with self._lock:
            for key_state in self._keys:
                if key_state.key == key:
                    key_state.is_rate_limited = True
                    key_state.failures += 1
                    cooldown = retry_after or self._cooldown
                    key_state.rate_limit_reset_time = time.time() + cooldown
                    logger.warning(
                        f"Key {self._mask_key(key)} rate limited. "
                        f"Cooldown: {cooldown}s, Failures: {key_state.failures}"
                    )
                    break

    async def report_success(self, key: str):
        """Report successful API call."""
        async with self._lock:
            for key_state in self._keys:
                if key_state.key == key:
                    # Reduce failure count on success
                    key_state.failures = max(0, key_state.failures - 1)
                    break

    async def report_error(self, key: str, error: Exception):
        """Report a non-rate-limit error."""
        async with self._lock:
            for key_state in self._keys:
                if key_state.key == key:
                    key_state.failures += 1
                    key_state.last_failure_time = time.time()
                    logger.error(
                        f"Key {self._mask_key(key)} error: {error}. "
                        f"Failures: {key_state.failures}"
                    )
                    break

    def _mask_key(self, key: str) -> str:
        """Mask API key for logging."""
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    def get_stats(self) -> dict:
        """Get statistics about key usage."""
        return {
            "total_keys": len(self._keys),
            "available_keys": sum(
                1
                for k in self._keys
                if not k.is_rate_limited and k.failures < self._max_failures
            ),
            "rate_limited_keys": sum(1 for k in self._keys if k.is_rate_limited),
            "total_calls": sum(k.total_calls for k in self._keys),
            "keys": [
                {
                    "masked": self._mask_key(k.key),
                    "calls": k.total_calls,
                    "failures": k.failures,
                    "rate_limited": k.is_rate_limited,
                }
                for k in self._keys
            ],
        }
