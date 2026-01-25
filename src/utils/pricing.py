# src/utils/pricing.py
"""
OpenAI API pricing constants for token usage and cost tracking.

Updated: January 2026
Source: https://openai.com/pricing
"""

from typing import Dict, Optional


# Pricing per 1M tokens (in USD)
OPENAI_PRICING: Dict[str, Dict[str, float]] = {
    # GPT-4o models
    "gpt-4o": {
        "input": 2.50,  # $2.50 per 1M input tokens
        "output": 10.00,  # $10.00 per 1M output tokens
    },
    "gpt-4o-2024-11-20": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-2024-08-06": {
        "input": 2.50,
        "output": 10.00,
    },
    "gpt-4o-2024-05-13": {
        "input": 5.00,
        "output": 15.00,
    },
    "gpt-4o-mini": {
        "input": 0.15,  # $0.15 per 1M input tokens
        "output": 0.60,  # $0.60 per 1M output tokens
    },
    "gpt-4o-mini-2024-07-18": {
        "input": 0.15,
        "output": 0.60,
    },
    # GPT-4 Turbo models
    "gpt-4-turbo": {
        "input": 10.00,
        "output": 30.00,
    },
    "gpt-4-turbo-2024-04-09": {
        "input": 10.00,
        "output": 30.00,
    },
    "gpt-4-turbo-preview": {
        "input": 10.00,
        "output": 30.00,
    },
    # GPT-4 models
    "gpt-4": {
        "input": 30.00,
        "output": 60.00,
    },
    "gpt-4-0613": {
        "input": 30.00,
        "output": 60.00,
    },
    "gpt-4-32k": {
        "input": 60.00,
        "output": 120.00,
    },
    # GPT-3.5 Turbo models
    "gpt-3.5-turbo": {
        "input": 0.50,
        "output": 1.50,
    },
    "gpt-3.5-turbo-0125": {
        "input": 0.50,
        "output": 1.50,
    },
    "gpt-3.5-turbo-1106": {
        "input": 1.00,
        "output": 2.00,
    },
}


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """
    Get pricing information for a specific model.

    Args:
        model_name: Name of the OpenAI model

    Returns:
        Dictionary with 'input' and 'output' pricing per 1M tokens
    """
    # Try exact match first
    if model_name in OPENAI_PRICING:
        return OPENAI_PRICING[model_name]

    # Try to match base model name (remove version suffixes)
    base_model = model_name.split("-")[0:2]  # e.g., "gpt-4o" from "gpt-4o-custom"
    base_key = "-".join(base_model)

    if base_key in OPENAI_PRICING:
        return OPENAI_PRICING[base_key]

    # Default to gpt-4o pricing if unknown
    return OPENAI_PRICING["gpt-4o"]


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost of an API call based on token usage.

    Args:
        model_name: Name of the OpenAI model
        input_tokens: Number of input tokens used
        output_tokens: Number of output tokens generated

    Returns:
        Cost in USD
    """
    pricing = get_model_pricing(model_name)

    # Pricing is per 1M tokens, so divide by 1,000,000
    input_cost = (input_tokens / 1_000_000) * pricing["input"]
    output_cost = (output_tokens / 1_000_000) * pricing["output"]

    return input_cost + output_cost


def format_cost(cost: float) -> str:
    """
    Format cost for display.

    Args:
        cost: Cost in USD

    Returns:
        Formatted string (e.g., "$0.0123" or "$1.23")
    """
    if cost < 0.01:
        return f"${cost:.4f}"
    elif cost < 1.0:
        return f"${cost:.3f}"
    else:
        return f"${cost:.2f}"


def get_all_models() -> list[str]:
    """
    Get list of all supported model names.

    Returns:
        List of model names
    """
    return list(OPENAI_PRICING.keys())


# Token limits for different models (context window sizes)
MODEL_TOKEN_LIMITS: Dict[str, int] = {
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-3.5-turbo": 16_385,
}


def get_token_limit(model_name: str) -> int:
    """
    Get the token limit (context window) for a model.

    Args:
        model_name: Name of the OpenAI model

    Returns:
        Maximum number of tokens (context window size)
    """
    # Try exact match
    if model_name in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[model_name]

    # Try base model
    base_model = model_name.split("-")[0:2]
    base_key = "-".join(base_model)

    if base_key in MODEL_TOKEN_LIMITS:
        return MODEL_TOKEN_LIMITS[base_key]

    # Default to 128k for newer models
    return 128_000
