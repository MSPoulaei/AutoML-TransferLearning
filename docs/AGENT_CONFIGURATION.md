# Agent Configuration Guide

This document explains how to configure individual agents with their own API keys, models, and base URLs in the Multi-Agent Transfer Learning Orchestrator.

## Overview

The system now supports per-agent configuration, allowing you to:
- Use different API keys for each agent
- Use different models for each agent (e.g., GPT-4 for Analyzer, GPT-3.5 for Executor)
- Point each agent to different API endpoints (OpenAI, Azure, local LLMs, etc.)
- Control costs by using lighter models for less critical tasks
- Better monitor and control API usage per agent

## Architecture

The project has two main agents:
1. **Analyzer Agent**: Analyzes dataset information and recommends optimal training configurations
2. **Executor Agent**: Executes training and analyzes results

Each agent can have its own:
- API keys (with automatic rotation on rate limits)
- Model name
- Base URL (API endpoint)

## Configuration Methods

### 1. Environment Variables (.env file)

Create a `.env` file in the project root (or copy from `.env.example`):

```bash
# Default settings (used if agent-specific settings are not provided)
OPENAI_API_KEYS=sk-default-key-1,sk-default-key-2
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# Analyzer Agent specific settings (optional)
ANALYZER_API_KEYS=sk-analyzer-key-1,sk-analyzer-key-2
ANALYZER_MODEL=gpt-4o
ANALYZER_BASE_URL=https://api.openai.com/v1

# Executor Agent specific settings (optional)
EXECUTOR_API_KEYS=sk-executor-key-1,sk-executor-key-2
EXECUTOR_MODEL=gpt-4o-mini
EXECUTOR_BASE_URL=https://api.openai.com/v1
```

### 2. Configuration File (default_config.yaml)

The `config/default_config.yaml` file includes a section for agent-specific settings:

```yaml
agents:
  analyzer:
    # model: "gpt-4o"
    # base_url: "https://api.openai.com/v1"
  
  executor:
    # model: "gpt-4o-mini"
    # base_url: "https://api.openai.com/v1"
```

**Note**: API keys should only be set via environment variables, never in the YAML config file.

## Configuration Priority

The system uses the following priority (highest to lowest):

1. **Agent-specific environment variables** (e.g., `ANALYZER_MODEL`)
2. **Default environment variables** (e.g., `OPENAI_MODEL`)
3. **Hardcoded defaults** in the code

## Use Cases

### Use Case 1: Cost Optimization

Use a more expensive, powerful model for the Analyzer (critical decisions) and a cheaper model for the Executor (analysis):

```bash
# .env
OPENAI_API_KEYS=sk-default-key

# Use GPT-4 for Analyzer (better recommendations)
ANALYZER_MODEL=gpt-4o

# Use GPT-3.5 for Executor (cost-effective for analysis)
EXECUTOR_MODEL=gpt-4o-mini
```

### Use Case 2: Different API Providers

Use OpenAI for Analyzer and Azure OpenAI for Executor:

```bash
# .env
OPENAI_API_KEYS=sk-openai-key
OPENAI_BASE_URL=https://api.openai.com/v1

EXECUTOR_API_KEYS=azure-key
EXECUTOR_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
EXECUTOR_MODEL=gpt-4
```

### Use Case 3: Local LLMs

Use local models via Ollama or LM Studio:

```bash
# .env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEYS=not-needed-for-local

ANALYZER_MODEL=llama3.1:70b  # Larger model for better analysis
EXECUTOR_MODEL=llama3.1:8b   # Smaller model for faster execution
```

### Use Case 4: Separate Budget Control

Use different API keys per agent to track costs and control budgets separately:

```bash
# .env
# Analyzer gets its own keys for budget tracking
ANALYZER_API_KEYS=sk-analyzer-budget-key-1,sk-analyzer-budget-key-2

# Executor gets separate keys
EXECUTOR_API_KEYS=sk-executor-budget-key-1,sk-executor-budget-key-2
```

### Use Case 5: Rate Limit Management

Distribute agents across different API key pools to avoid rate limits:

```bash
# .env
# Analyzer uses one set of keys
ANALYZER_API_KEYS=sk-key-1,sk-key-2,sk-key-3

# Executor uses a different set
EXECUTOR_API_KEYS=sk-key-4,sk-key-5,sk-key-6
```

## Implementation Details

### BaseAgent Class

The `BaseAgent` class now accepts:
```python
def __init__(
    self,
    key_manager: APIKeyManager,
    model_name: str = "gpt-4o",
    base_url: Optional[str] = None,
    temperature: float = 0.7,
    max_retries: int = 3,
):
```

### Agent Initialization

Each agent gets its own `APIKeyManager` instance with its specific keys:

```python
analyzer_key_manager = APIKeyManager(api_keys=analyzer_keys, ...)
executor_key_manager = APIKeyManager(api_keys=executor_keys, ...)

analyzer = AnalyzerAgent(
    key_manager=analyzer_key_manager,
    model_name=analyzer_model,
    base_url=analyzer_base_url,
)

executor = ExecutorAgent(
    key_manager=executor_key_manager,
    model_name=executor_model,
    base_url=executor_base_url,
)
```

## API Key Rotation

Each agent's `APIKeyManager` handles:
- Round-robin rotation through available keys
- Automatic cooldown on rate limit errors
- Failure tracking per key
- Async-safe key management

When an agent hits a rate limit:
1. The current key is marked as rate-limited
2. The agent automatically switches to the next available key
3. Training continues without interruption
4. Rate-limited keys are re-enabled after cooldown period

## Monitoring and Logging

The system logs agent-specific information:

```
INFO: Orchestrator initialized: analyzer_model=gpt-4o, executor_model=gpt-4o-mini, memory=15.0GB
INFO: APIKeyManager initialized with 3 keys (Analyzer)
INFO: APIKeyManager initialized with 2 keys (Executor)
```

## Testing Configuration

To test your configuration:

```bash
# Run with simulation mode to test without real training
python main.py run \
    --num-classes 10 \
    --num-samples 1000 \
    --domain natural \
    --domain-desc "Test dataset" \
    --budget 2 \
    --simulation
```

Check logs for agent initialization details and API usage.

## Troubleshooting

### Issue: Agent uses wrong API key

**Solution**: Ensure agent-specific environment variables are set correctly:
```bash
# Check your .env file
cat .env | grep ANALYZER
cat .env | grep EXECUTOR
```

### Issue: "No API keys found" error

**Solution**: Ensure at least the default `OPENAI_API_KEYS` is set:
```bash
# .env
OPENAI_API_KEYS=sk-your-key-here
```

### Issue: Agent uses wrong model

**Solution**: Agent-specific model settings override defaults. Check precedence:
1. `ANALYZER_MODEL` or `EXECUTOR_MODEL`
2. `OPENAI_MODEL`
3. Hardcoded default (`gpt-4o`)

### Issue: Cannot connect to custom base URL

**Solution**: Verify the base URL is accessible and includes the full path:
```bash
# Correct format
OPENAI_BASE_URL=https://api.openai.com/v1

# For Azure (include deployment path)
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment

# For local Ollama
OPENAI_BASE_URL=http://localhost:11434/v1
```

## Best Practices

1. **Keep API keys in .env**: Never commit API keys to version control
2. **Use multiple keys**: Enable automatic rotation on rate limits
3. **Separate agent budgets**: Use different keys per agent for better cost tracking
4. **Test locally first**: Use simulation mode to test configuration
5. **Monitor usage**: Check logs to verify agents use correct models/endpoints
6. **Document custom setups**: If using custom providers, document the configuration

## Example Configurations

See `.env.example` for more example configurations including:
- Azure OpenAI setup
- Local LLM integration (Ollama, LM Studio)
- Mixed provider setup
- Cost optimization strategies

## Related Files

- `config/settings.py`: Settings class with agent-specific properties
- `src/agents/base_agent.py`: Base agent implementation with base_url support
- `src/orchestrator/main_orchestrator.py`: Orchestrator with per-agent initialization
- `main.py`: CLI entry point that reads and applies settings
- `.env.example`: Example environment configuration
