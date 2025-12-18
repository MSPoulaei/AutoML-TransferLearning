# Per-Agent Configuration - Implementation Summary

## Overview

This refactoring adds support for configuring each agent (Analyzer and Executor) with its own API keys, model name, and base URL. This enables:

- Cost optimization (different models per agent)
- Better rate limit management (separate key pools)
- Flexibility to use different API providers per agent
- Support for local LLMs, Azure OpenAI, and custom endpoints

## Changes Made

### 1. Configuration Layer (`config/settings.py`)

**Added Settings:**
- `openai_base_url`: Default base URL for all agents
- `analyzer_api_keys`: API keys specific to Analyzer agent
- `analyzer_model`: Model name for Analyzer agent
- `analyzer_base_url`: Base URL for Analyzer agent
- `executor_api_keys`: API keys specific to Executor agent
- `executor_model`: Model name for Executor agent
- `executor_base_url`: Base URL for Executor agent

**Added Methods:**
- `get_agent_api_keys(agent_name)`: Get API keys for a specific agent with fallback
- `get_agent_model(agent_name)`: Get model name for a specific agent with fallback
- `get_agent_base_url(agent_name)`: Get base URL for a specific agent with fallback

### 2. Base Agent (`src/agents/base_agent.py`)

**Modified Constructor:**
```python
def __init__(
    self,
    key_manager: APIKeyManager,
    model_name: str = "gpt-4o",
    base_url: Optional[str] = None,  # NEW
    temperature: float = 0.7,
    max_retries: int = 3,
):
```

**Modified `_create_agent` Method:**
- Now passes `base_url` parameter to `OpenAIModel`

### 3. Agent Classes

**AnalyzerAgent (`src/agents/analyzer_agent.py`):**
- Added `base_url` parameter to constructor
- Passes `base_url` to `BaseAgent.__init__()`

**ExecutorAgent (`src/agents/executor_agent.py`):**
- Added `base_url` parameter to constructor
- Passes `base_url` to `BaseAgent.__init__()`

### 4. Orchestrator (`src/orchestrator/main_orchestrator.py`)

**Modified Constructor:**
Added parameters for per-agent configuration:
- `base_url`: Default base URL
- `analyzer_api_keys`, `analyzer_model`, `analyzer_base_url`
- `executor_api_keys`, `executor_model`, `executor_base_url`

**Implementation Changes:**
- Creates separate `APIKeyManager` instances for each agent
- Each agent gets its own key pool for independent rotation
- Falls back to default settings if agent-specific settings not provided

**Logging Enhancement:**
- Now logs both analyzer and executor model names at initialization

### 5. Main Entry Point (`main.py`)

**Updated Orchestrator Initialization:**
- Passes `base_url` from settings
- Retrieves and passes per-agent settings using new Settings methods
- Maintains backward compatibility with existing configurations

### 6. Configuration Files

**`config/default_config.yaml`:**
- Added `agents` section with `analyzer` and `executor` subsections
- Documented optional model and base_url overrides

**`.env.example`:**
- Complete rewrite with comprehensive documentation
- Added per-agent configuration examples
- Included examples for Azure OpenAI, local LLMs, and other providers
- Added detailed notes on usage patterns

### 7. Documentation

**Created `docs/AGENT_CONFIGURATION.md`:**
- Complete configuration guide
- Use cases and examples
- Implementation details
- Troubleshooting guide
- Best practices

**Created `docs/MIGRATION_GUIDE.md`:**
- Step-by-step migration instructions
- Backward compatibility explanation
- Code examples (before/after)
- Testing procedures
- Rollback plan

## Backward Compatibility

✅ **Fully backward compatible**

Existing configurations will work without any changes:
- If agent-specific settings are not provided, defaults are used
- Existing `.env` files work as-is
- No breaking changes to public APIs
- Optional parameters with sensible defaults

## Key Features

### 1. Independent API Key Pools

Each agent has its own `APIKeyManager`:
- Separate rate limit tracking
- Independent key rotation
- Better quota management

### 2. Flexible Model Selection

```python
# Example: Cost optimization
ANALYZER_MODEL=gpt-4o        # High-quality for critical decisions
EXECUTOR_MODEL=gpt-4o-mini   # Cost-effective for analysis
```

### 3. Multi-Provider Support

```python
# Example: Mixed providers
ANALYZER_BASE_URL=https://api.openai.com/v1
EXECUTOR_BASE_URL=http://localhost:11434/v1  # Local Ollama
```

### 4. Budget Control

Separate API keys enable per-agent cost tracking and budget limits.

## Testing Recommendations

1. **Basic Test** - Verify backward compatibility:
   ```bash
   python main.py run --num-classes 5 --num-samples 100 \
       --domain natural --domain-desc "Test" --budget 1 --simulation
   ```

2. **Per-Agent Test** - Set agent-specific models in `.env`:
   ```bash
   ANALYZER_MODEL=gpt-4o
   EXECUTOR_MODEL=gpt-4o-mini
   ```
   Run and verify logs show correct models.

3. **Local LLM Test** - Point to local Ollama:
   ```bash
   OPENAI_BASE_URL=http://localhost:11434/v1
   ```

## Files Modified

- `config/settings.py` - Added per-agent settings and helper methods
- `config/default_config.yaml` - Added agents configuration section
- `src/agents/base_agent.py` - Added base_url parameter
- `src/agents/analyzer_agent.py` - Added base_url parameter
- `src/agents/executor_agent.py` - Added base_url parameter
- `src/orchestrator/main_orchestrator.py` - Per-agent initialization
- `main.py` - Pass per-agent settings from config
- `.env.example` - Comprehensive documentation

## Files Created

- `docs/AGENT_CONFIGURATION.md` - Configuration guide
- `docs/MIGRATION_GUIDE.md` - Migration instructions

## Impact Analysis

### Performance
- ✅ No performance impact
- ✅ Separate key managers enable better parallelization

### Security
- ✅ API keys remain in environment variables only
- ✅ No keys in code or config files

### Maintainability
- ✅ Clear separation of concerns
- ✅ Well-documented configuration options
- ✅ Backward compatible design

### Extensibility
- ✅ Easy to add more agents in the future
- ✅ Pattern established for per-agent configuration

## Future Enhancements

Possible extensions based on this foundation:

1. **Per-Agent Temperature/Parameters**: Extend to other generation parameters
2. **Dynamic Model Selection**: Choose model based on task complexity
3. **Agent Pool**: Support for multiple instances of each agent type
4. **Cost Tracking**: Built-in per-agent cost monitoring
5. **Health Checks**: Per-agent API endpoint health monitoring

## Usage Examples

### Standard Usage (Unchanged)
```python
orchestrator = Orchestrator(
    api_keys=["sk-key1", "sk-key2"],
    model_name="gpt-4o",
)
```

### Cost-Optimized
```python
orchestrator = Orchestrator(
    api_keys=["sk-key"],
    model_name="gpt-4o",
    executor_model="gpt-4o-mini",  # Cheaper for analysis
)
```

### Multi-Provider
```python
orchestrator = Orchestrator(
    api_keys=["sk-openai-key"],
    base_url="https://api.openai.com/v1",
    executor_base_url="http://localhost:11434/v1",  # Local LLM
    executor_model="llama3.1:8b",
)
```

## Verification Checklist

- [x] Settings class updated with per-agent fields
- [x] BaseAgent accepts base_url parameter
- [x] AnalyzerAgent passes base_url to BaseAgent
- [x] ExecutorAgent passes base_url to BaseAgent
- [x] Orchestrator creates separate key managers
- [x] Orchestrator passes per-agent settings
- [x] Main.py retrieves and passes settings correctly
- [x] default_config.yaml documented
- [x] .env.example updated with examples
- [x] Configuration guide created
- [x] Migration guide created
- [x] Backward compatibility maintained
- [x] Logging shows agent-specific models

## Conclusion

This refactoring successfully adds per-agent configuration capabilities while maintaining full backward compatibility. The implementation is clean, well-documented, and sets a solid foundation for future enhancements.
