# Migration Guide: Per-Agent Configuration

This guide helps you migrate from the previous single-configuration system to the new per-agent configuration system.

## What Changed?

### Previous System (Before Refactoring)

All agents shared the same:
- API keys (single pool)
- Model name
- No support for different base URLs

Configuration in `.env`:
```bash
OPENAI_API_KEYS=sk-key1,sk-key2
OPENAI_MODEL=gpt-4o
```

### New System (After Refactoring)

Each agent can have its own:
- API keys (separate pools with independent rotation)
- Model name
- Base URL (API endpoint)

Configuration in `.env`:
```bash
# Default (backward compatible)
OPENAI_API_KEYS=sk-key1,sk-key2
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional per-agent overrides
ANALYZER_API_KEYS=sk-analyzer-key
ANALYZER_MODEL=gpt-4o
ANALYZER_BASE_URL=https://api.openai.com/v1

EXECUTOR_API_KEYS=sk-executor-key
EXECUTOR_MODEL=gpt-4o-mini
EXECUTOR_BASE_URL=https://api.openai.com/v1
```

## Migration Steps

### Step 1: Update Your .env File (Optional)

If you want to keep the same configuration (all agents share settings):

**No action needed!** The new system is backward compatible. Your existing `.env` will work as-is.

```bash
# Your existing .env - still works!
OPENAI_API_KEYS=sk-key1,sk-key2,sk-key3
OPENAI_MODEL=gpt-4o
```

### Step 2: Add Base URL (Recommended)

Add the base URL setting for clarity (defaults to OpenAI if not set):

```bash
OPENAI_API_KEYS=sk-key1,sk-key2,sk-key3
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1  # Add this line
```

### Step 3: Configure Per-Agent Settings (Optional)

If you want different settings per agent:

```bash
# Default settings (fallback)
OPENAI_API_KEYS=sk-default-key
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# Analyzer-specific (optional)
ANALYZER_API_KEYS=sk-analyzer-key-1,sk-analyzer-key-2
ANALYZER_MODEL=gpt-4o

# Executor-specific (optional)
EXECUTOR_API_KEYS=sk-executor-key-1,sk-executor-key-2
EXECUTOR_MODEL=gpt-4o-mini  # Cheaper model for analysis
```

## Code Changes

### If You're Using the Orchestrator Directly

#### Before (Old Code):
```python
from src.orchestrator import Orchestrator

orchestrator = Orchestrator(
    api_keys=["sk-key1", "sk-key2"],
    model_name="gpt-4o",
    memory_limit_gb=15.0,
    simulation_mode=True,
)
```

#### After (New Code - Backward Compatible):
```python
from src.orchestrator import Orchestrator

# Still works exactly as before!
orchestrator = Orchestrator(
    api_keys=["sk-key1", "sk-key2"],
    model_name="gpt-4o",
    memory_limit_gb=15.0,
    simulation_mode=True,
)
```

#### After (New Code - With Per-Agent Settings):
```python
from src.orchestrator import Orchestrator

orchestrator = Orchestrator(
    # Default settings
    api_keys=["sk-default-key"],
    model_name="gpt-4o",
    base_url="https://api.openai.com/v1",
    memory_limit_gb=15.0,
    simulation_mode=True,
    
    # Per-agent overrides (optional)
    analyzer_api_keys=["sk-analyzer-key"],
    analyzer_model="gpt-4o",
    analyzer_base_url="https://api.openai.com/v1",
    
    executor_api_keys=["sk-executor-key"],
    executor_model="gpt-4o-mini",
    executor_base_url="https://api.openai.com/v1",
)
```

### If You're Creating Agents Directly

#### Before (Old Code):
```python
from src.agents import AnalyzerAgent
from src.utils import APIKeyManager

key_manager = APIKeyManager(api_keys=["sk-key1", "sk-key2"])
analyzer = AnalyzerAgent(
    key_manager=key_manager,
    model_name="gpt-4o",
)
```

#### After (New Code):
```python
from src.agents import AnalyzerAgent
from src.utils import APIKeyManager

key_manager = APIKeyManager(api_keys=["sk-key1", "sk-key2"])
analyzer = AnalyzerAgent(
    key_manager=key_manager,
    model_name="gpt-4o",
    base_url="https://api.openai.com/v1",  # New parameter (optional)
)
```

## Testing Your Migration

### 1. Verify Configuration

```bash
# Check your .env file
cat .env

# Run a simple test with simulation
python main.py run \
    --num-classes 5 \
    --num-samples 100 \
    --domain natural \
    --domain-desc "Test migration" \
    --budget 1 \
    --simulation
```

### 2. Check Logs

Look for initialization messages:
```
INFO: Orchestrator initialized: analyzer_model=gpt-4o, executor_model=gpt-4o-mini, memory=15.0GB
INFO: APIKeyManager initialized with 2 keys (Analyzer)
INFO: APIKeyManager initialized with 2 keys (Executor)
```

### 3. Verify Agent Behavior

The agents should work exactly as before if you didn't change the configuration.

## Breaking Changes

### None for Basic Usage

If you're using:
- The CLI (`main.py`)
- Standard configuration via `.env`
- The `Orchestrator` class with basic parameters

**You don't need to change anything!**

### Potential Issues

#### Custom Orchestrator Initialization

If you're directly instantiating the `Orchestrator` with all parameters, you might need to adjust:

```python
# Old signature
Orchestrator(api_keys, model_name, memory_limit_gb, ...)

# New signature (backward compatible)
Orchestrator(
    api_keys, 
    model_name,
    base_url="https://api.openai.com/v1",  # New optional parameter
    memory_limit_gb,
    ...,
    # New optional parameters
    analyzer_api_keys=None,
    analyzer_model=None,
    analyzer_base_url=None,
    executor_api_keys=None,
    executor_model=None,
    executor_base_url=None,
)
```

#### Custom Agent Instantiation

If you're creating agents directly:

```python
# Old signature
AnalyzerAgent(key_manager, model_name, memory_limit_gb)
ExecutorAgent(key_manager, model_name, simulation_mode, data_dir)

# New signature (backward compatible)
AnalyzerAgent(key_manager, model_name, base_url=None, memory_limit_gb)
ExecutorAgent(key_manager, model_name, base_url=None, simulation_mode, data_dir)
```

## Rollback Plan

If you need to rollback, you can:

1. **Keep using the new code with old configuration**: The system is backward compatible
2. **Git revert**: Use git to revert to the previous commit
3. **No data loss**: All experiments and checkpoints remain compatible

## New Features Available

After migration, you can:

1. **Use different models per agent** for cost optimization
2. **Point agents to different APIs** (OpenAI, Azure, local LLMs)
3. **Separate API key pools** for better rate limit handling
4. **Independent budget control** per agent
5. **Better monitoring** of agent-specific API usage

## Example: Quick Cost Optimization

Want to save costs? Use GPT-4 for critical decisions, GPT-3.5 for analysis:

```bash
# Add to your .env
ANALYZER_MODEL=gpt-4o        # High-quality recommendations
EXECUTOR_MODEL=gpt-4o-mini   # Cost-effective analysis
```

No other changes needed!

## Support

If you encounter issues:

1. Check your `.env` file format
2. Review the logs for initialization messages
3. Test with `--simulation` mode first
4. See `docs/AGENT_CONFIGURATION.md` for detailed configuration guide
5. Verify API keys are valid and have sufficient quota

## Summary

- ✅ Backward compatible - existing configs work as-is
- ✅ Optional per-agent settings for advanced use cases
- ✅ No breaking changes for standard usage
- ✅ Easy to adopt incrementally
- ✅ Better flexibility and cost control

**Most users don't need to change anything!**
