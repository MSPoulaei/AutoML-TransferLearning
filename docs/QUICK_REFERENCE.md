# Quick Reference: Per-Agent Configuration

## Environment Variables

### Default Settings (used by all agents unless overridden)
```bash
OPENAI_API_KEYS=sk-key1,sk-key2,sk-key3
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1
```

### Analyzer Agent (optional overrides)
```bash
ANALYZER_API_KEYS=sk-analyzer-key-1,sk-analyzer-key-2
ANALYZER_MODEL=gpt-4o
ANALYZER_BASE_URL=https://api.openai.com/v1
```

### Executor Agent (optional overrides)
```bash
EXECUTOR_API_KEYS=sk-executor-key-1,sk-executor-key-2
EXECUTOR_MODEL=gpt-4o-mini
EXECUTOR_BASE_URL=https://api.openai.com/v1
```

## Common Configurations

### 1. All agents share same settings
```bash
OPENAI_API_KEYS=sk-key1,sk-key2
OPENAI_MODEL=gpt-4o
```

### 2. Cost optimization (different models)
```bash
OPENAI_API_KEYS=sk-key
ANALYZER_MODEL=gpt-4o        # High quality
EXECUTOR_MODEL=gpt-4o-mini   # Cost effective
```

### 3. Separate key pools (better rate limiting)
```bash
ANALYZER_API_KEYS=sk-a1,sk-a2,sk-a3
EXECUTOR_API_KEYS=sk-e1,sk-e2,sk-e3
```

### 4. Mixed providers
```bash
# Analyzer uses OpenAI
ANALYZER_API_KEYS=sk-openai-key
ANALYZER_BASE_URL=https://api.openai.com/v1

# Executor uses local LLM
EXECUTOR_BASE_URL=http://localhost:11434/v1
EXECUTOR_MODEL=llama3.1:8b
```

### 5. Azure OpenAI
```bash
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
OPENAI_API_KEYS=your-azure-key
OPENAI_MODEL=gpt-4
```

## Priority Order

1. Agent-specific env var (e.g., `ANALYZER_MODEL`)
2. Default env var (e.g., `OPENAI_MODEL`)
3. Code default (`gpt-4o`)

## CLI Usage

```bash
# Standard usage (all defaults)
python main.py run \
    --num-classes 10 \
    --num-samples 1000 \
    --domain natural \
    --domain-desc "My dataset" \
    --budget 5

# With simulation
python main.py run \
    --num-classes 10 \
    --num-samples 1000 \
    --domain natural \
    --domain-desc "My dataset" \
    --budget 5 \
    --simulation
```

## Quick Check

Test your configuration:
```bash
# 1. Verify .env
cat .env | grep -E "(OPENAI|ANALYZER|EXECUTOR)"

# 2. Quick test run
python main.py run --num-classes 5 --num-samples 100 \
    --domain natural --domain-desc "Test" --budget 1 --simulation

# 3. Check logs
cat logs/orchestrator.log | grep "initialized"
```

## Code Example

```python
from src.orchestrator import Orchestrator

# Simple (all agents use same settings)
orchestrator = Orchestrator(
    api_keys=["sk-key"],
    model_name="gpt-4o",
)

# Per-agent settings
orchestrator = Orchestrator(
    api_keys=["sk-default"],
    model_name="gpt-4o",
    
    analyzer_model="gpt-4o",
    executor_model="gpt-4o-mini",
)
```

## Troubleshooting

| Issue                | Solution                                          |
| -------------------- | ------------------------------------------------- |
| "No API keys found"  | Set `OPENAI_API_KEYS` in `.env`                   |
| Wrong model used     | Check agent-specific env vars override defaults   |
| Can't connect to API | Verify `BASE_URL` is correct and accessible       |
| Rate limits          | Use multiple keys or separate key pools per agent |

## Files

- `.env` - Your configuration (create from `.env.example`)
- `config/settings.py` - Settings class
- `docs/AGENT_CONFIGURATION.md` - Full documentation
- `docs/MIGRATION_GUIDE.md` - Migration guide

## Support

For detailed information, see:
- Full configuration guide: `docs/AGENT_CONFIGURATION.md`
- Migration instructions: `docs/MIGRATION_GUIDE.md`
- Implementation details: `docs/REFACTORING_SUMMARY.md`
