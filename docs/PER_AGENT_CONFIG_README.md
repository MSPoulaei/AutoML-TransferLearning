# Per-Agent Configuration Update

## What's New

The Multi-Agent Transfer Learning Orchestrator now supports **per-agent configuration**, allowing you to customize API keys, models, and base URLs for each agent individually.

## Key Features

### 🎯 Individual Agent Configuration
- **Analyzer Agent**: Configure separately for high-quality model selections
- **Executor Agent**: Configure separately for cost-effective training analysis

### 💰 Cost Optimization
Use different models per agent to balance quality and cost:
```bash
ANALYZER_MODEL=gpt-4o        # High-quality for critical decisions
EXECUTOR_MODEL=gpt-4o-mini   # Cost-effective for analysis
```

### 🔄 Better Rate Limit Management
Separate API key pools per agent enable:
- Independent key rotation
- Better quota distribution
- Reduced rate limit conflicts

### 🌐 Multi-Provider Support
Point each agent to different API endpoints:
- OpenAI API
- Azure OpenAI
- Local LLMs (Ollama, LM Studio)
- Custom OpenAI-compatible endpoints

### 📊 Budget Control
Separate API keys per agent enable:
- Per-agent cost tracking
- Independent budget limits
- Better usage monitoring

## Quick Start

### 1. Update Your .env File

```bash
# Default settings (all agents)
OPENAI_API_KEYS=sk-key1,sk-key2
OPENAI_MODEL=gpt-4o
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Override for specific agents
ANALYZER_MODEL=gpt-4o
EXECUTOR_MODEL=gpt-4o-mini
```

### 2. Run as Usual

```bash
python main.py run \
    --num-classes 10 \
    --num-samples 1000 \
    --domain natural \
    --domain-desc "My dataset" \
    --budget 5
```

## Backward Compatibility

✅ **Fully backward compatible** - Your existing configuration works without any changes!

If you don't specify agent-specific settings, all agents use the default configuration.

## Documentation

- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Cheat sheet for common configurations
- **[Configuration Guide](docs/AGENT_CONFIGURATION.md)** - Comprehensive configuration documentation
- **[Migration Guide](docs/MIGRATION_GUIDE.md)** - Step-by-step migration instructions
- **[Implementation Summary](docs/REFACTORING_SUMMARY.md)** - Technical details of the changes

## Examples

### Cost Optimization
```bash
# Use GPT-4 for Analyzer, GPT-3.5 for Executor
ANALYZER_MODEL=gpt-4o
EXECUTOR_MODEL=gpt-4o-mini
```

### Separate Key Pools
```bash
# Different keys for better rate limit handling
ANALYZER_API_KEYS=sk-a1,sk-a2,sk-a3
EXECUTOR_API_KEYS=sk-e1,sk-e2,sk-e3
```

### Mixed Providers
```bash
# Analyzer uses OpenAI, Executor uses local LLM
ANALYZER_BASE_URL=https://api.openai.com/v1
EXECUTOR_BASE_URL=http://localhost:11434/v1
EXECUTOR_MODEL=llama3.1:8b
```

### Azure OpenAI
```bash
OPENAI_BASE_URL=https://your-resource.openai.azure.com/openai/deployments/your-deployment
OPENAI_API_KEYS=your-azure-key
```

## Configuration Priority

1. Agent-specific environment variable (e.g., `ANALYZER_MODEL`)
2. Default environment variable (e.g., `OPENAI_MODEL`)
3. Hardcoded default in code (`gpt-4o`)

## Getting Started

1. Copy `.env.example` to `.env`
2. Set your API keys
3. (Optional) Configure per-agent settings
4. Run your experiments

## Need Help?

- Check [Quick Reference](docs/QUICK_REFERENCE.md) for common patterns
- See [Configuration Guide](docs/AGENT_CONFIGURATION.md) for detailed options
- Review [Migration Guide](docs/MIGRATION_GUIDE.md) if upgrading from previous version

## Technical Details

For implementation details and architecture decisions, see:
- [Refactoring Summary](docs/REFACTORING_SUMMARY.md)

---

**Note**: This is a non-breaking change. Existing configurations and workflows continue to work without modification.
