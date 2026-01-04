# Codex Django Bug-Fix Environment

A function-based environment that challenges LLMs to fix bugs in Django.

## How It Works

1. **Bug Injection**: Uses an LLM (via Chutes API) to introduce a subtle, realistic bug into a Django function
2. **Challenge Generation**: Creates a prompt asking the model to fix the buggy code
3. **Evaluation**: Runs Django's test suite to verify if the fix works

## Environment Type

This is a **function-based** environment. The `env.py` file contains an `Actor` class with methods that affinetes exposes via HTTP. The HTTP server is automatically injected during the image build process.

## Files

- `env.py` - Actor class with `evaluate()` method
- `codex_task.py` - Bug injection and evaluation logic
- `models.py` - Pydantic data models (Challenge, BugSpec)
- `Dockerfile` - Container image definition
- `requirements.txt` - Python dependencies

## Quick Start

### Build

```bash
afs build environments/codex --tag codex:latest
```

### Run

```bash
# On macOS/Docker Desktop (uses port publishing):
afs run codex:latest --env CHUTES_API_KEY=your_key --name codex --host-network

# Or specify environment variable from shell:
export CHUTES_API_KEY=your_key
afs run codex:latest --name codex --host-network
```

### Call Evaluate

```bash
# Run a single evaluation
afs call codex evaluate --arg seed=42

# With custom parameters
afs call codex evaluate \
  --arg seed=123 \
  --arg model=deepseek-ai/DeepSeek-V3 \
  --arg temperature=0.5
```

### Stop

```bash
docker stop codex && docker rm codex
```

## API Methods

### `evaluate()`

Run a single bug-fix evaluation:

Parameters:
- `model` - LLM model to evaluate (default: deepseek-ai/DeepSeek-V3)
- `base_url` - API endpoint (default: https://llm.chutes.ai/v1)
- `api_key` - Chutes API key (or uses CHUTES_API_KEY env var)
- `seed` - Random seed for reproducibility
- `task_id` - Optional task ID
- `temperature` - LLM temperature (default: 0.7)
- `timeout` - Timeout in seconds (default: 600)
- `bug_injection_model` - Separate model for bug injection (optional)

### Response Format

```json
{
  "task_name": "codex_django_bugfix",
  "score": 1.0,
  "success": true,
  "time_taken": 57.5,
  "extra": {
    "file_path": "django/utils/text.py",
    "bug_description": "Bug introduced in slugify",
    "test_result": "1/1 - Code restored to original",
    "conversation": [...],
    "usage": {...}
  }
}
```

## Target Functions

Bugs are injected into these Django functions:
- `django/utils/text.py`: slugify, capfirst, wrap, normalize_newlines
- `django/utils/html.py`: escape, strip_tags, escapejs
- `django/core/validators.py`: validate_integer, validate_ipv4_address
- `django/utils/http.py`: base36_to_int, int_to_base36, urlencode

## Scoring

- **1.0**: Model successfully fixed the bug and all tests pass
- **0.0**: Model failed to fix the bug or introduced new errors

## Notes

- On **macOS with Docker Desktop**, use `--host-network` flag for `afs run`
- The `--host-network` flag on macOS actually uses port publishing (8000:8000) since host network mode doesn't work on Docker Desktop
