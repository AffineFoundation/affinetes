# Affinetes Project Instructions

Lightweight container orchestration framework for Python environments.

## Project Structure

```
affinetes/
├── affinetes/          # Core framework code
├── environments/       # Environment implementations
│   ├── SWE-SYNTH/     # Synthetic SWE-bench evaluation
│   ├── primeintellect/ # Logic games environments
│   └── ...
├── examples/           # Usage examples
└── .claude/commands/   # Claude Code skills
```

## Available Skills (Slash Commands)

| Command | Description |
|---------|-------------|
| `/afs-build` | Build Docker image from environment |
| `/afs-run` | Start environment container |
| `/afs-call` | Call method on running container |
| `/afs-validate` | Validate seed consistency |
| `/afs-test` | Full test workflow (build + run + call) |

## Quick Start

```bash
# Activate virtual environment
source .venv/bin/activate

# Build and test an environment
afs build environments/SWE-SYNTH --tag swe-synth:v1
afs run swe-synth:v1 --name swe-synth --env CHUTES_API_KEY=$CHUTES_API_KEY
afs call swe-synth evaluate --arg task_id=1
```

## Environment Development

Each environment requires:
- `env.py` - Actor class with async methods
- `Dockerfile` - Build configuration
- `requirements.txt` (optional) - Python dependencies

## Testing Guidelines

1. Always validate seed consistency before deployment
2. Test with multiple task_ids to ensure robustness
3. Check cache behavior (R2 storage) for production environments
