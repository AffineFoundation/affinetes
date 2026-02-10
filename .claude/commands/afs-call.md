# Call Affinetes Environment Method

Call a method on a running affinetes environment container.

## Usage

```
/afs-call <container_name> <method> [--arg KEY=VALUE...] [--json JSON_STRING]
```

## Arguments

- `container_name`: Name of running container
- `method`: Method name to call (e.g., `evaluate`)
- `--arg`: Method arguments (can be specified multiple times)
- `--json`: JSON-formatted arguments
- `--timeout`: Timeout in seconds (default: 300)

## Argument Formats

- Simple values: `--arg task_id=10`
- Strings: `--arg model="deepseek-ai/DeepSeek-V3"`
- Lists: `--arg ids=[10,20,30]`
- JSON: `--json '{"task_id": 10, "model": "gpt-4"}'`

## Instructions

1. Ensure the container is running (`docker ps`)
2. Call the method:
   ```bash
   afs call <name> <method> --arg key=value
   ```
3. Wait for the result (may take time for long-running methods)

## Examples

```bash
# Simple call
afs call swe-synth evaluate --arg task_id=1

# With model and API parameters
afs call swe-synth evaluate \
  --arg task_id=1 \
  --arg model="deepseek-ai/DeepSeek-V3" \
  --arg base_url="https://llm.chutes.ai/v1"

# With JSON arguments
afs call swe-synth evaluate --json '{"task_id": 1, "temperature": 0.0}'

# With custom timeout (10 minutes)
afs call swe-synth evaluate --arg task_id=1 --timeout 600
```

## List Available Methods

```bash
# Check what methods are available on a running container
curl http://$(docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_name>):8000/methods
```
