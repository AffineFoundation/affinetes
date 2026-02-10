# Test Affinetes Environment (Full Workflow)

Complete workflow to build, run, and test an affinetes environment.

## Usage

```
/afs-test <env_dir> [--tag <tag>] [--task-id <id>] [--model <model>]
```

## Arguments

- `env_dir`: Environment directory path (e.g., `environments/SWE-SYNTH`)
- `--tag`: Image tag (default: `<env_name>:latest`)
- `--task-id`: Task ID to test (default: 1)
- `--model`: Model to use for evaluation
- `--base-url`: API base URL

## Full Workflow Steps

### Step 1: Build the image

```bash
afs build <env_dir> --tag <tag>
```

### Step 2: Start the container

```bash
afs run <tag> --name <env_name> --env CHUTES_API_KEY=$CHUTES_API_KEY
```

### Step 3: Call evaluate method

```bash
afs call <env_name> evaluate \
  --arg task_id=<task_id> \
  --arg model="<model>" \
  --arg base_url="<base_url>"
```

### Step 4: Stop container

```bash
docker stop <env_name>
```

## Example: Test SWE-SYNTH

```bash
# Set API key
export CHUTES_API_KEY=your_api_key

# 1. Build
afs build environments/SWE-SYNTH --tag swe-synth:v1

# 2. Run
afs run swe-synth:v1 --name swe-synth \
  --env CHUTES_API_KEY=$CHUTES_API_KEY

# 3. Test single task
afs call swe-synth evaluate \
  --arg task_id=1 \
  --arg model="deepseek-ai/DeepSeek-V3" \
  --arg base_url="https://llm.chutes.ai/v1" \
  --arg temperature=0.0

# 4. Cleanup
docker stop swe-synth
```

## Python API Alternative

```python
import affinetes as af_env
import asyncio

async def test_env():
    env = af_env.load_env(
        image="swe-synth:v1",
        env_vars={"CHUTES_API_KEY": "your-key"}
    )

    result = await env.evaluate(
        task_id=1,
        model="deepseek-ai/DeepSeek-V3",
        base_url="https://llm.chutes.ai/v1",
    )

    print(f"Score: {result['score']}")
    print(f"Success: {result['success']}")

    await env.cleanup()

asyncio.run(test_env())
```

## Quick Test (Build + Run in one command)

```bash
# Build from directory and start immediately
afs run --dir environments/SWE-SYNTH --tag swe-synth:latest \
  --env CHUTES_API_KEY=$CHUTES_API_KEY
```

## Troubleshooting

### Container won't start
```bash
docker logs <container_name>
```

### Method not found
```bash
# List available methods
afs call <container_name> list_methods
```

### Check container status
```bash
docker ps -a | grep <container_name>
```
