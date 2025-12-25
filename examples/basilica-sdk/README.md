# Basilica SDK Example: LLM Deployment for Affinetes

Deploy LLM service and affinetes evaluation environment on Basilica GPU.

[![demo](https://github.com/user-attachments/assets/e3a0e71b-d545-4946-8bbc-aa72179b0be4)](https://asciinema.org/a/eCspXmImKzKeSD4yp2itdESlg)


## Prerequisites

1. Basilica API token (set as `BASILICA_API_TOKEN`)
2. Python dependencies installed

## Setup

```bash
pip install -r requirements.txt
export BASILICA_API_TOKEN="your-token"
```

## Quick Start

### Step 1: Deploy LLM Server

```bash
python basilica_llm_server.py
```

Output:
```
Deployment Ready
============================================================
Deployment ID: abc123...
Base URL: https://abc123.deployments.basilica.ai
OpenAI API: https://abc123.deployments.basilica.ai/v1
```

### Step 2: Deploy Affinetes Environment

```bash
python basilica_env_server.py
```

Output:
```
Deployment Ready
============================================================
Deployment ID: xyz789...
Base URL: https://xyz789.deployments.basilica.ai
```

### Step 3: Run Evaluation

```bash
python basilica_evaluator.py \
    --llm-url "https://abc123.deployments.basilica.ai/v1" \
    --env-url "https://xyz789.deployments.basilica.ai" \
    --model-name "Qwen/Qwen2.5-0.5B-Instruct" \
    --task-id-start 1 \
    --task-id-end 5
```

## Scripts

| Script | Description |
|--------|-------------|
| `basilica_llm_server.py` | Deploy/delete vLLM on Basilica GPU |
| `basilica_env_server.py` | Deploy/delete affinetes environment on Basilica |
| `basilica_evaluator.py` | Run evaluation using Basilica-deployed services |
| `local_evaluator.py` | Run evaluation with local Docker (requires Docker) |
| `cleanup.py` | List and delete Basilica deployments |

## Configuration

### LLM Server Options

```bash
# Custom model
MODEL_NAME="Qwen/Qwen2.5-7B-Instruct" python basilica_llm_server.py

# Custom deployment name
DEPLOYMENT_NAME="my-llm" python basilica_llm_server.py
```

### Environment Server Options

```bash
# Math environment (default)
ENV_IMAGE="docker.io/affinefoundation/mth:pi" python basilica_env_server.py

# Logic environment
ENV_IMAGE="docker.io/affinefoundation/lgc:pi" python basilica_env_server.py

# General affine environment
ENV_IMAGE="docker.io/affinefoundation/affine-env:v4" python basilica_env_server.py
```

### Evaluator Options

| Option | Description | Default |
|--------|-------------|---------|
| `--llm-url` | Basilica LLM endpoint URL | Required |
| `--env-url` | Basilica environment URL | Required |
| `--model-name` | Model name for evaluation | Required |
| `--task-id-start` | Start of task ID range | 1 |
| `--task-id-end` | End of task ID range | 5 |
| `--task-ids` | Specific task IDs (comma-separated) | - |
| `--timeout` | Timeout per task (seconds) | 1800 |
| `--concurrent` | Concurrent evaluations | 1 |

## Architecture

Everything runs on Basilica cloud:

```
+-------------------+          +-------------------+
|   Basilica GPU    |          |   Basilica CPU    |
|                   |          |                   |
|  +-------------+  |   HTTP   |  +-------------+  |
|  |   vLLM      |<-|----------|->|  affinetes  |  |
|  | (OpenAI API)|  |          |  | environment |  |
|  +-------------+  |          |  +-------------+  |
+-------------------+          +-------------------+
         ^                              ^
         |          HTTP                |
         +------------------------------+
                     |
              +------+------+
              | Evaluator   |
              | (Your PC)   |
              +-------------+
```

## Example Output

```
============================================================
BASILICA SDK - EVALUATION
============================================================
LLM URL: https://abc123.deployments.basilica.ai/v1
Model: Qwen/Qwen2.5-0.5B-Instruct
Environment URL: https://xyz789.deployments.basilica.ai
Task IDs: [1, 2, 3, 4, 5]
Total Tasks: 5
============================================================

Connecting to affinetes environment...
Environment connected successfully

[Task 1] Starting evaluation...
[Task 1] Completed in 45.23s - Score: 0.85
...

============================================================
EVALUATION SUMMARY
============================================================
Total Time: 195.45s
Total Tasks: 5
Successful: 5
Failed: 0
Average Score: 0.8500
============================================================
```

## Available Environment Images

| Image | Description |
|-------|-------------|
| `docker.io/affinefoundation/mth:pi` | Math evaluation |
| `docker.io/affinefoundation/lgc:pi` | Logic evaluation |
| `docker.io/affinefoundation/affine-env:v4` | General affine |

## Cleanup

Deployments auto-delete after TTL (default 1 hour). To delete manually:

### Using the Cleanup Utility

```bash
# List all active deployments
python cleanup.py

# Delete a specific deployment
python cleanup.py --delete <deployment-name>

# Delete all deployments (with confirmation)
python cleanup.py --delete-all
```

### Using Deploy Scripts

Each deploy script supports `--delete` flag:

```bash
# Delete LLM deployment
python basilica_llm_server.py --delete --name <deployment-name>

# Delete environment deployment
python basilica_env_server.py --delete --name <deployment-name>
```

### Using SDK Directly

```python
from basilica import BasilicaClient

client = BasilicaClient()

# List all deployments
response = client.list_deployments()
for dep in response.deployments:
    print(f"{dep.instance_name}: {dep.state}")

# Delete by name
client.delete_deployment("your-deployment-name")
```

### Using CLI

```bash
# List deployments
basilica deploy ls

# Delete deployment
basilica deploy delete <deployment-name>
```
