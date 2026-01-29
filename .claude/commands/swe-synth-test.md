# SWE-SYNTH Test Workflow

Test SWE-SYNTH environment with model evaluation.

## Quick Start

```bash
# 1. Build image
afs build environments/SWE-SYNTH --tag swe-synth:v1

# 2. Run container (mount docker.sock for mini-swe-agent)
docker run -d --name swe-synth -p 8765:8000 \
  -v /tmp/swe-synth-cache:/tmp/swe-synth-cache \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -e CHUTES_API_KEY="your_chutes_key" \
  swe-synth:v1

# 3. Test single task
curl -X POST http://localhost:8765/call \
  -H "Content-Type: application/json" \
  -d '{
    "method": "evaluate",
    "kwargs": {
      "task_id": 3,
      "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
      "base_url": "https://llm.chutes.ai/v1",
      "api_key": "your_chutes_key",
      "max_iterations": 100
    }
  }'
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `task_id` | Task ID (encodes SWE instance + bug types + seed) | Required |
| `model` | Model name for fixer | Required |
| `base_url` | API endpoint for fixer | `https://llm.chutes.ai/v1` |
| `api_key` | API key for fixer model | `CHUTES_API_KEY` env |
| `max_iterations` | Max agent iterations | 30 |
| `skip_cache` | Force regenerate bug | false |

## Architecture

```
SWE-SYNTH Evaluation Flow:

1. task_id decode -> SWE instance + bug_types + seed

2. Cache check (L1: local, L2: R2)
   - Hit: Load cached bug
   - Miss: Generate with Breaker

3. Breaker (Qwen 480B) generates bug:
   - Injects fault into gold_patch area
   - Creates problem_statement
   - Verifies bug causes test failures

4. Fixer (user model) repairs bug:
   - Uses mini-swe-agent in container
   - Interacts with codebase via bash
   - Generates fix_patch

5. Verify fix:
   - Apply fix_patch
   - Run target tests
   - Score = passed_target_tests / total_target
```

## Cache Locations

- **L1 (Local)**: `/tmp/swe-synth-cache/task_<id>.json`
- **L2 (R2)**: `https://pub-4b43a94ed07d4ac38fae3f4cb5070d6c.r2.dev/task_<id>.json`

## Testing Multiple Tasks

```bash
# Batch test
for task_id in 3 444 478 501 737; do
  echo "Testing Task $task_id..."
  curl -s -X POST http://localhost:8765/call \
    -H "Content-Type: application/json" \
    -d "{
      \"method\": \"evaluate\",
      \"kwargs\": {
        \"task_id\": $task_id,
        \"model\": \"Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE\",
        \"base_url\": \"https://llm.chutes.ai/v1\",
        \"api_key\": \"$CHUTES_API_KEY\",
        \"max_iterations\": 100
      }
    }" | python3 -c "
import sys,json
r=json.load(sys.stdin)
res=r.get('result',{})
print(f\"  Score: {res.get('score',0)}, Time: {res.get('time_taken',0):.1f}s\")
"
done
```

## Using Different Models

### Qwen (Chutes)
```json
{
  "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8-TEE",
  "base_url": "https://llm.chutes.ai/v1",
  "api_key": "cpk_..."
}
```

### Claude (Anthropic)
```json
{
  "model": "claude-opus-4-5-20251101",
  "api_key": "sk-ant-api03-..."
}
```
Note: For Anthropic models, set `ANTHROPIC_API_KEY` env var or pass via `api_key`.

### DeepSeek
```json
{
  "model": "deepseek-ai/DeepSeek-V3",
  "base_url": "https://llm.chutes.ai/v1",
  "api_key": "cpk_..."
}
```

## Inspecting Results

```bash
# Check cached bug
cat /tmp/swe-synth-cache/task_3.json | python3 -c "
import sys,json
d=json.load(sys.stdin)
print('Bug types:', d['bug']['bug_types'])
print('Problem:', d['bug']['problem_statement'])
print('Patch:', d['bug']['patch'][:500])
"

# Check container logs
docker logs swe-synth 2>&1 | tail -50
```

## Troubleshooting

### Docker-in-Docker Error (exit 125)
Mount docker socket:
```bash
-v /var/run/docker.sock:/var/run/docker.sock
```

### API Key Issues
- Chutes: `CHUTES_API_KEY` env or `api_key` param
- Anthropic: `ANTHROPIC_API_KEY` env (auto-detected for `claude-*` models)

### JSON Parse Error in Breaker
Check breaker.py JSON parsing logic. The code uses balanced brace matching for nested JSON.

### No Patch Generated
Check docker logs - usually indicates model API authentication failure.
