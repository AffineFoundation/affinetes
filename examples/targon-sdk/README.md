## Prerequisites

### Install Targon SDK

```bash
pip install git+https://github.com/manifold-inc/targon-sdk.git
```

### Setup Targon Credentials

```bash
pip install keyrings.alt

targon setup
```

## Usage

### Deploy the validator function:
```bash
targon deploy examples/targon-sdk/validator.py
```

### Run evaluation:
```bash
targon run examples/targon-sdk/validator.py \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --task-ids "1,2,3,4,5,6,7,8,9,10" \
    --image "docker.io/affinefoundation/mth:pi" \
    --timeout 1800
```

## Output

The validator returns a JSON object with:
- `model_name`: Model name used for evaluation
- `total_tasks`: Total number of tasks evaluated
- `successful_tasks`: Number of successfully completed tasks
- `failed_tasks`: Number of failed tasks
- `average_score`: Average score across all tasks
- `total_score`: Sum of all task scores
- `results`: Detailed results for each task

Results are also saved to `rollouts_<timestamp>.json`.