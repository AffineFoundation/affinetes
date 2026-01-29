# Validate Affinetes Environment

Validate environment seed consistency and generate test rollouts.

## Usage

```
/afs-validate <env_dir> [--num-tests N] [--task-id-start N] [--output DIR]
```

## Arguments

- `env_dir`: Environment directory path
- `--num-tests`: Number of tests to run (default: 100)
- `--task-id-start`: Starting task_id (default: 1)
- `--task-id-end`: Ending task_id (default: start + num_tests - 1)
- `--output`: Output directory for results (default: `rollouts/`)
- `--temperature`: Temperature for LLM (default: 0.7)
- `--timeout`: Timeout per evaluation in seconds (default: 60)

## Environment Variables Required

```bash
export CHUTES_API_TOKEN=your_token
export MINER_SLUG=your_slug
```

## What It Validates

1. **Seed Consistency**: Same seed generates identical questions
2. **Seed Diversity**: Different seeds generate different questions
3. Each test runs twice to verify deterministic behavior

## Instructions

1. Set required environment variables
2. Run validation:
   ```bash
   afs validate <env_dir> --num-tests 50
   ```
3. Check output in `rollouts/` directory

## Examples

```bash
# Basic validation (seed consistency only)
afs validate environments/SWE-SYNTH

# Run 50 tests with model
export CHUTES_API_TOKEN=your_token
export MINER_SLUG=your_slug
afs validate environments/SWE-SYNTH --num-tests 50

# Test specific task_id range
afs validate environments/SWE-SYNTH --task-id-start 100 --task-id-end 199

# Custom output directory
afs validate environments/SWE-SYNTH --output my_results --num-tests 20
```

## Output

```
Running 100 tests (each test runs twice to validate seed consistency)
--------------------------------------------------------------------------------
Progress: 10/100 tests completed
...

✓ Completed 100 tests
Output directory: rollouts/
Success rate: 45/100 (45.0%)
Seed consistency: 100/100 (100.0%)
Seed diversity: 100/100 unique questions (100.0%)
```

## Generated Files

- `test_task00001.json` ~ `test_taskNNNNN.json`: Individual test results
- `summary.json`: Aggregated statistics
