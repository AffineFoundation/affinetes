# Build Affinetes Environment

Build a Docker image from an affinetes environment directory.

## Usage

```
/afs-build <env_dir> [--tag <tag>] [--no-cache]
```

## Arguments

- `env_dir`: Environment directory path (e.g., `environments/SWE-SYNTH`)
- `--tag`: Image tag (default: `<env_name>:latest`)
- `--no-cache`: Build without Docker cache

## Instructions

1. First, verify the environment directory exists and contains required files:
   - `env.py` (required)
   - `Dockerfile` (required)
   - `requirements.txt` (optional)

2. Run the build command:
   ```bash
   afs build <env_dir> --tag <tag>
   ```

3. If build fails, check:
   - Dockerfile syntax errors
   - Missing dependencies in requirements.txt
   - Python syntax errors in env.py

## Example

```bash
# Build SWE-SYNTH environment
afs build environments/SWE-SYNTH --tag swe-synth:v1

# Build with no cache
afs build environments/SWE-SYNTH --tag swe-synth:v1 --no-cache
```
