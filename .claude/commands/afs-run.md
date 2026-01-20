# Run Affinetes Environment

Start an affinetes environment container from a Docker image.

## Usage

```
/afs-run <image> [--name <name>] [--env KEY=VALUE...]
```

## Arguments

- `image`: Docker image name (e.g., `swe-synth:v1`)
- `--name`: Container name (default: derived from image)
- `--env`: Environment variables (can be specified multiple times)
- `--mem-limit`: Memory limit (e.g., `512m`, `1g`, `2g`)

## Common Environment Variables

| Variable | Description |
|----------|-------------|
| `CHUTES_API_KEY` | API key for LLM service |
| `R2_ENDPOINT_URL` | R2 storage endpoint (for caching) |
| `R2_BUCKET` | R2 bucket name |
| `R2_ACCESS_KEY_ID` | R2 access key |
| `R2_SECRET_ACCESS_KEY` | R2 secret key |

## Instructions

1. Ensure Docker daemon is running
2. Run the container:
   ```bash
   afs run <image> --name <name> --env KEY=VALUE
   ```
3. After starting, note the available methods shown in the output

## Example

```bash
# Basic run
afs run swe-synth:v1 --name swe-synth

# Run with environment variables
afs run swe-synth:v1 --name swe-synth \
  --env CHUTES_API_KEY=your_key \
  --env R2_BUCKET=your_bucket

# Run with memory limit
afs run swe-synth:v1 --name swe-synth --mem-limit 2g
```

## Stop Container

```bash
docker stop <container_name>
```
