# SWE-SYNTH Breaker Service
#
# Build (from SWE-SYNTH directory):
#   docker build -f breaker/Dockerfile -t swe-synth-breaker .
#
# Run breaker:
#   docker run -v /var/run/docker.sock:/var/run/docker.sock \
#              -e CHUTES_API_KEY=xxx \
#              -e R2_ENDPOINT_URL=xxx \
#              -e R2_BUCKET=xxx \
#              -e R2_ACCESS_KEY_ID=xxx \
#              -e R2_SECRET_ACCESS_KEY=xxx \
#              swe-synth-breaker
#
# Run release (copy tasks from private to public R2):
#   docker run -e R2_ENDPOINT_URL=xxx \
#              -e R2_BUCKET=xxx \
#              -e R2_PUBLIC_BUCKET=xxx \
#              -e R2_ACCESS_KEY_ID=xxx \
#              -e R2_SECRET_ACCESS_KEY=xxx \
#              --entrypoint python swe-synth-breaker \
#              -m breaker.release_tasks --up-to 99

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY breaker/ ./breaker/
COPY cache.py ./
COPY ridges/ ./ridges/

ENTRYPOINT ["python", "-m", "breaker.service", \
            "--run-scripts-dir", "/app/ridges/run_scripts"]

# Default: unlimited, starting from task 0
CMD ["--start-from", "0"]
