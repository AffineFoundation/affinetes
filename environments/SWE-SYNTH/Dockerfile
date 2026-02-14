# SWE-SYNTH Environment Dockerfile
# Self-contained with all dependencies

FROM python:3.11-slim

# Install system dependencies + Node.js (for codex CLI)
RUN apt-get update && apt-get install -y \
    git \
    curl \
    docker.io \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Install OpenAI Codex CLI (pin to 0.94.0 for wire_api="chat" support)
# Extract the static musl binary so we can docker-cp it into SWE-bench containers
RUN npm install -g @openai/codex@0.94.0 \
    && cp "$(find /usr/local/lib/node_modules/@openai/codex -name codex -path '*/x86_64-unknown-linux-musl/codex/codex')" \
       /usr/local/bin/codex-static \
    && chmod +x /usr/local/bin/codex-static

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create cache directory
RUN mkdir -p /app/swe-synth-cache

# Copy source code, config, and agent scaffolds
COPY env.py cache.py __init__.py config.yaml utils.py ./
COPY fixer/ ./fixer/
COPY agent/ ./agent/

# Set default paths for ridges data
ENV RIDGE_PROJECT_PATH=/app/agent/ridges
ENV PYTHONUNBUFFERED=1
