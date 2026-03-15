# SWE-INFINITE Environment Dockerfile

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
# Extract the static musl binary so we can docker-cp it into task containers
RUN npm install -g @openai/codex@0.94.0 \
    && cp "$(find /usr/local/lib/node_modules/@openai/codex -name codex -path '*/x86_64-unknown-linux-musl/codex/codex')" \
       /usr/local/bin/codex-static \
    && chmod +x /usr/local/bin/codex-static

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code and config
COPY env.py cache.py utils.py __init__.py ./
COPY agents/ ./agents/

ENV PYTHONUNBUFFERED=1
