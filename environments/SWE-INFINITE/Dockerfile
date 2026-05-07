# SWE-INFINITE Environment Dockerfile

# Build affent (Go agent loop) as a static binary that we can docker-cp into
# task containers — same delivery model as codex-static.
FROM golang:1.22-alpine AS affent-builder
WORKDIR /src/affent
COPY affent/go.mod affent/go.sum ./
RUN go mod download
COPY affent/ ./
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /out/affent-static ./cmd/affentctl

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

# affent static binary — docker-cp'd into task containers by AffentAgent
COPY --from=affent-builder /out/affent-static /usr/local/bin/affent-static
RUN chmod +x /usr/local/bin/affent-static

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code and config
COPY env.py cache.py utils.py canary.py __init__.py ./
COPY agents/ ./agents/

ENV PYTHONUNBUFFERED=1
