# SWE-FRONTIER Environment Dockerfile

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    docker.io \
    golang-1.22-go \
    && rm -rf /var/lib/apt/lists/*

# afentctl static binary — agent loop core from affine-agents (Go).
# Build from source so the binary always matches the repo we develop
# against. Static linkage (CGO_ENABLED=0) makes the binary portable
# across the various distros our task release images use.
ARG AFENT_REPO=https://github.com/affine-io/affine-agents.git
ARG AFENT_REF=main
RUN export PATH="/usr/lib/go-1.22/bin:$PATH" \
 && git clone --depth=1 --branch "$AFENT_REF" "$AFENT_REPO" /tmp/affine-agents \
 && cd /tmp/affine-agents/afent \
 && CGO_ENABLED=0 go build -ldflags="-s -w" -o /usr/local/bin/afentctl-static ./cmd/afentctl \
 && chmod +x /usr/local/bin/afentctl-static \
 && rm -rf /tmp/affine-agents /root/.cache/go-build

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY env.py __init__.py ./
COPY agents/ ./agents/

ENV PYTHONUNBUFFERED=1
