# syntax=docker/dockerfile:1.4

# ---------------------------------------------------------------------------
# Stage 1: build affentctl (Go agent loop binary that drives evaluations)
# ---------------------------------------------------------------------------
FROM golang:1.23-bookworm AS affent-build

# Pin commit for reproducibility. Update when affent ships a fix we need.
ARG AFFENT_REPO=https://github.com/AffineFoundation/affent.git
ARG AFFENT_REF=dbf9df9

WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --filter=blob:none "$AFFENT_REPO" affent \
    && cd affent && git checkout "$AFFENT_REF"

WORKDIR /src/affent
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/affentctl ./cmd/affentctl

# ---------------------------------------------------------------------------
# Stage 2: qqr runtime (Python)
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Affent CLI used by env.py to drive the real agent loop
COPY --from=affent-build /out/affentctl /usr/local/bin/affentctl
RUN chmod +x /usr/local/bin/affentctl

WORKDIR /app

# Install qqr package from GitHub (includes mcp tools)
RUN pip install --no-cache-dir git+https://github.com/Alibaba-NLP/qqr.git

# Patch qqr's __init__.py to avoid slime import (slime is internal Alibaba package)
# We only need the MCP tools, not the full qqr training framework
RUN echo '# Patched to avoid slime dependency\n__version__ = "0.1.1"' > /usr/local/lib/python3.11/site-packages/qqr/__init__.py


# Copy environment requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV AFFENTCTL_BIN=/usr/local/bin/affentctl
