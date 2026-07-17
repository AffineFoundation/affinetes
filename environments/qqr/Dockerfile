# syntax=docker/dockerfile:1.4

ARG AFFENT_REPO=https://github.com/AffineFoundation/affent.git
ARG AFFENT_REF=8f4f8f6e765ec8faabd2a27a457811f0f2218b74

# ---------------------------------------------------------------------------
# Stage 1: build affentctl (Go agent loop binary that drives evaluations)
# ---------------------------------------------------------------------------
FROM golang:1.24.13-bookworm AS affent-build

# Pin commit for reproducibility. Update when affent ships a fix we need.
ARG AFFENT_REPO
ARG AFFENT_REF

WORKDIR /src
RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && git clone --filter=blob:none --no-checkout "$AFFENT_REPO" affent \
    && cd affent \
    && git checkout --detach "$AFFENT_REF" \
    && test "$(git rev-parse HEAD)" = "$AFFENT_REF"

WORKDIR /src/affent
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" \
    -o /out/affentctl ./cmd/affentctl \
    && /out/affentctl help >/dev/null 2>&1 \
    && go version -m /out/affentctl > /out/affent-build-info \
    && grep -Fq "vcs.revision=$AFFENT_REF" /out/affent-build-info \
    && sha256sum /out/affentctl | cut -d' ' -f1 > /out/affent-sha256 \
    && printf '%s\n' "$AFFENT_REF" > /out/affent-ref

# ---------------------------------------------------------------------------
# Stage 2: qqr runtime (Python)
# ---------------------------------------------------------------------------
FROM python:3.11-slim
ARG AFFENT_REF
LABEL io.affine.affent.revision="$AFFENT_REF"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Affent CLI used by env.py to drive the real agent loop
COPY --from=affent-build /out/affentctl /usr/local/bin/affentctl
COPY --from=affent-build /out/affent-build-info /usr/local/share/affent-build-info
COPY --from=affent-build /out/affent-ref /usr/local/share/affent-ref
COPY --from=affent-build /out/affent-sha256 /usr/local/share/affent-sha256
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
ENV AFFENT_REF="$AFFENT_REF"
