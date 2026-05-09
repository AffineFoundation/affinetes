# SWE-FRONTIER Environment Dockerfile

# Build affent (Go agent loop) as a static binary — same submodule + same
# build recipe as SWE-INFINITE, so both environments speak to one upstream
# (AffineFoundation/affent) with one binary name.
FROM golang:1.22-alpine AS affent-builder
WORKDIR /src/affent
COPY affent/go.mod affent/go.sum ./
RUN go mod download
COPY affent/ ./
RUN CGO_ENABLED=0 go build -ldflags="-s -w" -o /out/affent-static ./cmd/affentctl

FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    ca-certificates \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# affent static binary — docker-cp'd into task containers by AffentAgent.
COPY --from=affent-builder /out/affent-static /usr/local/bin/affent-static
RUN chmod +x /usr/local/bin/affent-static

WORKDIR /app

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY env.py __init__.py ./
COPY agents/ ./agents/

ENV PYTHONUNBUFFERED=1
