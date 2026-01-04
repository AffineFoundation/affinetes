FROM python:3.11-slim

ARG TARGETARCH
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    tar \
    && rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    arch="${TARGETARCH:-$(uname -m)}"; \
    if [ "$arch" = "amd64" ] || [ "$arch" = "x86_64" ]; then ARCH="x86_64"; \
    elif [ "$arch" = "arm64" ] || [ "$arch" = "aarch64" ]; then ARCH="aarch64"; \
    else echo "Unsupported TARGETARCH=$arch"; exit 1; fi; \
    curl -L -o /tmp/codex.tgz "https://github.com/openai/codex/releases/latest/download/codex-${ARCH}-unknown-linux-musl.tar.gz"; \
    tar -xzf /tmp/codex.tgz -C /usr/local/bin; \
    mv "/usr/local/bin/codex-${ARCH}-unknown-linux-musl" /usr/local/bin/codex; \
    chmod +x /usr/local/bin/codex; \
    rm /tmp/codex.tgz

WORKDIR /app
COPY . /app/

RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# HTTP server will be injected by affinetes framework
