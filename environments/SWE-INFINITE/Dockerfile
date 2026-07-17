# SWE-INFINITE Environment Dockerfile

ARG AFFENT_REPO=https://github.com/AffineFoundation/affent.git
ARG AFFENT_REF=8f4f8f6e765ec8faabd2a27a457811f0f2218b74

# Build affent (Go agent loop) as a static binary that we can docker-cp into
# task containers — same delivery model as codex-static.
FROM golang:1.24.13-alpine AS affent-builder
ARG AFFENT_REPO
ARG AFFENT_REF
RUN apk add --no-cache ca-certificates git
WORKDIR /src
RUN git clone --filter=blob:none --no-checkout "$AFFENT_REPO" affent \
    && cd affent \
    && git checkout --detach "$AFFENT_REF" \
    && test "$(git rev-parse HEAD)" = "$AFFENT_REF"
WORKDIR /src/affent
RUN go mod download
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" \
    -o /out/affent-static ./cmd/affentctl \
    && /out/affent-static help >/dev/null 2>&1 \
    && go version -m /out/affent-static > /out/affent-build-info \
    && grep -Fq "vcs.revision=$AFFENT_REF" /out/affent-build-info \
    && sha256sum /out/affent-static | cut -d' ' -f1 > /out/affent-sha256 \
    && printf '%s\n' "$AFFENT_REF" > /out/affent-ref

FROM python:3.11-slim
ARG AFFENT_REF
LABEL io.affine.affent.revision="$AFFENT_REF"

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
COPY --from=affent-builder /out/affent-build-info /usr/local/share/affent-build-info
COPY --from=affent-builder /out/affent-ref /usr/local/share/affent-ref
COPY --from=affent-builder /out/affent-sha256 /usr/local/share/affent-sha256
RUN chmod +x /usr/local/bin/affent-static

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code and config
COPY env.py cache.py utils.py canary.py __init__.py ./
COPY agents/ ./agents/

ENV PYTHONUNBUFFERED=1 \
    AFFENT_REF="$AFFENT_REF"
