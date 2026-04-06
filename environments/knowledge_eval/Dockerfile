FROM python:3.12-slim

# System deps for HF datasets + TLS bundle
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so the data preload step is cached separately.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment source.
COPY . /app/

# HF token is supplied as a build arg so it never persists in the image.
# GPQA and HLE are gated datasets and will fail without a valid token.
ARG HF_TOKEN
RUN HF_TOKEN="${HF_TOKEN}" \
    HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
    python3 /app/preprocess.py /app/data \
    && rm -rf /root/.cache/huggingface

# HTTP server is injected by the affinetes framework.
