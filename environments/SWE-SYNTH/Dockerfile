# SWE-SYNTH Environment Dockerfile
# Self-contained with all dependencies

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create cache directory
RUN mkdir -p /app/swe-synth-cache

# Copy source code, config, and ridges data
COPY env.py cache.py __init__.py config.yaml utils.py ./
COPY fixer/ ./fixer/
COPY ridges/ ./ridges/

# Set default paths for ridges data
ENV RIDGE_PROJECT_PATH=/app/ridges
ENV PYTHONUNBUFFERED=1
