# SWE-SYNTH Environment Dockerfile
# Reuses SWE-bench infrastructure

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Set working directory early
WORKDIR /app

# Install Python dependencies (includes mini-swe-agent)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Clone SWE-bench Pro repository to get run_scripts and dockerfiles
RUN git clone https://github.com/scaleapi/SWE-bench_Pro-os.git /swe-bench-pro && \
    ln -s /swe-bench-pro/run_scripts /app/run_scripts && \
    ln -s /swe-bench-pro/dockerfiles /app/dockerfiles

# Create cache directory
RUN mkdir -p /app/swe-synth-cache

# Copy environment files
COPY . /app/

# Disable Python stdout buffering for real-time logs
ENV PYTHONUNBUFFERED=1
