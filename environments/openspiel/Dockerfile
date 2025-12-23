FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy environment files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# HTTP server will be injected by affinetes framework