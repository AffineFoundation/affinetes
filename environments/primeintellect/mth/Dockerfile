FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Pre-download HuggingFace dataset during build to avoid runtime downloads
RUN pip install datasets && python3 -c "from datasets import load_dataset; load_dataset('PrimeIntellect/INTELLECT-3-RL', 'math', split='train')"

# Set working directory
WORKDIR /app

# Copy environment files
COPY . /app/

# Install Python dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# HTTP server will be injected by affinetes framework