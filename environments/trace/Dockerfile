FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install datasets package
RUN pip install --no-cache-dir datasets

# Pre-download satpalsr/rl-python dataset
RUN python3 -c "from datasets import load_dataset; \
    print('Downloading satpalsr/rl-python dataset for cache...'); \
    load_dataset('satpalsr/rl-python', split='train'); \
    print('Dataset cached successfully!')"

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Set environment variable for logging
ENV TRACE_LOG_LEVEL=INFO
ENV UVICORN_WORKERS=1

# Default command runs the actor
CMD ["python", "-m", "env"]

