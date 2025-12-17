FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download DeepDive dataset for caching
RUN python3 -c "from datasets import load_dataset; \
    print('Downloading DeepDive dataset...'); \
    load_dataset('zai-org/DeepDive', split='qa_rl'); \
    print('Dataset cached successfully!')"

# Copy environment code
COPY . .

# Set environment variable for logging
ENV UVICORN_WORKERS=1

# Default command runs the actor
CMD ["python", "-m", "env"]