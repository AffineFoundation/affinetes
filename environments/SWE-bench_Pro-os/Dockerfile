FROM python:3.10-slim

# Install Docker CLI for DOOD (Docker-out-of-Docker)
RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code and agent configuration
COPY env.py .
COPY swebench_config.yaml .

# Expose port for affinetes HTTP server
EXPOSE 8000

# Note: Container should be run with -v /var/run/docker.sock:/var/run/docker.sock
# to enable Docker-out-of-Docker functionality

CMD ["python", "-m", "affinetes.server"]