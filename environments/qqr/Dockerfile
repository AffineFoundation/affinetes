FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install qqr package from GitHub (includes mcp tools)
RUN pip install --no-cache-dir git+https://github.com/Alibaba-NLP/qqr.git

# Patch qqr's __init__.py to avoid slime import (slime is internal Alibaba package)
# We only need the MCP tools, not the full qqr training framework
RUN echo '# Patched to avoid slime dependency\n__version__ = "0.1.1"' > /usr/local/lib/python3.11/site-packages/qqr/__init__.py

# Copy environment requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment files
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
