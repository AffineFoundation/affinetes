FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install swesmith from git
RUN pip install --no-cache-dir git+https://github.com/SWE-bench/SWE-smith.git@7329a01b56ce1e64873b9d60a6bcb1a9375a0f3d

# Copy environment files
COPY . .

# Set Python path
ENV PYTHONPATH=/app:$PYTHONPATH

# Default command
CMD ["python", "-m", "http.server", "8000"]