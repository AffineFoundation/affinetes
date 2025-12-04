FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install datasets package (shared layer across all primeintellect environments)
RUN pip install --no-cache-dir datasets

# Pre-download ALL INTELLECT-3-RL dataset subsets (shared layer across all environments)
# This ensures all four environments (mth/sci/cde/lgc) share the same cached dataset layer
RUN python3 -c "from datasets import load_dataset; \
    print('Downloading all INTELLECT-3-RL subsets for cache sharing...'); \
    load_dataset('PrimeIntellect/INTELLECT-3-RL', 'math', split='train'); \
    load_dataset('PrimeIntellect/INTELLECT-3-RL', 'science', split='train'); \
    load_dataset('PrimeIntellect/INTELLECT-3-RL', 'code', split='train'); \
    load_dataset('PrimeIntellect/INTELLECT-3-RL', 'logic', split='train'); \
    print('All subsets cached successfully!')"

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Set environment variable for logging
ENV I3_CODE_LOG_LEVEL=INFO

# Default command runs the actor
CMD ["python", "-m", "env"]