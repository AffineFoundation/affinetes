FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Pre-download the dataset to speed up runtime
RUN python -c "from datasets import load_dataset; load_dataset('PrimeIntellect/INTELLECT-3-RL', 'code', split='train')"

# Set environment variable for logging
ENV I3_CODE_LOG_LEVEL=INFO

# Default command runs the actor
CMD ["python", "-m", "env"]