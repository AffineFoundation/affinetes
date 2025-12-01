FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy environment files
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# HTTP server will be injected by affinetes framework
