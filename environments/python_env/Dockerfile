FROM python:3.12-slim

WORKDIR /app

# Install environment dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . /app/

# Note: HTTP server will be auto-injected by affinetes
