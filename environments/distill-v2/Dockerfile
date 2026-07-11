FROM python:3.12-slim

# Many worker threads doing large transient allocations (echo-logprob JSON
# bodies) make glibc grow one malloc arena per thread and never return the
# RSS. Two arenas keep the container's memory close to live usage.
ENV MALLOC_ARENA_MAX=2

RUN apt-get update && apt-get install -y \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first so they cache across env.py changes.
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# HTTP server is injected by the affinetes framework on container start.
