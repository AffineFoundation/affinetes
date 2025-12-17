FROM python:3.11-slim

# Install Docker CLI for DOOD (Docker-out-of-Docker) and git
RUN apt-get update && apt-get install -y \
    docker.io \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone SWE-bench Pro repository to get run_scripts, dockerfiles, and helper_code
RUN git clone https://github.com/scaleapi/SWE-bench_Pro-os.git /swe-bench-pro && \
    ln -s /swe-bench-pro/run_scripts /app/run_scripts && \
    ln -s /swe-bench-pro/dockerfiles /app/dockerfiles && \
    ln -s /swe-bench-pro/helper_code /app/helper_code

# Copy environment code
COPY env.py .
COPY swebench_config.yaml .

# Expose port for affinetes HTTP server
EXPOSE 8000
ENV UVICORN_WORKERS=1

# Note: Container should be run with -v /var/run/docker.sock:/var/run/docker.sock
# to enable Docker-out-of-Docker functionality

CMD ["python", "-m", "affinetes.server"]