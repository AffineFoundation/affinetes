# syntax=docker/dockerfile:1.4
# Two-stage wrapper Dockerfile for function-based environments (injects HTTP server + shared helpers)

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

USER root

RUN pip install --no-cache-dir fastapi uvicorn[standard] httpx pydantic

RUN mkdir -p /app/_affinetes
COPY http_server.py /app/_affinetes/server.py
COPY request_logger.py /app/request_logger.py
RUN echo "" > /app/_affinetes/__init__.py

# Install minimal tools needed for pip installing from GitHub SSH, then install affinetes (no deps).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && pip install git+https://github.com/AffineFoundation/affinetes.git@main

RUN chmod -R 777 /app/_affinetes

WORKDIR /app
CMD sh -c "python -m uvicorn _affinetes.server:app --host 0.0.0.0 --port ${AFFINETES_PORT:-8000} --workers ${UVICORN_WORKERS:-1}"