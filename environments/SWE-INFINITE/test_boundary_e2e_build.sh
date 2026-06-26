#!/bin/bash
# Build the minimal image used by test_boundary_e2e.py: a python repo at /app
# with a buggy calc.add (returns a-b) under git, plus pytest + git.
set -e
D="$(mktemp -d)"
cat > "$D/calc.py" <<'PY'
def add(a, b):
    return a - b
PY
cat > "$D/Dockerfile" <<'DOCKER'
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir pytest
WORKDIR /app
COPY calc.py /app/calc.py
RUN git init -q && git config user.email a@b.c && git config user.name t \
    && git add -A && git commit -qm base
DOCKER
docker build -t swe-inf-boundary-test:latest "$D"
rm -rf "$D"
echo "built swe-inf-boundary-test:latest"
