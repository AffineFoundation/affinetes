FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app/

# Shards are fetched on demand at runtime into CORPUS_CACHE_DIR.
# No build-time data preload — the image stays small and the corpus
# can be updated without rebuilding.
