#!/bin/bash
# Build the JS image for test_boundary_l2_js_e2e.py: a node repo at /app with a
# buggy src/calc.js (add returns a-b) under git, plus jest + git.
set -e
D="$(mktemp -d)"
mkdir -p "$D/src"
cat > "$D/src/calc.js" <<'JS'
function add(a, b) {
  return a - b;
}
module.exports = { add };
JS
cat > "$D/package.json" <<'JSON'
{ "name": "calc", "version": "1.0.0", "scripts": { "test": "jest" } }
JSON
cat > "$D/Dockerfile" <<'DOCKER'
FROM node:20-slim
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g jest@29 --silent
WORKDIR /app
COPY package.json /app/package.json
COPY src/calc.js /app/src/calc.js
RUN git init -q && git config user.email a@b.c && git config user.name t \
    && git add -A && git commit -qm base
DOCKER
docker build -t swe-inf-boundary-test-js:latest "$D"
rm -rf "$D"
echo "built swe-inf-boundary-test-js:latest"
