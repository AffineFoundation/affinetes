#!/bin/bash
# Build the Ruby image for test_boundary_l2_ruby_e2e.py: a ruby repo at /app with
# a buggy lib/calc.rb (add returns a-b) under git, plus rspec + git.
set -e
D="$(mktemp -d)"
mkdir -p "$D/lib"
cat > "$D/lib/calc.rb" <<'RB'
def add(a, b)
  a - b
end
RB
cat > "$D/Dockerfile" <<'DOCKER'
FROM ruby:3.2-slim
ENV PATH="/usr/local/bundle/bin:${PATH}"
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && gem install rspec -N --silent
WORKDIR /app
COPY lib/calc.rb /app/lib/calc.rb
RUN git init -q && git config user.email a@b.c && git config user.name t \
    && git add -A && git commit -qm base
DOCKER
docker build -t swe-inf-boundary-test-ruby:latest "$D"
rm -rf "$D"
echo "built swe-inf-boundary-test-ruby:latest"
