#!/bin/bash
# Build the Go image for test_boundary_go_e2e.py: a module at /app with a buggy
# calc.Add (returns a-b) under git, plus the go toolchain.
set -e
D="$(mktemp -d)"
mkdir -p "$D/calc"
cat > "$D/go.mod" <<'GOMOD'
module example.com/calc

go 1.22
GOMOD
cat > "$D/calc/calc.go" <<'GO'
package calc

func Add(a, b int) int {
	return a - b
}
GO
cat > "$D/Dockerfile" <<'DOCKER'
FROM golang:1.22-bookworm
WORKDIR /app
COPY go.mod /app/go.mod
COPY calc /app/calc
RUN git init -q && git config user.email a@b.c && git config user.name t \
    && git add -A && git commit -qm base
DOCKER
docker build -t swe-inf-boundary-go:latest "$D"
rm -rf "$D"
echo "built swe-inf-boundary-go:latest"
