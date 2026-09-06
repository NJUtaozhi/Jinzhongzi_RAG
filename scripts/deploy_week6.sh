#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root on the 4-core/8-GB Ubuntu server.
cd "$(dirname "$0")/.."

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not installed." >&2
  exit 1
fi

if [[ ! -f .env ]]; then
  echo "Missing .env. Copy .env.example to .env and fill in API keys first." >&2
  exit 1
fi

echo "Building the pinned Ubuntu 20.04 + OpenFace 2.2.0 Vision image..."
docker compose build vision

echo "Starting the complete Week 6 stack..."
docker compose up -d

echo "Waiting for Vision health check..."
for _ in $(seq 1 30); do
  health="$(curl -fsS http://localhost:8001/health || true)"
  if echo "$health" | grep -q '"openface_available":true'; then
    echo "$health"
    echo "Vision and OpenFace are ready."
    docker compose ps
    exit 0
  fi
  sleep 5
done

echo "Vision did not become ready. Recent logs:" >&2
docker compose logs --tail=100 vision >&2
exit 1

