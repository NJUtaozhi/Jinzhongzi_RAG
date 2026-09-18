#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${SENTIMENT_MODEL_HOST_DIR:-./models/chinese-emotion}"
MODEL_FILE="${MODEL_DIR}/model.safetensors"

if [[ ! -s "${MODEL_FILE}" ]]; then
  echo "ERROR: offline sentiment weight missing: ${MODEL_FILE}" >&2
  echo "Run scripts/download_sentiment_model.py on a connected computer first." >&2
  exit 1
fi

echo "[1/5] Offline model present"
du -h "${MODEL_FILE}"

echo "[2/5] Compose services"
docker compose ps

echo "[3/5] Knowledge and sentiment health"
curl -fsS http://localhost:8002/health
echo
curl -fsS -X POST http://localhost:8002/v1/text/analyze-sentiment \
  -H 'Content-Type: application/json' \
  -d '{"text":"我最近压力很大，晚上也很害怕，常常睡不着"}'
echo

echo "[4/5] Agent pure-text regression"
curl -fsS -X POST http://localhost:8003/v1/agent/chat \
  -H 'Content-Type: application/json' \
  -d '{"text":"我最近压力很大，晚上也很害怕，常常睡不着","session_id":"week7-verify","history":[]}'
echo

echo "[5/5] Memory and resource snapshot"
free -h
docker stats --no-stream

echo "Week 7.1 verification finished. Confirm total steady memory is below 7 GB."

