#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 /absolute/path/to/clear-face.jpg" >&2
  exit 2
fi

image_path="$1"
if [[ ! -f "$image_path" ]]; then
  echo "Image not found: $image_path" >&2
  exit 2
fi

echo "1/3 Health"
curl -fsS http://localhost:8001/health
echo

echo "2/3 Real face analysis (must contain AU12_r and finish under 10 s)"
started="$(date +%s)"
result="$(curl -fsS -F "image=@${image_path}" http://localhost:8001/v1/vision/analyze-face)"
elapsed="$(( $(date +%s) - started ))"
echo "$result"
echo "elapsed=${elapsed}s"
if [[ "$elapsed" -ge 10 ]] || ! echo "$result" | grep -q '"AU12_r"'; then
  echo "Vision acceptance check failed." >&2
  exit 1
fi

echo "3/3 Agent compatibility route"
curl -fsS \
  -F 'text=最近有些焦虑，请结合量表和照片给我建议' \
  -F 'assessment={"scale":"PHQ-9","total_score":12,"severity":"中度抑郁倾向","item9_score":0}' \
  -F "image=@${image_path}" \
  http://localhost:8003/v1/agent/analyze
echo

