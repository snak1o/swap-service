#!/bin/bash
set -e

MODELS_DIR="${MODELS_DIR:-/runpod-volume/models}"
WAN_LOCAL_DIR="${MODELS_DIR}/Wan2.2-Animate-14B"

echo "=== Wan2.2 Replace Worker ==="
echo "Models: ${MODELS_DIR}"

# Check & download Wan2.2
if [ ! -d "${WAN_LOCAL_DIR}" ] || [ -z "$(ls -A ${WAN_LOCAL_DIR} 2>/dev/null)" ]; then
    echo "[start] Downloading Wan2.2-Animate-14B (~50GB, first time only)..."
    python /app/download_models.py
else
    echo "[start] Wan2.2-Animate-14B found. Skipping download."
fi

echo "[start] Starting handler..."
exec python -u /app/handler.py
