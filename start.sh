#!/bin/bash
# Entrypoint for RunPod GPU Worker
# Checks for models on Network Volume, downloads if missing, then starts handler

set -e

MODELS_DIR="${MODELS_DIR:-/runpod-volume/models}"
WAN_MODEL_NAME="${WAN_MODEL_NAME:-Wan-AI/Wan2.2-Animate-14B}"
WAN_LOCAL_DIR="${MODELS_DIR}/Wan2.2-Animate-14B"

echo "=== BodySwap GPU Worker (Wan2.2-Animate-14B) ==="
echo "Models dir: ${MODELS_DIR}"

# --- Check & download Wan2.2-Animate-14B ---
if [ ! -d "${WAN_LOCAL_DIR}" ] || [ -z "$(ls -A ${WAN_LOCAL_DIR} 2>/dev/null)" ]; then
    echo "[start.sh] Wan2.2-Animate-14B not found. Downloading to Network Volume..."
    echo "[start.sh] This will take ~30-60 min on first run. Subsequent runs will be instant."
    python /app/download_models.py
else
    echo "[start.sh] Wan2.2-Animate-14B found at ${WAN_LOCAL_DIR}. Skipping download."
fi

# --- Verify key model files ---
echo "[start.sh] Verifying models..."

if [ ! -f "${MODELS_DIR}/sam2.1_hiera_large.pt" ]; then
    echo "[start.sh] SAM2 not found, downloading..."
    wget -q -P "${MODELS_DIR}" \
        https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt || true
fi

if [ ! -f "${MODELS_DIR}/GFPGANv1.4.pth" ]; then
    echo "[start.sh] GFPGAN not found, downloading..."
    wget -q -P "${MODELS_DIR}" \
        https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth || true
fi

# InsightFace buffalo_l
if [ ! -d "${MODELS_DIR}/insightface" ]; then
    echo "[start.sh] InsightFace models not found, will download on first use."
fi

echo "[start.sh] All models verified. Starting handler..."
exec python -u /app/handler.py
