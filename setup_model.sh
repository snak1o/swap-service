#!/bin/bash
# Run this ONCE on a machine with the Network Volume mounted at /runpod-volume
# to download the Wan2.2 model.
#
# Example: bash setup_model.sh

set -e

MODELS_DIR="${MODELS_DIR:-/runpod-volume/models}"
WAN_MODEL_DIR="${MODELS_DIR}/Wan2.2-Animate-14B"

echo ""
echo "=========================================="
echo "  Wan2.2 Model Download"
echo "=========================================="

if [ -d "${WAN_MODEL_DIR}" ] && [ "$(ls -A ${WAN_MODEL_DIR} 2>/dev/null)" ]; then
    MODEL_SIZE=$(du -sh "${WAN_MODEL_DIR}" 2>/dev/null | cut -f1)
    echo "Model already downloaded (${MODEL_SIZE}). Skipping."
else
    echo "Downloading Wan2.2-Animate-14B (~50GB)..."
    echo "This will take 30-60 minutes. One time only."
    pip install -q huggingface_hub 2>/dev/null
    mkdir -p "${MODELS_DIR}"
    huggingface-cli download Wan-AI/Wan2.2-Animate-14B \
        --local-dir "${WAN_MODEL_DIR}" \
        --local-dir-use-symlinks False
    echo "Model downloaded!"
fi

echo ""
echo "=========================================="
echo "  ✅ Done! Model is at: ${WAN_MODEL_DIR}"
echo "=========================================="
