#!/bin/bash
# ============================================
# Wan2.2 Replace — автоустановка на RunPod Pod
# Запусти: bash setup.sh
# ============================================
set -e

VOLUME="/runpod-volume"
MODELS_DIR="${VOLUME}/models"
WAN_MODEL_DIR="${MODELS_DIR}/Wan2.2-Animate-14B"
WAN_REPO="/workspace/Wan2.2"

echo ""
echo "=========================================="
echo "  Wan2.2 Replace — Setup"
echo "=========================================="
echo ""

# --- 1. Python зависимости ---
echo "[1/4] Устанавливаю зависимости..."
pip install -q flask gunicorn huggingface_hub[cli] pillow tqdm

# --- 2. Wan2.2 repo (код) ---
if [ -d "${WAN_REPO}" ]; then
    echo "[2/4] Wan2.2 repo уже есть, обновляю..."
    cd "${WAN_REPO}" && git pull && cd -
else
    echo "[2/4] Клонирую Wan2.2 repo..."
    git clone https://github.com/Wan-Video/Wan2.2.git "${WAN_REPO}"
fi

echo "[2/4] Устанавливаю зависимости Wan2.2..."
pip install -q -r "${WAN_REPO}/requirements.txt" 2>/dev/null || true

# FlashAttention (для H100)
pip install -q flash-attn --no-build-isolation 2>/dev/null || echo "[!] FlashAttention не установлен (не H100?)"

# --- 3. Модель на Network Volume ---
if [ -d "${WAN_MODEL_DIR}" ] && [ "$(ls -A ${WAN_MODEL_DIR} 2>/dev/null)" ]; then
    MODEL_SIZE=$(du -sh "${WAN_MODEL_DIR}" 2>/dev/null | cut -f1)
    echo "[3/4] Модель уже скачана (${MODEL_SIZE}). Пропускаю."
else
    echo "[3/4] Скачиваю Wan2.2-Animate-14B (~50GB)..."
    echo "       Это займёт 30-60 минут. Один раз."
    mkdir -p "${MODELS_DIR}"
    huggingface-cli download Wan-AI/Wan2.2-Animate-14B \
        --local-dir "${WAN_MODEL_DIR}" \
        --local-dir-use-symlinks False
    echo "[3/4] Модель скачана!"
fi

# --- 4. Копируем server.py в workspace ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cp "${SCRIPT_DIR}/server.py" /workspace/server.py 2>/dev/null || true

echo ""
echo "=========================================="
echo "  ✅ Готово!"
echo ""
echo "  Запусти сервер:"
echo "  python /workspace/server.py"
echo ""
echo "  API: https://POD_ID-8000.proxy.runpod.net"
echo "=========================================="
echo ""
