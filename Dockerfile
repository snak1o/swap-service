FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV WAN_REPO=/workspace/Wan2.2
ENV MODELS_DIR=/runpod-volume/models
ENV WAN_CKPT_DIR=/runpod-volume/models/Wan2.2-Animate-14B

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip \
    git wget curl ffmpeg libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxext6 libxrender-dev libsndfile1 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Clone Wan2.2 repo
RUN git clone https://github.com/Wan-Video/Wan2.2.git ${WAN_REPO}

# Install Wan2.2 requirements
RUN pip install --no-cache-dir -r ${WAN_REPO}/requirements.txt 2>/dev/null || true

# Install project requirements (may override some Wan2.2 deps)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Remove CPU-only onnxruntime if it snuck in
RUN pip uninstall onnxruntime -y 2>/dev/null || true

# Copy handler
COPY handler.py /handler.py

# Download model script (for first-time setup on network volume)
COPY setup_model.sh /setup_model.sh
RUN chmod +x /setup_model.sh

CMD ["python", "-u", "/handler.py"]
