#!/bin/bash
set -e

echo "=== CrownAI Server Setup ==="

# Install Python dependencies
echo "Installing dependencies..."
pip3 install fastapi uvicorn opencv-python-headless requests pydantic python-multipart -q

# Download api_server.py
echo "Downloading api_server.py..."
curl -s https://raw.githubusercontent.com/mkdamike/crownai-server/main/api_server.py -o /workspace/api_server.py

# Find ComfyUI models directory
MODELS_DIR=""
for dir in /workspace/ComfyUI/models/checkpoints /comfyui/models/checkpoints /root/ComfyUI/models/checkpoints; do
    if [ -d "$dir" ]; then
        MODELS_DIR="$dir"
        break
    fi
done

if [ -z "$MODELS_DIR" ]; then
    echo "Creating ComfyUI models directory..."
    MODELS_DIR="/workspace/ComfyUI/models/checkpoints"
    mkdir -p "$MODELS_DIR"
fi

# Download model if missing
MODEL_PATH="$MODELS_DIR/sd-v1-5-inpainting.ckpt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading inpainting model (~4GB, please wait)..."
    wget -q --show-progress -O "$MODEL_PATH" "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
    echo "Model downloaded!"
else
    echo "Model already exists, skipping download."
fi

# Auto-detect ComfyUI URL from pod ID
POD_ID=$(hostname)
COMFYUI_URL="https://${POD_ID}-8188.proxy.runpod.net"
echo "ComfyUI URL: $COMFYUI_URL"

# Start server
echo "Starting API server..."
cd /workspace
COMFYUI_URL="$COMFYUI_URL" python3 api_server.py
