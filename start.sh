#!/bin/bash
set -e

echo "=== CrownAI Server Setup ==="

# Install Python dependencies
echo "Installing dependencies..."
pip3 install fastapi uvicorn opencv-python-headless requests pydantic python-multipart -q

# Download api_server.py (saved as hair_api.py to avoid name clash with ComfyUI internals)
echo "Downloading hair_api.py..."
curl -s https://raw.githubusercontent.com/mkdamike/crownai-server/main/api_server.py -o /workspace/hair_api.py

# Find ComfyUI
echo "Finding ComfyUI..."
COMFYUI_DIR=""
for path in /workspace/ComfyUI /workspace/runpod-slim/ComfyUI /comfyui /root/ComfyUI; do
    if [ -f "$path/main.py" ]; then
        COMFYUI_DIR="$path"
        break
    fi
done

if [ -z "$COMFYUI_DIR" ]; then
    echo "ERROR: ComfyUI not found! Make sure you're using a ComfyUI pod template."
    exit 1
fi
echo "Found ComfyUI at: $COMFYUI_DIR"

# Find Python for ComfyUI (prefer venv)
COMFYUI_PYTHON="python3"
if [ -f "$COMFYUI_DIR/.venv/bin/python" ]; then
    COMFYUI_PYTHON="$COMFYUI_DIR/.venv/bin/python"
fi

# Find/create models directory
MODELS_DIR="$COMFYUI_DIR/models/checkpoints"
mkdir -p "$MODELS_DIR"

# Download model if missing
MODEL_PATH="$MODELS_DIR/sd-v1-5-inpainting.ckpt"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading inpainting model (~4GB, please wait)..."
    wget -q --show-progress -O "$MODEL_PATH" "https://huggingface.co/runwayml/stable-diffusion-inpainting/resolve/main/sd-v1-5-inpainting.ckpt"
    echo "Model downloaded!"
else
    echo "Model already exists, skipping download."
fi

# Start ComfyUI in background
echo "Starting ComfyUI in background..."
cd "$COMFYUI_DIR"
nohup "$COMFYUI_PYTHON" main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &
echo "ComfyUI starting (PID $!)..."

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
for i in $(seq 1 90); do
    if curl -s http://localhost:8188 > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "ERROR: ComfyUI failed to start. Check /workspace/comfyui.log"
        tail -20 /workspace/comfyui.log
        exit 1
    fi
    printf "."
    sleep 2
done

# Start API server using localhost for ComfyUI
echo ""
echo "Starting API server..."
cd /workspace
COMFYUI_URL="http://localhost:8188" python3 hair_api.py
