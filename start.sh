#!/bin/bash
set -e

echo "=== CrownAI Server Setup ==="

# Install Python dependencies
echo "Installing dependencies..."
pip3 install fastapi uvicorn opencv-python-headless requests pydantic python-multipart -q

# Download hair_api.py (named to avoid clash with ComfyUI internals)
echo "Downloading hair_api.py..."
curl -s https://raw.githubusercontent.com/mkdamike/crownai-server/main/api_server.py -o /workspace/hair_api.py

# Always do a clean ComfyUI install to avoid broken pod templates
COMFYUI_DIR="/workspace/ComfyUI"
if [ ! -f "$COMFYUI_DIR/main.py" ] || [ ! -d "$COMFYUI_DIR/api_server" ]; then
    echo "Installing clean ComfyUI..."
    rm -rf "$COMFYUI_DIR"
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    cd "$COMFYUI_DIR"
    pip3 install -r requirements.txt -q
    echo "ComfyUI installed!"
else
    echo "ComfyUI already installed and healthy."
fi

# Download model if missing (check both possible locations)
MODEL_PATH="$COMFYUI_DIR/models/checkpoints/sd-v1-5-inpainting.ckpt"
mkdir -p "$COMFYUI_DIR/models/checkpoints"
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
nohup python3 main.py --listen 0.0.0.0 --port 8188 > /workspace/comfyui.log 2>&1 &
echo "ComfyUI starting (PID $!)..."

# Wait for ComfyUI to be ready
echo "Waiting for ComfyUI to be ready..."
for i in $(seq 1 150); do
    if curl -s http://localhost:8188 > /dev/null 2>&1; then
        echo "ComfyUI is ready!"
        break
    fi
    if [ $i -eq 150 ]; then
        echo "ERROR: ComfyUI failed to start. Check /workspace/comfyui.log"
        tail -20 /workspace/comfyui.log
        exit 1
    fi
    printf "."
    sleep 2
done

# Kill anything on port 8000 before starting
echo ""
echo "Clearing port 8000..."
pkill -f hair_api.py 2>/dev/null || true
sleep 2

# Start API server
echo "Starting API server..."
cd /workspace
COMFYUI_URL="http://localhost:8188" python3 hair_api.py
