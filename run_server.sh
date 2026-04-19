#!/bin/bash
# =============================================================================
# Run KidsChat demo server on a RunPod pod
# This script runs ON the pod (uploaded by deploy.sh)
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  KidsChat Demo Server Setup"
echo "============================================"

# --- Install uv if missing ---
if ! command -v uv &>/dev/null; then
    echo "[1/5] Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
else
    echo "[1/5] uv already installed."
fi
export PATH="$HOME/.local/bin:$PATH"

# --- Clone nanochat if not present ---
NANOCHAT_DIR="/workspace/nanochat"
if [ ! -d "$NANOCHAT_DIR" ]; then
    echo "[2/5] Cloning nanochat..."
    git clone https://github.com/anthropics/nanochat.git "$NANOCHAT_DIR"
else
    echo "[2/5] nanochat already present."
fi

# --- Install Python deps ---
echo "[3/5] Installing Python dependencies (GPU)..."
cd "$NANOCHAT_DIR"
uv sync --extra gpu

# --- Set up checkpoint symlinks so nanochat finds them ---
echo "[4/5] Linking checkpoints..."
NANOCHAT_BASE_DIR="/workspace/nanochat_data"
mkdir -p "$NANOCHAT_BASE_DIR"

# Symlink tokenizer + SFT checkpoints into nanochat's expected location
ln -sfn /workspace/checkpoints/tokenizer "$NANOCHAT_BASE_DIR/tokenizer"
ln -sfn /workspace/checkpoints/chatsft_checkpoints "$NANOCHAT_BASE_DIR/chatsft_checkpoints"

# --- Start the chat server ---
echo "[5/5] Starting chat server on port 8000..."
echo ""

export NANOCHAT_BASE_DIR="$NANOCHAT_BASE_DIR"
cd "$NANOCHAT_DIR"

# Detect GPU vs CPU
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    DEVICE="cuda"
    NUM_GPUS=$(nvidia-smi -L | wc -l)
    echo "  Device: $DEVICE ($NUM_GPUS GPU(s))"
else
    DEVICE="cpu"
    NUM_GPUS=1
    echo "  Device: $DEVICE"
fi

echo "  Port:   8000"
echo "  Model:  depth-12 SFT (step 299)"
echo ""
echo "  Server will be available at http://localhost:8000"
echo "  Waiting for ngrok tunnel (run in another tmux pane)..."
echo ""

uv run python -m scripts.chat_web \
    --device-type "$DEVICE" \
    --num-gpus "$NUM_GPUS" \
    --source sft \
    --port 8000
