#!/bin/bash
# =============================================================================
# Deploy KidsChat demo to a RunPod pod
# Usage: ./deploy.sh <ssh-host> <ssh-port>
#   e.g. ./deploy.sh ssh.runpod.io 12345
#        ./deploy.sh root@209.x.x.x 22
#
# What it does:
#   1. Uploads checkpoints + run_server.sh to the pod
#   2. Starts the chat server in a tmux session
#   3. Starts ngrok tunnel for a public URL
#   4. Prints the public URL
# =============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <ssh-host> <ssh-port>"
    echo ""
    echo "Find these in RunPod → My Pods → Connect → SSH:"
    echo "  ssh root@<host> -p <port> -i ~/.ssh/id_ed25519"
    echo ""
    echo "Example: $0 ssh.runpod.io 12345"
    exit 1
fi

HOST="$1"
PORT="$2"
SSH_OPTS="-o StrictHostKeyChecking=no -p $PORT"
SCP_OPTS="-o StrictHostKeyChecking=no -P $PORT"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "============================================"
echo "  Deploying KidsChat Demo"
echo "  Host: $HOST:$PORT"
echo "============================================"

# --- Upload checkpoints ---
echo ""
echo "[1/4] Uploading checkpoints (~3.8 GB)..."

ssh $SSH_OPTS "$HOST" "mkdir -p /workspace/checkpoints/tokenizer /workspace/checkpoints/chatsft_checkpoints/d12"

echo "  Uploading tokenizer..."
scp $SCP_OPTS "$SCRIPT_DIR"/checkpoints/tokenizer/* "$HOST":/workspace/checkpoints/tokenizer/

echo "  Uploading SFT model (depth-12, step 299)..."
# Upload model weights and metadata (skip optimizer — not needed for inference)
scp $SCP_OPTS \
    "$SCRIPT_DIR"/checkpoints/chatsft_checkpoints/d12/model_000299.pt \
    "$SCRIPT_DIR"/checkpoints/chatsft_checkpoints/d12/meta_000299.json \
    "$HOST":/workspace/checkpoints/chatsft_checkpoints/d12/

echo "  Done."

# --- Upload server script ---
echo ""
echo "[2/4] Uploading server script..."
scp $SCP_OPTS "$SCRIPT_DIR"/run_server.sh "$HOST":/workspace/
ssh $SSH_OPTS "$HOST" "chmod +x /workspace/run_server.sh"
echo "  Done."

# --- Start server in tmux ---
echo ""
echo "[3/4] Starting chat server on pod..."

ssh $SSH_OPTS "$HOST" bash -s <<'REMOTE_SCRIPT'
# Kill any existing demo session
tmux kill-session -t demo 2>/dev/null || true

# Start server in tmux
tmux new-session -d -s demo -n server '/workspace/run_server.sh; bash'

echo "  Server starting in tmux session 'demo'..."
REMOTE_SCRIPT

echo "  Done."

# --- Start ngrok tunnel ---
echo ""
echo "[4/4] Setting up ngrok tunnel..."

ssh $SSH_OPTS "$HOST" bash -s <<'REMOTE_SCRIPT'
# Install ngrok if missing
if ! command -v ngrok &>/dev/null; then
    echo "  Installing ngrok..."
    curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok-v3-stable-linux-amd64.tgz | tar xz -C /usr/local/bin
fi

# Start ngrok in a tmux window
tmux new-window -t demo -n ngrok 'ngrok http 8000 --log=stdout > /tmp/ngrok.log 2>&1; bash'

# Wait for ngrok to establish tunnel
echo "  Waiting for ngrok tunnel..."
for i in $(seq 1 30); do
    sleep 2
    URL=$(curl -s http://localhost:4040/api/tunnels 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    tunnels = data.get('tunnels', [])
    for t in tunnels:
        if t.get('proto') == 'https':
            print(t['public_url'])
            break
    else:
        if tunnels:
            print(tunnels[0]['public_url'])
except:
    pass
" 2>/dev/null) || true
    if [ -n "${URL:-}" ]; then
        echo "$URL"
        exit 0
    fi
done
echo "TIMEOUT"
REMOTE_SCRIPT

# Capture the URL from ssh output
echo ""
echo "============================================"
echo "  Deployment complete!"
echo "============================================"
echo ""
echo "The public URL was printed above (https://xxxx.ngrok-free.app)"
echo ""
echo "To check the URL later:"
echo "  ssh $SSH_OPTS $HOST 'curl -s http://localhost:4040/api/tunnels | python3 -c \"import sys,json; print(json.load(sys.stdin)[\\\"tunnels\\\"][0][\\\"public_url\\\"])\"'"
echo ""
echo "To monitor the server:"
echo "  ssh $SSH_OPTS $HOST -t 'tmux attach -t demo'"
echo ""
echo "To stop:"
echo "  ssh $SSH_OPTS $HOST 'tmux kill-session -t demo'"
echo ""
echo "Note: If ngrok requires auth, run this first on the pod:"
echo "  ngrok config add-authtoken <your-token>"
echo "  (Get a free token at https://dashboard.ngrok.com)"
