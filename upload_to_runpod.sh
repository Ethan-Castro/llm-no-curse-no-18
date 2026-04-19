#!/bin/bash
# =============================================================================
# Upload data + scripts to RunPod instance
# Usage: ./upload_to_runpod.sh <ssh-host> <ssh-port>
#   e.g. ./upload_to_runpod.sh ssh.runpod.io 12345
#        ./upload_to_runpod.sh root@209.x.x.x 22
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
echo "  Uploading KidsChat data to RunPod"
echo "  Host: $HOST:$PORT"
echo "============================================"

# Create remote directories
echo "[1/4] Creating remote directories..."

ssh $SSH_OPTS "$HOST" "mkdir -p /workspace/data/clean_shards"

# Upload clean shards (~1.1 GB)
echo "[2/4] Uploading 24 clean corpus shards (~1.1 GB)..."
scp $SCP_OPTS "$SCRIPT_DIR"/kidschat_data/clean_shards/*.parquet "$HOST":/workspace/data/clean_shards/
echo "  Done."

# Upload dialogues
echo "[3/4] Uploading dialogues (6.4 MB)..."
scp $SCP_OPTS "$SCRIPT_DIR"/dialogues/kidschat_dialogues.jsonl "$HOST":/workspace/data/
echo "  Done."

# Upload setup script
echo "[4/4] Uploading setup script..."
scp $SCP_OPTS "$SCRIPT_DIR"/vastai_setup.sh "$HOST":/workspace/
ssh $SSH_OPTS "$HOST" "chmod +x /workspace/vastai_setup.sh"
echo "  Done."

echo ""
echo "============================================"
echo "  Upload complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. SSH in:    ssh $SSH_OPTS $HOST"
echo "  2. Start tmux: tmux"
echo "  3. Run setup:  /workspace/vastai_setup.sh"
echo "  4. Train:      /workspace/run_train.sh"
echo "  5. Detach:     Ctrl+B, then D"
echo "  6. Close laptop. Check back in ~8-12 hrs."
