#!/bin/bash
# =============================================================================
# Upload data to Vast.ai instance
# Usage: ./upload_to_vastai.sh <instance-ip> <ssh-port>
# Example: ./upload_to_vastai.sh 192.168.1.1 22
# =============================================================================
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <instance-ip> <ssh-port>"
    echo "Example: $0 ssh4.vast.ai 12345"
    exit 1
fi

HOST=$1
PORT=$2
SSH="ssh -p $PORT root@$HOST"
SCP="scp -P $PORT"

echo "Uploading to $HOST:$PORT..."

# Create directories on remote
$SSH "mkdir -p /workspace/data/clean_shards"

# Upload clean shards
echo "Uploading 24 clean corpus shards..."
$SCP kidschat_data/clean_shards/*.parquet "root@$HOST:/workspace/data/clean_shards/"

# Upload dialogues
echo "Uploading dialogues..."
$SCP dialogues/kidschat_dialogues.jsonl "root@$HOST:/workspace/data/"

# Upload setup script
echo "Uploading setup script..."
$SCP vastai_setup.sh "root@$HOST:/workspace/"
$SSH "chmod +x /workspace/vastai_setup.sh"

echo ""
echo "Upload complete! Now SSH in and run:"
echo "  ssh -p $PORT root@$HOST"
echo "  /workspace/vastai_setup.sh"
echo "  /workspace/run_train.sh"
