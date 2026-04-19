#!/bin/bash
# =============================================================================
# KidsChat Training Script for Vast.ai
# Run on: 4x RTX 4090 instance
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  KidsChat Training Pipeline"
echo "============================================"

# ---------------------------------------------------------------------------
# 1. Setup environment
# ---------------------------------------------------------------------------
echo "[1/6] Setting up environment..."

cd /workspace

# Install uv if not present
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone nanochat if not present
if [ ! -d "nanochat" ]; then
    git clone https://github.com/karpathy/nanochat.git
fi

cd nanochat
uv venv --python 3.12
source .venv/bin/activate
uv sync --extra gpu

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/nanochat_data

mkdir -p "$NANOCHAT_BASE_DIR/base_data_climbmix"

# ---------------------------------------------------------------------------
# 2. Copy data into nanochat expected locations
# ---------------------------------------------------------------------------
echo "[2/6] Setting up data..."

# Copy clean shards to nanochat data dir
if [ ! -f "$NANOCHAT_BASE_DIR/base_data_climbmix/shard_00000.parquet" ]; then
    echo "  Copying clean corpus shards..."
    cp /workspace/data/clean_shards/*.parquet "$NANOCHAT_BASE_DIR/base_data_climbmix/"
    echo "  $(ls $NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet | wc -l) shards copied"
fi

# Copy dialogues as identity conversations for SFT
if [ ! -f "$NANOCHAT_BASE_DIR/kidschat_dialogues.jsonl" ]; then
    echo "  Copying dialogue data..."
    cp /workspace/data/kidschat_dialogues.jsonl "$NANOCHAT_BASE_DIR/kidschat_dialogues.jsonl"
fi

echo "  Data ready."

# ---------------------------------------------------------------------------
# 3. Train tokenizer
# ---------------------------------------------------------------------------
echo "[3/6] Training tokenizer..."

if [ ! -d "$NANOCHAT_BASE_DIR/tokenizer" ]; then
    python -m scripts.tok_train \
        --max-chars=2000000000 \
        --doc-cap=10000 \
        --vocab-size=32768
    echo "  Tokenizer trained."
else
    echo "  Tokenizer already exists, skipping."
fi

# ---------------------------------------------------------------------------
# 4. Pretrain base model (~300M params)
# ---------------------------------------------------------------------------
echo "[4/6] Pretraining base model (depth=20, ~300M params)..."
echo "  Using 4x GPUs with torchrun"

NGPU=$(nvidia-smi -L | wc -l)
echo "  Detected $NGPU GPUs"

torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
    --depth=20 \
    --device-batch-size=16 \
    --run="kidschat_pretrain"

echo "  Pretraining complete."

# ---------------------------------------------------------------------------
# 5. SFT with kid-safe dialogues
# ---------------------------------------------------------------------------
echo "[5/6] Running SFT with kidschat dialogues..."

torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run="kidschat_sft"

echo "  SFT complete."

# ---------------------------------------------------------------------------
# 6. Export model
# ---------------------------------------------------------------------------
echo "[6/6] Training complete!"
echo ""
echo "Checkpoints saved in: $NANOCHAT_BASE_DIR/"
echo "  Base model:  $NANOCHAT_BASE_DIR/base_checkpoints/"
echo "  SFT model:   $NANOCHAT_BASE_DIR/chatsft_checkpoints/"
echo ""
echo "To chat with the model:"
echo "  python -m scripts.chat_web"
echo ""
echo "============================================"
echo "  KidsChat Training DONE"
echo "============================================"
