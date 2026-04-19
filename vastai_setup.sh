#!/bin/bash
# =============================================================================
# Vast.ai Setup — Run this FIRST after SSH'ing into the instance
# Uploads data, patches nanochat to use our kid-safe corpus + dialogues
# =============================================================================
set -euo pipefail

echo "============================================"
echo "  KidsChat — Vast.ai Setup"
echo "============================================"

cd /workspace

# ---------------------------------------------------------------------------
# 1. Clone nanochat & install deps
# ---------------------------------------------------------------------------
echo "[1/4] Cloning nanochat & installing dependencies..."
if [ ! -d "nanochat" ]; then
    git clone https://github.com/karpathy/nanochat.git
fi
cd nanochat

# Install uv
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

UV_VENV_CLEAR=1 uv venv --python 3.12
source .venv/bin/activate
uv sync --extra gpu

# ---------------------------------------------------------------------------
# 2. Set up data directories
# ---------------------------------------------------------------------------
echo "[2/4] Setting up data directories..."
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
mkdir -p "$NANOCHAT_BASE_DIR/base_data_climbmix"

# Copy our clean shards as the training data (copy, not symlink, for reliability)
echo "  Copying clean corpus shards..."
cp /workspace/data/clean_shards/*.parquet "$NANOCHAT_BASE_DIR/base_data_climbmix/"
echo "  $(ls $NANOCHAT_BASE_DIR/base_data_climbmix/*.parquet | wc -l) shards copied"

# Copy dialogues
cp /workspace/data/kidschat_dialogues.jsonl "$NANOCHAT_BASE_DIR/kidschat_dialogues.jsonl"
echo "  Dialogues copied"

# ---------------------------------------------------------------------------
# 3. Patch chat_sft.py to use our dialogues instead of SmolTalk
# ---------------------------------------------------------------------------
echo "[3/4] Patching SFT to use kidschat dialogues..."

# Replace the training mixture in chat_sft.py
python3 -c "
import re

with open('scripts/chat_sft.py', 'r') as f:
    content = f.read()

# Replace the training tasks block
old_block = '''identity_conversations_filepath = os.path.join(base_dir, \"identity_conversations.jsonl\")
train_tasks = [
    SmolTalk(split=\"train\"), # 460K rows of general conversations
    CustomJSON(filepath=identity_conversations_filepath), # 1000 rows of synthetic identity conversations
    CustomJSON(filepath=identity_conversations_filepath), # 2 epochs of these
    *[MMLU(subset=\"all\", split=\"auxiliary_train\") for _ in range(args.mmlu_epochs)], # 100K rows per epoch
    *[GSM8K(subset=\"main\", split=\"train\") for _ in range(args.gsm8k_epochs)], # 8K rows per epoch
    SimpleSpelling(size=200000, split=\"train\"), # 200K rows of Simple Spelling (e.g. spell the word 'apple')
    SpellingBee(size=80000, split=\"train\"), # 80K rows of Spelling Bee (e.g. how many 'r' are in 'strawberry'?)
]'''

new_block = '''kidschat_filepath = os.path.join(base_dir, \"kidschat_dialogues.jsonl\")
train_tasks = [
    # KidsChat dialogues — 4500+ kid-safe tutoring conversations (10 epochs)
    *[CustomJSON(filepath=kidschat_filepath) for _ in range(10)],
    # Keep some general tasks for well-roundedness
    *[MMLU(subset=\"all\", split=\"auxiliary_train\") for _ in range(args.mmlu_epochs)],
    *[GSM8K(subset=\"main\", split=\"train\") for _ in range(args.gsm8k_epochs)],
    SimpleSpelling(size=200000, split=\"train\"),
    SpellingBee(size=80000, split=\"train\"),
]'''

content = content.replace(old_block, new_block)

with open('scripts/chat_sft.py', 'w') as f:
    f.write(content)

if 'kidschat_dialogues.jsonl' in content:
    print('  chat_sft.py patched successfully')
else:
    print('  ERROR: Patch failed! nanochat may have updated. Exiting.')
    import sys; sys.exit(1)
"

# ---------------------------------------------------------------------------
# 4. Write the run script
# ---------------------------------------------------------------------------
echo "[4/4] Creating run script..."

cat > /workspace/run_train.sh << 'TRAINEOF'
#!/bin/bash
set -euo pipefail
cd /workspace/nanochat
source .venv/bin/activate
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=/workspace/nanochat_cache
# Disable wandb so it never prompts (would block unattended training)
export WANDB_MODE=disabled

NGPU=$(nvidia-smi -L | wc -l)
echo "Training with $NGPU GPUs"

# Step 1: Train tokenizer (skip if already done)
if [ ! -f "$NANOCHAT_BASE_DIR/tokenizer/tokenizer.pkl" ]; then
    echo "=== Training tokenizer ==="
    python -m scripts.tok_train --max-chars=2000000000 --vocab-size=32768
else
    echo "=== Tokenizer already trained, skipping ==="
fi

# Step 2: Pretrain (~300M params, depth=20)
# Use --window-pattern L for full attention (SDPA fallback on Blackwell doesn't support sliding window)
echo "=== Pretraining base model (depth=12) ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.base_train -- \
    --depth=12 \
    --device-batch-size=16 \
    --window-pattern=L \
    --run="kidschat_pretrain" 2>&1 | tee /workspace/pretrain.log

echo "=== Pretraining finished ==="

# Step 3: SFT with kidschat dialogues
echo "=== SFT with kidschat dialogues ==="
torchrun --standalone --nproc_per_node=$NGPU -m scripts.chat_sft -- \
    --device-batch-size=16 \
    --run="kidschat_sft" 2>&1 | tee /workspace/sft.log

echo "=== DONE ==="
echo "Checkpoints: $NANOCHAT_BASE_DIR/"
echo "Chat: python -m scripts.chat_web"
TRAINEOF
chmod +x /workspace/run_train.sh

echo ""
echo "============================================"
echo "  Setup complete!"
echo "============================================"
echo ""
echo "To start training, run:"
echo "  /workspace/run_train.sh"
echo ""
echo "Or step by step:"
echo "  cd /workspace/nanochat && source .venv/bin/activate"
echo "  export NANOCHAT_BASE_DIR=/workspace/nanochat_cache"
echo "  export OMP_NUM_THREADS=1"
echo ""
echo "  # Tokenizer"
echo "  python -m scripts.tok_train"
echo ""
echo "  # Pretrain"
echo "  torchrun --standalone --nproc_per_node=\$NGPU -m scripts.base_train -- --depth=20 --device-batch-size=16 --window-pattern=L --run=kidschat_pretrain"
echo ""
echo "  # SFT"
echo "  torchrun --standalone --nproc_per_node=\$NGPU -m scripts.chat_sft -- --device-batch-size=16 --run=kidschat_sft"
