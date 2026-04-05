# KidsChat

A from-scratch kid-safe language model trained on a curated corpus with **zero profanity and zero sexual content**. Built on [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy. Designed for children under 13.

## How It Works

1. **Build a clean corpus** from [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) (synthetic educational text) and [Project Gutenberg children's books](https://www.gutenberg.org/ebooks/bookshelf/18)
2. **Audit every document** with two-pass filtering (profanity-check + detoxify/toxic-bert)
3. **Generate tutoring dialogues** for supervised fine-tuning
4. **Train from scratch** using nanochat (~300M-700M parameters)
5. **Convert to GGUF** for local inference via Ollama or llama.cpp

Total cost: ~$86-137 (mostly GPU rental).

## Setup

```bash
cd kidschat
pip install -r requirements.txt
```

## Pipeline (Run in Order)

### Step 1: Build the Corpus

Downloads ~3B tokens from Cosmopedia (kid-safe subsets only) and Project Gutenberg children's literature. Outputs nanochat-compatible Parquet shards.

```bash
python -m kidschat.data.build_corpus --output-dir ./kidschat_data --target-tokens 3000000000
```

This takes several hours due to streaming downloads. It's fully resumable — rerun the same command if interrupted.

### Step 2: Audit the Corpus

Two-pass content safety filter:
- **Pass 1**: `profanity-check` (fast CPU-based) flags obvious profanity
- **Pass 2**: `detoxify` (BERT-based) catches toxicity, sexual content, threats

```bash
python -m kidschat.data.audit_corpus \
  --input-dir ./kidschat_data/shards \
  --output-dir ./kidschat_data/clean_shards
```

Audit logs are saved to `audit_log_profanity.jsonl` and `audit_log_toxicity.jsonl` for manual review.

### Step 3: Check Corpus Stats

Verify the corpus looks right before training.

```bash
python -m kidschat.data.corpus_stats --input-dir ./kidschat_data/clean_shards
```

### Step 4: Generate Tutoring Dialogues (Optional)

Generate 1,000 kid-friendly tutoring dialogues using Gemma 4 31B via OpenRouter for SFT. Requires an OpenRouter API key ($0.14/M input, $0.40/M output — ~$0.50 total for 1000 dialogues).

```bash
export OPENROUTER_API_KEY=...
python -m kidschat.dialogues.generate_dialogues --output dialogues/kidschat_dialogues.jsonl
```

### Step 5: Setup nanochat Data Path

Validates your shards and creates a symlink so nanochat can find them.

```bash
python -m kidschat.train.setup_data --shard-dir ./kidschat_data/clean_shards
```

### Step 6: Train with nanochat

```bash
# Clone nanochat
git clone https://github.com/karpathy/nanochat.git
cd nanochat
pip install -r requirements.txt

# Train tokenizer on your corpus
python -m scripts.tok_train

# Pretrain (~300M params, requires 4x GPU)
torchrun --nproc_per_node=4 -m scripts.base_train --depth=20

# SFT with tutoring dialogues (after pretraining)
python -m scripts.chat_sft \
  --train-files /path/to/kidschat_dialogues.jsonl \
  --depth=20
```

### Step 7: Convert to GGUF (Optional)

Convert the trained checkpoint for local inference. This is a lossy conversion — nanochat's native inference gives better quality.

```bash
python -m kidschat.convert.to_gguf \
  --checkpoint-dir ./checkpoints/d20 \
  --output kidschat.gguf
```

Then quantize and run locally:

```bash
# Quantize with llama.cpp
./llama-quantize kidschat.gguf kidschat-q4km.gguf Q4_K_M

# Run with Ollama
echo 'FROM ./kidschat-q4km.gguf' > Modelfile
ollama create kidschat -f Modelfile
ollama run kidschat
```

## GPU Rental

You need multi-GPU compute for training. Budget: $85-135.

| Provider | Hardware | Cost | Link |
|----------|----------|------|------|
| Vast.ai | 4x RTX 4090 spot | ~$1.20/hr | https://vast.ai |
| RunPod | 4x RTX 4090 spot | ~$1.50/hr | https://runpod.io |

Estimated training time: 70-90 hours on 4x RTX 4090.

## Data Sources

| Source | Content | Tokens | License |
|--------|---------|--------|---------|
| [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) | Synthetic textbooks, stories, educational articles | ~2.5-3.5B | ODC-By 1.0 |
| [Project Gutenberg](https://www.gutenberg.org/ebooks/bookshelf/18) | Children's fiction and literature | ~0.6-1.5B | Public domain |
| Gemma 4 31B generated (via OpenRouter) | Tutoring dialogues for SFT | ~500K | Your usage |

## Content Safety

Every document passes two independent filters before entering the training data:

1. **profanity-check** — ML-based profanity detector (threshold: 0.5)
2. **detoxify** — BERT toxicity classifier checking: toxic, severe_toxic, obscene, threat, insult, identity_hate, sexual_explicit (threshold: 0.3 on any label)

Audit logs are preserved for manual review. We recommend sampling 500+ documents from the final corpus for human verification.

## What to Expect

At 300M-700M parameters, the model will be able to:
- Hold simple conversations on educational topics
- Explain basic science, math, history, and nature
- Ask follow-up questions and encourage learning
- Write short stories and poems

It will NOT match the fluency of GPT-4 or Claude — this is a micro-tutor, not a frontier model.

## Project Structure

```
kidschat/
├── data/
│   ├── build_corpus.py        # Downloads and assembles training corpus
│   ├── audit_corpus.py        # Two-pass profanity/toxicity filtering
│   └── corpus_stats.py        # Corpus analytics and validation
├── train/
│   └── setup_data.py          # Validates shards, configures nanochat data path
├── dialogues/
│   └── generate_dialogues.py  # Generates tutoring dialogues via Gemma 4 31B (OpenRouter)
├── convert/
│   └── to_gguf.py             # Checkpoint → GGUF conversion (lossy v1)
├── requirements.txt
└── README.md
```
