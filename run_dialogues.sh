#!/bin/bash
export OPENROUTER_API_KEY="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"
cd /Users/ethancastro/llm-no-curse-no-18
python3 -m kidschat.dialogues.generate_dialogues \
    --output dialogues/kidschat_dialogues.jsonl \
    --num-dialogues 5000 \
    --variations-per-topic 200
