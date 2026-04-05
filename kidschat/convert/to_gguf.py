"""
to_gguf.py — Convert a nanochat checkpoint to GGUF format for local inference.

This is a "lossy v1" conversion that maps nanochat's custom architecture to a
LLaMA-like structure. Some nanochat-specific features (smear gate, value
embeddings, backout lambda, x0 lambdas) are dropped during conversion.

For best-quality inference, use nanochat's native chat_cli.py or chat_web.py.
This conversion enables deployment via llama.cpp and Ollama at some quality cost.

Usage:
    python -m kidschat.convert.to_gguf --checkpoint-dir ./checkpoints/d20 --output model.gguf

Requires: pip install gguf torch
"""

import argparse
import json
import os
import struct
import sys

import torch
import numpy as np

try:
    import gguf
except ImportError:
    print("ERROR: gguf package not installed. Run: pip install gguf")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def find_latest_checkpoint(checkpoint_dir: str) -> tuple[str, str]:
    """Find the latest model checkpoint and its metadata file."""
    model_files = sorted(
        f for f in os.listdir(checkpoint_dir)
        if f.startswith("model_") and f.endswith(".pt")
    )
    if not model_files:
        raise FileNotFoundError(f"No model_*.pt files in {checkpoint_dir}")

    latest = model_files[-1]
    step = latest.replace("model_", "").replace(".pt", "")
    meta_file = f"meta_{step}.json"

    model_path = os.path.join(checkpoint_dir, latest)
    meta_path = os.path.join(checkpoint_dir, meta_file)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found: {meta_path}")

    return model_path, meta_path


def load_checkpoint(model_path: str, meta_path: str) -> tuple[dict, dict]:
    """Load model weights and metadata."""
    print(f"Loading model from {model_path}...")
    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

    # Strip _orig_mod. prefix from torch.compile
    cleaned = {}
    for k, v in state_dict.items():
        key = k.replace("_orig_mod.", "")
        cleaned[key] = v
    state_dict = cleaned

    with open(meta_path) as f:
        meta = json.load(f)

    return state_dict, meta


# ---------------------------------------------------------------------------
# Architecture extraction
# ---------------------------------------------------------------------------

def extract_config(state_dict: dict, meta: dict) -> dict:
    """Extract model configuration from metadata and weights."""
    # Try to get from metadata first
    config = {}

    # From metadata
    config["depth"] = meta.get("depth", meta.get("n_layer"))
    config["max_seq_len"] = meta.get("max_seq_len", 2048)

    # Infer from weight shapes
    wte = state_dict.get("transformer.wte.weight")
    if wte is not None:
        config["vocab_size"] = wte.shape[0]
        config["model_dim"] = wte.shape[1]

    # Count layers
    layer_keys = [k for k in state_dict if k.startswith("transformer.h.")]
    if layer_keys:
        layer_nums = set(int(k.split(".")[2]) for k in layer_keys)
        config["n_layer"] = max(layer_nums) + 1

    # Infer head dimensions from attention weights
    # Q projection: transformer.h.0.attn.c_q.weight -> (n_head * head_dim, model_dim)
    q_key = "transformer.h.0.attn.c_q.weight"
    if q_key in state_dict:
        q_shape = state_dict[q_key].shape
        config["head_dim"] = 128  # nanochat default
        config["n_head"] = q_shape[0] // config["head_dim"]

    # KV projection for GQA
    kv_key = "transformer.h.0.attn.c_k.weight"
    if kv_key in state_dict:
        kv_shape = state_dict[kv_key].shape
        config["n_kv_head"] = kv_shape[0] // config["head_dim"]

    config["rope_theta"] = 100000.0  # nanochat default

    return config


# ---------------------------------------------------------------------------
# Weight mapping: nanochat -> LLaMA-like GGUF
# ---------------------------------------------------------------------------

def map_weights(state_dict: dict, config: dict) -> dict:
    """Map nanochat weights to LLaMA-like tensor names for GGUF."""
    mapped = {}
    n_layer = config["n_layer"]

    # Token embeddings
    if "transformer.wte.weight" in state_dict:
        mapped["token_embd.weight"] = state_dict["transformer.wte.weight"]

    # Output head (untied in nanochat)
    if "lm_head.weight" in state_dict:
        mapped["output.weight"] = state_dict["lm_head.weight"]

    # Per-layer mappings
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"

        # Attention Q, K, V projections
        for src, dst in [
            ("attn.c_q.weight", "attn_q.weight"),
            ("attn.c_k.weight", "attn_k.weight"),
            ("attn.c_v.weight", "attn_v.weight"),
            ("attn.c_proj.weight", "attn_output.weight"),
        ]:
            key = f"{prefix}.{src}"
            if key in state_dict:
                mapped[f"blk.{i}.{dst}"] = state_dict[key]

        # MLP
        for src, dst in [
            ("mlp.c_fc.weight", "ffn_up.weight"),
            ("mlp.c_proj.weight", "ffn_down.weight"),
        ]:
            key = f"{prefix}.{src}"
            if key in state_dict:
                mapped[f"blk.{i}.{dst}"] = state_dict[key]

        # If there's a gate projection (SwiGLU-style)
        gate_key = f"{prefix}.mlp.c_gate.weight"
        if gate_key in state_dict:
            mapped[f"blk.{i}.ffn_gate.weight"] = state_dict[gate_key]

    # Log what we skipped
    mapped_src_keys = set()
    for i in range(n_layer):
        prefix = f"transformer.h.{i}"
        for src in ["attn.c_q.weight", "attn.c_k.weight", "attn.c_v.weight",
                     "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight",
                     "mlp.c_gate.weight"]:
            mapped_src_keys.add(f"{prefix}.{src}")
    mapped_src_keys.add("transformer.wte.weight")
    mapped_src_keys.add("lm_head.weight")

    skipped = [k for k in state_dict if k not in mapped_src_keys]
    if skipped:
        print(f"\n  Skipped {len(skipped)} nanochat-specific tensors:")
        for k in skipped[:20]:
            print(f"    - {k} {list(state_dict[k].shape)}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    return mapped


# ---------------------------------------------------------------------------
# GGUF writing
# ---------------------------------------------------------------------------

def write_gguf(mapped_weights: dict, config: dict, output_path: str, quantize: str = "f16"):
    """Write weights to GGUF format."""
    print(f"\nWriting GGUF to {output_path}...")

    writer = gguf.GGUFWriter(output_path, arch="llama")

    # Metadata
    writer.add_context_length(config["max_seq_len"])
    writer.add_embedding_length(config["model_dim"])
    writer.add_block_count(config["n_layer"])
    writer.add_head_count(config["n_head"])
    writer.add_head_count_kv(config.get("n_kv_head", config["n_head"]))
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_feed_forward_length(config["model_dim"] * 4)  # approximate

    # Add tensors
    for name, tensor in mapped_weights.items():
        data = tensor.numpy().astype(np.float16 if quantize == "f16" else np.float32)
        writer.add_tensor(name, data)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    file_size = os.path.getsize(output_path)
    print(f"Wrote {output_path} ({file_size / 1024 / 1024:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Convert nanochat checkpoint to GGUF")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing model_*.pt and meta_*.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="kidschat.gguf",
        help="Output GGUF file path",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="f16",
        choices=["f16", "f32"],
        help="Weight precision (f16 recommended, further quantization via llama.cpp)",
    )
    args = parser.parse_args()

    # Load checkpoint
    model_path, meta_path = find_latest_checkpoint(args.checkpoint_dir)
    state_dict, meta = load_checkpoint(model_path, meta_path)

    # Extract config
    config = extract_config(state_dict, meta)
    print(f"\nModel config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Map weights
    mapped = map_weights(state_dict, config)
    print(f"\nMapped {len(mapped)} tensors for GGUF export")

    # Write GGUF
    write_gguf(mapped, config, args.output, args.quantize)

    # Print next steps
    print(f"""
{'=' * 60}
GGUF CONVERSION COMPLETE
{'=' * 60}

NOTE: This is a lossy conversion. nanochat-specific features
(smear gate, value embeddings, relu^2 activation, residual lambdas)
are NOT preserved. For best quality, use nanochat's native inference.

To further quantize (Q4_K_M) using llama.cpp:
  ./llama-quantize {args.output} kidschat-q4km.gguf Q4_K_M

To run with Ollama:
  1. Create a Modelfile:
     echo 'FROM ./kidschat-q4km.gguf' > Modelfile
     echo 'PARAMETER temperature 0.7' >> Modelfile
     echo 'SYSTEM "You are a friendly, educational assistant for children."' >> Modelfile

  2. Create and run:
     ollama create kidschat -f Modelfile
     ollama run kidschat

To run with llama.cpp directly:
  ./llama-cli -m kidschat-q4km.gguf -p "Hello! Can you teach me about space?" -n 256
""")


if __name__ == "__main__":
    main()
