"""
audit_corpus.py — Two-pass content safety filtering for the kid-safe corpus.

Pass 1: profanity-check (fast, CPU-based) — flags obvious profanity
Pass 2: detoxify (BERT-based) — catches subtler toxicity, sexual content, threats

Reads Parquet shards from the raw corpus, filters out flagged documents,
and writes clean shards in the same nanochat-compatible format.

Usage:
    python -m kidschat.data.audit_corpus --input-dir ./kidschat_data/shards --output-dir ./kidschat_data/clean_shards
"""

import argparse
import json
import os
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROFANITY_THRESHOLD = 0.3   # aggressive — false positives acceptable, false negatives are not
TOXICITY_THRESHOLD = 0.30   # detoxify score threshold (any label)
BATCH_SIZE_PROFANITY = 1000
BATCH_SIZE_TOXICITY = 64
ROW_GROUP_SIZE = 1024

# Detoxify labels to check
TOXICITY_LABELS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "sexual_explicit"]


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def load_progress(output_dir: str) -> dict:
    path = os.path.join(output_dir, "audit_progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"completed_shards": [], "stats": {"total_docs": 0, "removed_blocklist": 0, "removed_profanity": 0, "removed_toxicity": 0, "kept": 0}}


def save_progress(output_dir: str, progress: dict):
    path = os.path.join(output_dir, "audit_progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Pass 0: Hardcoded blocklist (failsafe)
# ---------------------------------------------------------------------------

def run_blocklist_pass(texts: list[str], log_file) -> list[tuple[str, bool]]:
    """Run hardcoded blocklist check. Returns list of (text, is_clean) pairs."""
    from kidschat.data.blocklist import contains_blocked_content

    results = []
    for text in texts:
        is_blocked, matched_term = contains_blocked_content(text)
        if is_blocked:
            log_entry = {
                "pass": "blocklist",
                "matched_term": matched_term,
                "snippet": text[:200],
            }
            log_file.write(json.dumps(log_entry) + "\n")
            results.append((text, False))
        else:
            results.append((text, True))
    return results


# ---------------------------------------------------------------------------
# Pass 1: profanity-check
# ---------------------------------------------------------------------------

def run_profanity_pass(texts: list[str], log_file) -> list[tuple[str, bool]]:
    """Run profanity-check over texts. Returns list of (text, is_clean) pairs."""
    from alt_profanity_check import predict_prob

    results = []
    flagged_count = 0

    for i in range(0, len(texts), BATCH_SIZE_PROFANITY):
        batch = texts[i : i + BATCH_SIZE_PROFANITY]
        probs = predict_prob(batch)

        for text, prob in zip(batch, probs):
            if prob > PROFANITY_THRESHOLD:
                flagged_count += 1
                log_entry = {
                    "pass": "profanity",
                    "score": float(prob),
                    "snippet": text[:200],
                }
                log_file.write(json.dumps(log_entry) + "\n")
                results.append((text, False))
            else:
                results.append((text, True))

    return results


# ---------------------------------------------------------------------------
# Pass 2: detoxify (toxic-bert)
# ---------------------------------------------------------------------------

def run_toxicity_pass(texts: list[str], log_file, device: str = "cpu") -> list[tuple[str, bool]]:
    """Run detoxify over texts that passed profanity check."""
    from detoxify import Detoxify

    model = Detoxify("original", device=device)
    results = []
    flagged_count = 0

    for i in range(0, len(texts), BATCH_SIZE_TOXICITY):
        batch = texts[i : i + BATCH_SIZE_TOXICITY]
        predictions = model.predict(batch)

        for j, text in enumerate(batch):
            is_toxic = False
            max_score = 0.0
            max_label = ""

            for label in TOXICITY_LABELS:
                if label in predictions:
                    score = predictions[label][j]
                    if score > max_score:
                        max_score = score
                        max_label = label
                    if score > TOXICITY_THRESHOLD:
                        is_toxic = True

            if is_toxic:
                flagged_count += 1
                log_entry = {
                    "pass": "toxicity",
                    "max_label": max_label,
                    "max_score": float(max_score),
                    "scores": {
                        label: float(predictions[label][j])
                        for label in TOXICITY_LABELS
                        if label in predictions
                    },
                    "snippet": text[:200],
                }
                log_file.write(json.dumps(log_entry) + "\n")
                results.append((text, False))
            else:
                results.append((text, True))

    return results


# ---------------------------------------------------------------------------
# Shard processing
# ---------------------------------------------------------------------------

def write_clean_shard(texts: list[str], shard_path: str):
    """Write clean texts to a nanochat-compatible Parquet shard."""
    import random

    # Pad to multiple of ROW_GROUP_SIZE
    remainder = len(texts) % ROW_GROUP_SIZE
    if remainder != 0:
        padding_needed = ROW_GROUP_SIZE - remainder
        padding = random.choices(texts, k=padding_needed)
        texts = texts + padding

    table = pa.Table.from_pydict({"text": texts})
    pq.write_table(
        table,
        shard_path,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )


def process_shard(
    shard_path: str,
    output_path: str,
    blocklist_log,
    profanity_log,
    toxicity_log,
    device: str,
) -> dict:
    """Process a single shard through all three filtering passes."""
    # Read all texts from the shard
    table = pq.read_table(shard_path, columns=["text"])
    texts = table.column("text").to_pylist()
    total = len(texts)

    # Pass 0: hardcoded blocklist (failsafe)
    blocklist_results = run_blocklist_pass(texts, blocklist_log)
    clean_after_p0 = [text for text, is_clean in blocklist_results if is_clean]
    removed_p0 = total - len(clean_after_p0)

    # Pass 1: profanity-check
    if clean_after_p0:
        profanity_results = run_profanity_pass(clean_after_p0, profanity_log)
        clean_after_p1 = [text for text, is_clean in profanity_results if is_clean]
        removed_p1 = len(clean_after_p0) - len(clean_after_p1)
    else:
        clean_after_p1 = []
        removed_p1 = 0

    # Pass 2: detoxify on remaining texts
    if clean_after_p1:
        toxicity_results = run_toxicity_pass(clean_after_p1, toxicity_log, device)
        clean_final = [text for text, is_clean in toxicity_results if is_clean]
        removed_p2 = len(clean_after_p1) - len(clean_final)
    else:
        clean_final = []
        removed_p2 = 0

    # Write clean shard (only if there are documents left)
    if clean_final:
        write_clean_shard(clean_final, output_path)

    return {
        "total": total,
        "removed_blocklist": removed_p0,
        "removed_profanity": removed_p1,
        "removed_toxicity": removed_p2,
        "kept": len(clean_final),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def detect_device() -> str:
    """Detect best available device for detoxify."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Audit corpus for profanity and toxicity")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing raw Parquet shards",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for clean (filtered) Parquet shards",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for detoxify model (auto-detected if not set)",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    device = args.device or detect_device()
    print(f"Using device: {device}")

    progress = load_progress(output_dir)
    completed = set(progress["completed_shards"])
    stats = progress["stats"]

    # Enumerate input shards
    shard_files = sorted(
        f for f in os.listdir(input_dir) if f.endswith(".parquet")
    )
    print(f"\nFound {len(shard_files)} shards in {input_dir}")
    print(f"Already completed: {len(completed)}")

    # Open audit logs
    blocklist_log_path = os.path.join(output_dir, "audit_log_blocklist.jsonl")
    profanity_log_path = os.path.join(output_dir, "audit_log_profanity.jsonl")
    toxicity_log_path = os.path.join(output_dir, "audit_log_toxicity.jsonl")

    with open(blocklist_log_path, "a") as blocklist_log, \
         open(profanity_log_path, "a") as profanity_log, \
         open(toxicity_log_path, "a") as toxicity_log:

        for shard_file in tqdm(shard_files, desc="Auditing shards", unit=" shards"):
            if shard_file in completed:
                continue

            shard_path = os.path.join(input_dir, shard_file)
            output_path = os.path.join(output_dir, shard_file)

            print(f"\n  Processing {shard_file}...")
            shard_stats = process_shard(
                shard_path, output_path, blocklist_log, profanity_log, toxicity_log, device
            )

            # Update running stats
            stats["total_docs"] += shard_stats["total"]
            stats["removed_blocklist"] += shard_stats["removed_blocklist"]
            stats["removed_profanity"] += shard_stats["removed_profanity"]
            stats["removed_toxicity"] += shard_stats["removed_toxicity"]
            stats["kept"] += shard_stats["kept"]

            print(f"    Total: {shard_stats['total']:,}")
            print(f"    Removed (blocklist): {shard_stats['removed_blocklist']:,}")
            print(f"    Removed (profanity): {shard_stats['removed_profanity']:,}")
            print(f"    Removed (toxicity):  {shard_stats['removed_toxicity']:,}")
            print(f"    Kept: {shard_stats['kept']:,}")

            completed.add(shard_file)
            progress["completed_shards"] = list(completed)
            progress["stats"] = stats
            save_progress(output_dir, progress)

    # Print final summary
    print("\n" + "=" * 60)
    print("AUDIT COMPLETE")
    print("=" * 60)
    print(f"Total documents processed:    {stats['total_docs']:,}")
    print(f"Removed by blocklist:         {stats['removed_blocklist']:,}")
    print(f"Removed by profanity-check:   {stats['removed_profanity']:,}")
    print(f"Removed by detoxify:          {stats['removed_toxicity']:,}")
    total_removed = stats["removed_blocklist"] + stats["removed_profanity"] + stats["removed_toxicity"]
    print(f"Total removed:                {total_removed:,} ({100*total_removed/max(stats['total_docs'],1):.2f}%)")
    print(f"Documents kept:               {stats['kept']:,}")
    print(f"\nClean shards written to: {output_dir}")
    print(f"Blocklist audit log: {blocklist_log_path}")
    print(f"Profanity audit log: {profanity_log_path}")
    print(f"Toxicity audit log:  {toxicity_log_path}")
    print(f"\nNext step: python -m kidschat.data.corpus_stats --input-dir {output_dir}")


if __name__ == "__main__":
    main()
