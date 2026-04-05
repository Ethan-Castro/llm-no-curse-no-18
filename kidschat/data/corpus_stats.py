"""
corpus_stats.py — Compute and display statistics about the assembled corpus.

Reads audited (or raw) Parquet shards and reports token counts,
document counts, shard sizes, and sample documents.

Usage:
    python -m kidschat.data.corpus_stats --input-dir ./kidschat_data/clean_shards
"""

import argparse
import json
import os
import random

import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SAMPLE_SIZE_FOR_TOKEN_ESTIMATE = 1000
CHARS_PER_TOKEN_FALLBACK = 4.8


# ---------------------------------------------------------------------------
# Stats computation
# ---------------------------------------------------------------------------

def estimate_tokens(texts: list[str], sample_size: int = SAMPLE_SIZE_FOR_TOKEN_ESTIMATE) -> tuple[int, float]:
    """Estimate total token count by sampling documents and tokenizing."""
    enc = tiktoken.get_encoding("cl100k_base")

    sample = random.sample(texts, min(sample_size, len(texts)))
    total_chars_in_sample = sum(len(t) for t in sample)
    total_tokens_in_sample = sum(len(enc.encode(t)) for t in sample)

    if total_chars_in_sample == 0:
        return 0, CHARS_PER_TOKEN_FALLBACK

    chars_per_token = total_chars_in_sample / total_tokens_in_sample
    total_chars = sum(len(t) for t in texts)
    estimated_tokens = int(total_chars / chars_per_token)

    return estimated_tokens, chars_per_token


def length_histogram(texts: list[str], bins: int = 10) -> list[tuple[str, int]]:
    """Compute a simple histogram of document lengths (in characters)."""
    lengths = [len(t) for t in texts]
    if not lengths:
        return []

    min_len = min(lengths)
    max_len = max(lengths)
    bin_width = max((max_len - min_len) // bins, 1)

    histogram = []
    for i in range(bins):
        lo = min_len + i * bin_width
        hi = lo + bin_width if i < bins - 1 else max_len + 1
        count = sum(1 for l in lengths if lo <= l < hi)
        label = f"{lo:>8,} - {hi - 1:>8,}"
        histogram.append((label, count))

    return histogram


def main():
    parser = argparse.ArgumentParser(description="Compute corpus statistics")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing Parquet shards",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of sample documents to print",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to write stats as JSON",
    )
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    shard_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))

    if not shard_files:
        print(f"No .parquet files found in {input_dir}")
        return

    # Read all texts
    print(f"Reading {len(shard_files)} shards from {input_dir}...")
    all_texts = []
    shard_info = []

    for sf in tqdm(shard_files, desc="Reading shards", unit=" shards"):
        path = os.path.join(input_dir, sf)
        table = pq.read_table(path, columns=["text"])
        texts = table.column("text").to_pylist()
        file_size = os.path.getsize(path)
        pf = pq.ParquetFile(path)

        shard_info.append({
            "name": sf,
            "docs": len(texts),
            "chars": sum(len(t) for t in texts),
            "size_mb": file_size / (1024 * 1024),
            "row_groups": pf.metadata.num_row_groups,
        })
        all_texts.extend(texts)

    total_docs = len(all_texts)
    total_chars = sum(len(t) for t in all_texts)

    print("\nEstimating token count (sampling and tokenizing)...")
    estimated_tokens, chars_per_token = estimate_tokens(all_texts)

    # Print report
    print("\n" + "=" * 60)
    print("CORPUS STATISTICS")
    print("=" * 60)
    print(f"Total documents:      {total_docs:,}")
    print(f"Total characters:     {total_chars:,}")
    print(f"Estimated tokens:     {estimated_tokens:,}")
    print(f"Chars per token:      {chars_per_token:.2f}")
    print(f"Number of shards:     {len(shard_files)}")

    avg_doc_len = total_chars / max(total_docs, 1)
    print(f"Avg doc length:       {avg_doc_len:,.0f} chars")
    print(f"Avg tokens per doc:   {avg_doc_len / chars_per_token:,.0f}")

    # Shard details
    print(f"\n{'Shard':<25} {'Docs':>8} {'Chars':>14} {'Size (MB)':>10} {'RowGroups':>10}")
    print("-" * 70)
    for si in shard_info:
        print(f"{si['name']:<25} {si['docs']:>8,} {si['chars']:>14,} {si['size_mb']:>10.1f} {si['row_groups']:>10}")

    # Length histogram
    print("\nDocument length distribution (characters):")
    print("-" * 40)
    hist = length_histogram(all_texts)
    for label, count in hist:
        bar = "#" * min(count * 50 // max(total_docs, 1), 50)
        print(f"  {label}  {count:>8,}  {bar}")

    # Sample documents
    print(f"\n{args.num_samples} random sample documents:")
    print("-" * 60)
    random.seed(123)
    samples = random.sample(all_texts, min(args.num_samples, total_docs))
    for i, text in enumerate(samples, 1):
        preview = text[:300].replace("\n", " ")
        print(f"\n  [{i}] ({len(text):,} chars)")
        print(f"      {preview}...")

    # Validation shard info
    print(f"\nValidation shard (last alphabetically): {shard_files[-1]}")
    print(f"  Docs: {shard_info[-1]['docs']:,}, Chars: {shard_info[-1]['chars']:,}")

    # Write JSON report if requested
    if args.output_json:
        report = {
            "total_docs": total_docs,
            "total_chars": total_chars,
            "estimated_tokens": estimated_tokens,
            "chars_per_token": chars_per_token,
            "num_shards": len(shard_files),
            "avg_doc_length_chars": avg_doc_len,
            "shards": shard_info,
        }
        with open(args.output_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report written to: {args.output_json}")


if __name__ == "__main__":
    main()
