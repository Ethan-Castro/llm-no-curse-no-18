"""
build_corpus.py — Downloads and assembles a kid-safe training corpus.

Sources:
  1. Cosmopedia (HuggingFace) — synthetic educational text, filtered by audience
  2. Project Gutenberg — children's fiction and literature (public domain)

Outputs nanochat-compatible Parquet shards with:
  - 'text' column (string)
  - 1024-row row groups
  - zstd level 3 compression
  - ~250M characters per shard

Usage:
    python -m kidschat.data.build_corpus --output-dir ./kidschat_data --target-tokens 3000000000
"""

import argparse
import json
import os
import random
import re
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests
from datasets import load_dataset
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Cosmopedia subsets and their kid-safe audience filters.
# None means "take all rows" (the entire subset is kid-safe).
COSMOPEDIA_SUBSETS = {
    "stories": ["young_children"],
    "auto_math_text": ["grade_school_students"],
    "openstax": ["young_children", "middle_school_students"],
    "stanford": ["young_children"],
    "khanacademy": ["young_children", "grade_school_students", "middle_school_students"],
    "wikihow": ["young_children", "grade_school_students", "middle_school_students"],
}

# Target corpus composition (by token count)
COSMOPEDIA_FRACTION = 0.70  # 70% cosmopedia
GUTENBERG_FRACTION = 0.30   # 30% gutenberg

# Shard settings — match nanochat's expected format
SHARD_CHAR_TARGET = 250_000_000  # ~250M chars per shard
ROW_GROUP_SIZE = 1024

# Gutenberg API
GUTENDEX_BASE = "https://gutendex.com/books"

# Characters per token estimate (for progress tracking)
CHARS_PER_TOKEN = 4.8

# ---------------------------------------------------------------------------
# Progress tracking for resumability
# ---------------------------------------------------------------------------

def load_progress(output_dir: str) -> dict:
    path = os.path.join(output_dir, "build_progress.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {
        "cosmopedia_completed_subsets": [],
        "cosmopedia_tokens_collected": 0,
        "gutenberg_downloaded_ids": [],
        "shards_written": 0,
        "phase": "cosmopedia",  # cosmopedia -> gutenberg -> merge
    }


def save_progress(output_dir: str, progress: dict):
    path = os.path.join(output_dir, "build_progress.json")
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Phase 1: Cosmopedia streaming download
# ---------------------------------------------------------------------------

def download_cosmopedia(output_dir: str, target_tokens: int, progress: dict) -> list[str]:
    """Stream kid-safe Cosmopedia subsets and collect texts."""
    raw_dir = os.path.join(output_dir, "raw_cosmopedia")
    os.makedirs(raw_dir, exist_ok=True)

    collected_texts = []
    tokens_so_far = progress.get("cosmopedia_tokens_collected", 0)
    cosmopedia_target = int(target_tokens * COSMOPEDIA_FRACTION)

    # Load any previously saved raw texts
    existing_raw = os.path.join(raw_dir, "texts.jsonl")
    if os.path.exists(existing_raw):
        print(f"Resuming from {existing_raw}...")
        with open(existing_raw) as f:
            for line in f:
                collected_texts.append(json.loads(line)["text"])

    completed = set(progress.get("cosmopedia_completed_subsets", []))

    for subset_name, audiences in COSMOPEDIA_SUBSETS.items():
        if subset_name in completed:
            print(f"  Skipping {subset_name} (already completed)")
            continue

        if tokens_so_far >= cosmopedia_target:
            print(f"  Reached token target ({tokens_so_far:,} / {cosmopedia_target:,}), stopping.")
            break

        print(f"\n  Streaming subset: {subset_name}")
        if audiences:
            print(f"    Filtering by audience: {audiences}")

        try:
            ds = load_dataset(
                "HuggingFaceTB/cosmopedia",
                subset_name,
                split="train",
                streaming=True,
            )
        except Exception as e:
            print(f"    ERROR loading {subset_name}: {e}")
            continue

        subset_count = 0
        subset_tokens = 0

        with open(existing_raw, "a") as out_f:
            for row in tqdm(ds, desc=f"    {subset_name}", unit=" docs"):
                # Filter by audience if specified (case-insensitive, null-safe)
                if audiences:
                    audience = (row.get("audience") or "").lower().strip()
                    if audience not in [a.lower() for a in audiences]:
                        continue

                text = row.get("text", "")
                if not text or len(text.strip()) < 50:
                    continue

                # Track tokens using the metadata column if available
                tok_count = row.get("text_token_length", int(len(text) / CHARS_PER_TOKEN))
                tokens_so_far += tok_count
                subset_tokens += tok_count
                subset_count += 1

                collected_texts.append(text)
                out_f.write(json.dumps({"text": text}) + "\n")

                if tokens_so_far >= cosmopedia_target:
                    break

        print(f"    Collected {subset_count:,} docs ({subset_tokens:,} tokens) from {subset_name}")
        completed.add(subset_name)
        progress["cosmopedia_completed_subsets"] = list(completed)
        progress["cosmopedia_tokens_collected"] = tokens_so_far
        save_progress(output_dir, progress)

    print(f"\nCosmopedia total: {len(collected_texts):,} docs, ~{tokens_so_far:,} tokens")
    return collected_texts


# ---------------------------------------------------------------------------
# Phase 2: Project Gutenberg children's books
# ---------------------------------------------------------------------------

def fetch_gutenberg_book_ids() -> list[int]:
    """Fetch children's book IDs from Gutendex API."""
    book_ids = set()
    url = f"{GUTENDEX_BASE}?topic=children&languages=en"

    print("  Fetching book list from Gutendex...")
    while url:
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"    Error fetching {url}: {e}")
            break

        for book in data.get("results", []):
            book_ids.add(book["id"])

        url = data.get("next")
        time.sleep(0.5)  # be polite to the API

    print(f"  Found {len(book_ids)} children's books")
    return sorted(book_ids)


def download_gutenberg_text(book_id: int, cache_dir: str) -> str | None:
    """Download a single Gutenberg book as plain text."""
    cache_path = os.path.join(cache_dir, f"pg{book_id}.txt")

    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return f.read()

    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return None
        raw = resp.text
    except Exception:
        return None

    # Strip Project Gutenberg header/footer
    text = strip_gutenberg_boilerplate(raw)

    if text and len(text.strip()) > 200:
        with open(cache_path, "w") as f:
            f.write(text)
        return text
    return None


def strip_gutenberg_boilerplate(raw: str) -> str:
    """Remove Project Gutenberg header and footer from raw text."""
    # Find start marker
    start_pattern = r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    start_match = re.search(start_pattern, raw, re.IGNORECASE)
    if start_match:
        raw = raw[start_match.end():]

    # Find end marker
    end_pattern = r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*"
    end_match = re.search(end_pattern, raw, re.IGNORECASE)
    if end_match:
        raw = raw[:end_match.start()]

    # Clean up whitespace
    raw = raw.strip()
    # Normalize excessive newlines
    raw = re.sub(r"\n{4,}", "\n\n\n", raw)

    return raw


def download_gutenberg(output_dir: str, target_tokens: int, progress: dict) -> list[str]:
    """Download children's books from Project Gutenberg."""
    cache_dir = os.path.join(output_dir, "gutenberg_cache")
    os.makedirs(cache_dir, exist_ok=True)

    gutenberg_target = int(target_tokens * GUTENBERG_FRACTION)
    downloaded_ids = set(progress.get("gutenberg_downloaded_ids", []))

    book_ids = fetch_gutenberg_book_ids()
    texts = []
    tokens_collected = 0

    # Load previously cached texts
    for bid in downloaded_ids:
        cache_path = os.path.join(cache_dir, f"pg{bid}.txt")
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                text = f.read()
                texts.append(text)
                tokens_collected += int(len(text) / CHARS_PER_TOKEN)

    print(f"  Resumed with {len(texts)} cached books (~{tokens_collected:,} tokens)")

    for book_id in tqdm(book_ids, desc="  Downloading books", unit=" books"):
        if book_id in downloaded_ids:
            continue
        if tokens_collected >= gutenberg_target:
            break

        text = download_gutenberg_text(book_id, cache_dir)
        if text:
            texts.append(text)
            tokens_collected += int(len(text) / CHARS_PER_TOKEN)
            downloaded_ids.add(book_id)
            time.sleep(1.0)  # be polite to gutenberg.org

    progress["gutenberg_downloaded_ids"] = list(downloaded_ids)
    save_progress(output_dir, progress)

    print(f"\nGutenberg total: {len(texts):,} books, ~{tokens_collected:,} tokens")
    return texts


# ---------------------------------------------------------------------------
# Phase 3: Merge, shuffle, and shard into Parquet
# ---------------------------------------------------------------------------

def write_parquet_shard(texts: list[str], shard_path: str):
    """Write a list of texts as a nanochat-compatible Parquet shard."""
    # Pad to a multiple of ROW_GROUP_SIZE so every row group is exactly 1024
    remainder = len(texts) % ROW_GROUP_SIZE
    if remainder != 0:
        # Duplicate random texts to fill the last row group
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


def merge_and_shard(
    cosmopedia_texts: list[str],
    gutenberg_texts: list[str],
    output_dir: str,
):
    """Merge all texts, shuffle, and write nanochat-compatible Parquet shards."""
    shard_dir = os.path.join(output_dir, "shards")
    os.makedirs(shard_dir, exist_ok=True)

    all_texts = cosmopedia_texts + gutenberg_texts
    print(f"\nTotal documents before sharding: {len(all_texts):,}")

    # Seeded shuffle for reproducibility
    random.seed(42)
    random.shuffle(all_texts)

    # Split into shards based on character count
    shard_idx = 0
    current_shard = []
    current_chars = 0

    for text in tqdm(all_texts, desc="Sharding", unit=" docs"):
        current_shard.append(text)
        current_chars += len(text)

        if current_chars >= SHARD_CHAR_TARGET:
            shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}.parquet")
            write_parquet_shard(current_shard, shard_path)
            print(f"  Wrote {shard_path} ({len(current_shard):,} docs, {current_chars:,} chars)")
            shard_idx += 1
            current_shard = []
            current_chars = 0

    # Write remaining documents as the final shard (this becomes the validation shard)
    if current_shard:
        shard_path = os.path.join(shard_dir, f"shard_{shard_idx:05d}.parquet")
        write_parquet_shard(current_shard, shard_path)
        print(f"  Wrote {shard_path} ({len(current_shard):,} docs, {current_chars:,} chars) [validation shard]")

    total_shards = shard_idx + 1
    print(f"\nDone! Wrote {total_shards} shards to {shard_dir}/")
    print(f"  Training shards: shard_00000 through shard_{total_shards - 2:05d}")
    print(f"  Validation shard: shard_{total_shards - 1:05d}")

    return shard_dir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build kid-safe training corpus")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./kidschat_data",
        help="Directory to store downloaded data and output shards",
    )
    parser.add_argument(
        "--target-tokens",
        type=int,
        default=3_000_000_000,
        help="Target total token count (default: 3B)",
    )
    args = parser.parse_args()

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    progress = load_progress(output_dir)

    print("=" * 60)
    print("KidsChat Corpus Builder")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Target tokens:    {args.target_tokens:,}")
    print(f"  Cosmopedia:     ~{int(args.target_tokens * COSMOPEDIA_FRACTION):,} tokens (70%)")
    print(f"  Gutenberg:      ~{int(args.target_tokens * GUTENBERG_FRACTION):,} tokens (30%)")
    print()

    # Phase 1: Cosmopedia
    print("Phase 1: Downloading Cosmopedia (kid-safe subsets)...")
    print("-" * 60)
    cosmopedia_texts = download_cosmopedia(output_dir, args.target_tokens, progress)

    # Phase 2: Gutenberg
    print("\nPhase 2: Downloading Project Gutenberg children's books...")
    print("-" * 60)
    gutenberg_texts = download_gutenberg(output_dir, args.target_tokens, progress)

    # Phase 3: Merge and shard
    print("\nPhase 3: Merging, shuffling, and writing Parquet shards...")
    print("-" * 60)
    shard_dir = merge_and_shard(cosmopedia_texts, gutenberg_texts, output_dir)

    # Update progress
    progress["phase"] = "complete"
    save_progress(output_dir, progress)

    print("\n" + "=" * 60)
    print("Corpus build complete!")
    print(f"Shards are in: {shard_dir}")
    print("Next step: python -m kidschat.data.audit_corpus --input-dir", shard_dir)
    print("=" * 60)


if __name__ == "__main__":
    main()
