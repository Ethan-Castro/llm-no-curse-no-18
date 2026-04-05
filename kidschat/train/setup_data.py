"""
setup_data.py — Validate Parquet shards and configure nanochat's data path.

Verifies that the audited corpus shards are in nanochat-compatible format,
then creates a symlink (or prints instructions) so nanochat can find the data.

Usage:
    python -m kidschat.train.setup_data --shard-dir ./kidschat_data/clean_shards
"""

import argparse
import os
import sys

import pyarrow.parquet as pq


# ---------------------------------------------------------------------------
# nanochat data directory
# ---------------------------------------------------------------------------

def get_nanochat_data_dir() -> str:
    """Return the path nanochat expects for base pretraining data."""
    base = os.path.join(os.path.expanduser("~"), ".cache", "nanochat")
    return os.path.join(base, "base_data_climbmix")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_shard(shard_path: str) -> list[str]:
    """Validate a single Parquet shard. Returns list of issues (empty = OK)."""
    issues = []

    try:
        pf = pq.ParquetFile(shard_path)
    except Exception as e:
        return [f"Cannot open file: {e}"]

    schema = pf.schema_arrow

    # Check 'text' column exists
    if "text" not in schema.names:
        issues.append(f"Missing 'text' column. Found columns: {schema.names}")

    # Check row group size
    metadata = pf.metadata
    for rg_idx in range(metadata.num_row_groups):
        rg = metadata.row_group(rg_idx)
        if rg.num_rows != 1024:
            # Last row group can be smaller if not padded, but we pad in build
            if rg_idx < metadata.num_row_groups - 1:
                issues.append(
                    f"Row group {rg_idx} has {rg.num_rows} rows (expected 1024)"
                )

    # Check compression
    if metadata.num_row_groups > 0:
        col_meta = metadata.row_group(0).column(0)
        compression = str(col_meta.compression).lower()
        if "zstd" not in compression:
            issues.append(f"Compression is '{compression}', expected 'zstd'")

    # Quick content check
    table = pq.read_table(shard_path, columns=["text"])
    texts = table.column("text").to_pylist()
    empty = sum(1 for t in texts if not t or len(t.strip()) == 0)
    if empty > 0:
        issues.append(f"{empty} empty documents found")

    return issues


def validate_all_shards(shard_dir: str) -> bool:
    """Validate all shards in a directory. Returns True if all pass."""
    shard_files = sorted(f for f in os.listdir(shard_dir) if f.endswith(".parquet"))

    if not shard_files:
        print("ERROR: No .parquet files found!")
        return False

    print(f"Validating {len(shard_files)} shards...")
    all_ok = True

    for sf in shard_files:
        path = os.path.join(shard_dir, sf)
        issues = validate_shard(path)

        if issues:
            print(f"  FAIL  {sf}")
            for issue in issues:
                print(f"        - {issue}")
            all_ok = False
        else:
            pf = pq.ParquetFile(path)
            rgs = pf.metadata.num_row_groups
            rows = pf.metadata.num_rows
            print(f"  OK    {sf}  ({rows:,} rows, {rgs} row groups)")

    if all_ok:
        print(f"\nAll {len(shard_files)} shards pass validation!")
        print(f"Training shards: {shard_files[0]} through {shard_files[-2]}")
        print(f"Validation shard: {shard_files[-1]} (last alphabetically)")
    else:
        print("\nSome shards have issues. Fix them before training.")

    return all_ok


# ---------------------------------------------------------------------------
# Symlink setup
# ---------------------------------------------------------------------------

def setup_symlink(shard_dir: str, force: bool = False) -> bool:
    """Create symlink from nanochat's expected data path to our shard directory."""
    target = get_nanochat_data_dir()
    shard_dir = os.path.abspath(shard_dir)

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(target), exist_ok=True)

    if os.path.exists(target):
        if os.path.islink(target):
            current = os.readlink(target)
            if current == shard_dir:
                print(f"Symlink already points to correct directory:")
                print(f"  {target} -> {shard_dir}")
                return True
            elif force:
                os.unlink(target)
                print(f"Removed existing symlink: {target} -> {current}")
            else:
                print(f"WARNING: {target} already exists and points elsewhere:")
                print(f"  Current: {current}")
                print(f"  Wanted:  {shard_dir}")
                print(f"\nRun with --force to overwrite, or manually remove it.")
                return False
        else:
            print(f"WARNING: {target} exists and is not a symlink.")
            print(f"Move or remove it before proceeding.")
            return False

    os.symlink(shard_dir, target)
    print(f"Created symlink:")
    print(f"  {target} -> {shard_dir}")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Validate shards and setup nanochat data path")
    parser.add_argument(
        "--shard-dir",
        type=str,
        required=True,
        help="Directory containing audited Parquet shards",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing symlink if it points elsewhere",
    )
    parser.add_argument(
        "--skip-symlink",
        action="store_true",
        help="Only validate, don't create symlink",
    )
    args = parser.parse_args()

    shard_dir = os.path.abspath(args.shard_dir)

    if not os.path.isdir(shard_dir):
        print(f"ERROR: {shard_dir} is not a directory")
        sys.exit(1)

    print("=" * 60)
    print("KidsChat Data Setup for nanochat")
    print("=" * 60)
    print(f"Shard directory: {shard_dir}")
    print()

    # Step 1: Validate
    if not validate_all_shards(shard_dir):
        sys.exit(1)

    # Step 2: Symlink
    if not args.skip_symlink:
        print()
        if not setup_symlink(shard_dir, force=args.force):
            sys.exit(1)

    # Step 3: Print training instructions
    print("\n" + "=" * 60)
    print("READY TO TRAIN")
    print("=" * 60)
    print("""
Next steps:

1. Clone nanochat (if not already):
   git clone https://github.com/karpathy/nanochat.git
   cd nanochat
   pip install -r requirements.txt

2. Train tokenizer on your corpus:
   python -m scripts.tok_train

3. Start pretraining (~300M params):
   torchrun --nproc_per_node=4 -m scripts.base_train --depth=20

   Or for ~700M params:
   torchrun --nproc_per_node=4 -m scripts.base_train --depth=30

4. After pretraining, run SFT with tutoring dialogues:
   python -m scripts.chat_sft \\
     --train-files /path/to/kidschat_dialogues.jsonl \\
     --depth=20

GPU rental options (for ~$85-135 total):
  - Vast.ai: https://vast.ai (4x RTX 4090 spot ~$1.20/hr)
  - RunPod:  https://runpod.io (4x RTX 4090 spot ~$1.50/hr)
""")


if __name__ == "__main__":
    main()
