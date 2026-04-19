"""
audit_corpus.py — Fast corpus safety filtering with per-shard progress.

Strategy:
- Build two combined regex strings from the blocklist once at startup.
- Run regex matching over each shard with pyarrow.compute in C++ instead of
  Python loops over every document.
- Process shards sequentially with progress output after each shard, so the
  audit does not look frozen and avoids multiprocessing oversubscription.

Usage:
    python -m kidschat.data.audit_corpus --input-dir ./kidschat_data/shards --output-dir ./kidschat_data/clean_shards
"""

import argparse
import os
import random
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROW_GROUP_SIZE = 1024


def _build_fast_patterns() -> tuple[str | None, str | None]:
    from kidschat.data.blocklist import _ALL_TERMS, _WHOLE_WORD_TERMS, _build_leet_pattern

    whole_word_leet = []
    any_leet = []
    seen = set()

    for term in _ALL_TERMS:
        term_lower = term.lower()
        if term_lower in seen:
            continue
        seen.add(term_lower)

        leet = _build_leet_pattern(term_lower)
        if term_lower in _WHOLE_WORD_TERMS:
            whole_word_leet.append(leet)
        else:
            any_leet.append(leet)

    whole_pattern = None
    any_pattern = None
    if whole_word_leet:
        whole_pattern = r"\b(?:" + "|".join(whole_word_leet) + r")\b"
    if any_leet:
        any_pattern = r"(?:" + "|".join(any_leet) + r")"
    return whole_pattern, any_pattern


_WHOLE_PATTERN, _ANY_PATTERN = _build_fast_patterns()


def process_shard(input_path: str, output_path: str) -> dict:
    table = pq.read_table(input_path, columns=["text"])
    texts = table.column("text").to_pylist()
    total = len(texts)

    lowered = pc.utf8_lower(table["text"])
    if _WHOLE_PATTERN is not None:
        blocked_mask = pc.match_substring_regex(lowered, _WHOLE_PATTERN)
    else:
        blocked_mask = pc.equal(lowered, "__never__")
    if _ANY_PATTERN is not None:
        blocked_mask = pc.or_(blocked_mask, pc.match_substring_regex(lowered, _ANY_PATTERN))

    blocked = blocked_mask.to_pylist()
    clean = [text for text, is_blocked in zip(texts, blocked) if not is_blocked]
    kept_raw = len(clean)
    removed_blocklist = total - kept_raw

    if clean:
        remainder = kept_raw % ROW_GROUP_SIZE
        if remainder:
            clean += random.choices(clean, k=ROW_GROUP_SIZE - remainder)
        output_table = pa.Table.from_pydict({"text": clean})
    else:
        output_table = pa.Table.from_pydict({"text": []})
    written_rows = len(clean)

    tmp_output = output_path + ".tmp"
    pq.write_table(
        output_table,
        tmp_output,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )
    os.replace(tmp_output, output_path)

    return {
        "shard": os.path.basename(input_path),
        "total": total,
        "removed": removed_blocklist,
        "kept": kept_raw,
        "written": written_rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    shards = sorted(f for f in os.listdir(input_dir) if f.endswith(".parquet"))
    done = set(f for f in os.listdir(output_dir) if f.endswith(".parquet"))
    todo = [(os.path.join(input_dir, f), os.path.join(output_dir, f)) for f in shards if f not in done]

    if not todo:
        print("All shards already done.")
        return

    cpu_count = pa.cpu_count()
    print(f"Processing {len(todo)} shards with pyarrow on {cpu_count} CPUs...", flush=True)

    start_time = time.time()
    results = []
    for idx, (input_path, output_path) in enumerate(todo, start=1):
        shard_start = time.time()
        result = process_shard(input_path, output_path)
        results.append(result)
        shard_elapsed = time.time() - shard_start
        total_elapsed = time.time() - start_time
        print(
            f"[{idx}/{len(todo)}] {result['shard']}: "
            f"{result['total']:,} -> {result['kept']:,} kept "
            f"(removed {result['removed']:,}, wrote {result['written']:,}) in {shard_elapsed:.1f}s "
            f"| elapsed {total_elapsed:.1f}s",
            flush=True,
        )

    total_docs = sum(r["total"] for r in results)
    total_removed = sum(r["removed"] for r in results)
    total_kept = sum(r["kept"] for r in results)

    print(f"\n{'=' * 50}")
    print("AUDIT COMPLETE")
    print(f"{'=' * 50}")
    for result in results:
        print(
            f"  {result['shard']}: {result['total']:,} -> {result['kept']:,} kept "
            f"(removed {result['removed']:,}, wrote {result['written']:,})"
        )
    print(f"\nTotal: {total_docs:,} docs | Removed: {total_removed:,} | Kept: {total_kept:,}")
    print(f"Clean shards: {output_dir}")


if __name__ == "__main__":
    main()
