"""Secondary content filter — removes docs with kid-inappropriate themes."""
import pyarrow.parquet as pq, pyarrow as pa, os, random

d = '/Users/ethancastro/llm-no-curse-no-18/kidschat_data/clean_shards'
ROW_GROUP_SIZE = 1024
BAD = ['murder', 'suicide', 'self-harm', 'sexual violence', 'bank robber',
       'kill yourself', 'killed himself', 'killed herself',
       'rape', 'rapist', 'molest', 'pedophil', 'child abuse',
       'shoot guns', 'drug dealer', 'cocaine', 'heroin', 'methamphetamine',
       'kill her', 'kill him', 'torture', 'dismember']

total_removed = 0
for fname in sorted(os.listdir(d)):
    if not fname.endswith('.parquet'):
        continue
    path = os.path.join(d, fname)
    texts = pq.read_table(path, columns=['text']).column('text').to_pylist()
    clean = [t for t in texts if not any(b in t.lower() for b in BAD)]
    removed = len(texts) - len(clean)
    if removed == 0:
        continue
    remainder = len(clean) % ROW_GROUP_SIZE
    if remainder:
        clean += random.choices(clean, k=ROW_GROUP_SIZE - remainder)
    tbl = pa.Table.from_pydict({'text': clean})
    pq.write_table(tbl, path, row_group_size=ROW_GROUP_SIZE,
                   use_dictionary=False, compression='zstd', compression_level=3,
                   write_statistics=False)
    total_removed += removed
    print(f'  {fname}: removed {removed}')
print(f'\nTotal removed: {total_removed}')
