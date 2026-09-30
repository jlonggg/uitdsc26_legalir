#!/usr/bin/env python3
"""K-chunk evidence per (question, pool document), label-blind and symmetric train/inference.
Evidence = top-(K//2) wordseg-BM25 chunks inside the doc (already in the pool rows) + top-(K//2) Vietnamese_Embedding_v2
chunks inside the doc (exact cosine against the corpus chunk vectors built by `prepare`), de-duplicated, padded with the
doc's first chunks. Output rows share the pass-1 format: {qid, docs:[{doc_id, rank, chunks}]} over the WHOLE pool.
usage: evidence.py <questions.json> <pool.jsonl> <extra.jsonl,extra2.jsonl> <out.jsonl> [--k 4]"""
import argparse, json, os, sys, time
import numpy as np
from pathlib import Path
from mil_common import pick


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('questions'); ap.add_argument('pool'); ap.add_argument('extras'); ap.add_argument('out')
    ap.add_argument('--k', type=int, default=4); ap.add_argument('--device', default='cuda'); a = ap.parse_args()
    corpus = Path(os.environ['CLEAN_CORPUS']); half = a.k // 2
    rows = {}
    for line in open(a.pool, encoding='utf-8'):
        r = json.loads(line); rows[str(r['qid'])] = [dict(doc_id=d['doc_id'], rank=i + 1, chunks=d['chunks'][:half]) for i, d in enumerate(r['docs'][:60])]
    for p in a.extras.split(','):
        for line in open(p, encoding='utf-8'):
            r = json.loads(line); q = str(r['qid']); seen = {d['doc_id'] for d in rows[q]}
            for d in r['docs']:
                if d['doc_id'] not in seen: seen.add(d['doc_id']); rows[q].append(dict(doc_id=d['doc_id'], rank=d['rank'], chunks=d['chunks'][:half]))
    qs = json.load(open(a.questions, encoding='utf-8'))
    # chunk rows of every doc in corpus order (= row order of the vector file written by dense.py encode)
    doc_rows, chunk_ids = {}, []
    for i, line in enumerate(open(corpus / 'chunks.jsonl', encoding='utf-8')):
        c = json.loads(line); chunk_ids.append(c['chunk_id']); doc_rows.setdefault(c['doc_id'], []).append(i)
    doc_rows = {d: np.array(v) for d, v in doc_rows.items()}
    vec = np.load(corpus / 'Vietnamese_Embedding_v2.npy', mmap_mode='r')
    sys.path.insert(0, os.path.dirname(__file__)); from dense import Encoder
    enc = Encoder('Vietnamese_Embedding_v2', a.device); t0 = time.time()
    with open(a.out, 'w', encoding='utf-8') as out:
        for n, q in enumerate(sorted(rows), 1):
            qv = enc.encode([qs[q]['question']], query=True)[0]
            for d in rows[q]:
                ix = doc_rows.get(d['doc_id'])
                if ix is None: continue  # empty document: no chunks at all
                s = vec[ix] @ qv; order = ix[np.lexsort((ix, -s))]  # deterministic ties by corpus order
                d['chunks'] = pick(d['chunks'], [chunk_ids[j] for j in order[:half]], [chunk_ids[j] for j in ix], a.k)
            out.write(json.dumps({'qid': q, 'docs': [d for d in rows[q] if d['chunks']]}, ensure_ascii=False) + '\n')
            if n % 500 == 0 or n == len(rows): print(f'{n}/{len(rows)} {n/(time.time()-t0)*60:.0f}/min', flush=True)


if __name__ == '__main__':
    main()
