#!/usr/bin/env python3
"""Lexical-alignment feature channel: phrase-level alignment of the question with the K evidence chunks of every pool document.
Per (question, doc) = max over its chunks of [bigram precision, trigram precision, longest common syllable run (cap 10)/10,
share of question positions covered by shared trigrams] + 2 in-query comparatives (bigram/trigram precision minus the query max).
Label-blind, symmetric train/inference; chunk text from CLEAN_CORPUS/chunks.jsonl. Output rows {qid, docs:[{doc_id, lex:[6]}]}.
usage: lexalign.py <questions.json> <evidence.jsonl> <out.jsonl>"""
import json, os, re, sys, time, unicodedata
import numpy as np
from multiprocessing import Pool
TEXT = {}


def toks(s): return re.findall(r'\w+', unicodedata.normalize('NFC', s).casefold())
def ngr(t, n): return set(zip(*[t[i:] for i in range(n)]))


def align(question, chunk_lists):
    """One row of 6 features per chunk list (= per doc)."""
    qt = toks(question); q2 = ngr(qt, 2); q3 = ngr(qt, 3); L = min(len(qt), 10); qn = {n: ngr(qt, n) for n in range(4, L + 1)}
    out = []
    for chunks in chunk_lists:
        best = np.zeros(4)
        for c in chunks:
            ct = toks(TEXT.get(c, ''))
            if not ct: continue
            c2 = ngr(ct, 2); c3 = ngr(ct, 3); m3 = q3 & c3
            f = [len(q2 & c2) / max(len(q2), 1), len(m3) / max(len(q3), 1), 0.0, 0.0]
            if m3:
                cov = set()
                for i in range(len(qt) - 2):
                    if tuple(qt[i:i + 3]) in m3: cov.update((i, i + 1, i + 2))
                f[3] = len(cov) / len(qt); f[2] = 0.3
                for n in range(4, L + 1):
                    if qn[n] & ngr(ct, n): f[2] = n / 10
                    else: break
            elif q2 & c2: f[2] = 0.2
            best = np.maximum(best, f)
        out.append(best)
    a = np.array(out) if out else np.zeros((0, 4)); rel = a[:, :2] - a[:, :2].max(0) if len(a) else a[:, :2]
    return np.hstack([a, rel]).round(5).tolist()


def _row(args):
    qid, question, docs = args
    return {'qid': qid, 'docs': [{'doc_id': d['doc_id'], 'lex': v} for d, v in zip(docs, align(question, [d['chunks'] for d in docs]))]}


def main():
    qpath, evpath, out = sys.argv[1:4]; qs = json.load(open(qpath, encoding='utf-8')); t0 = time.time()
    rows = [json.loads(l) for l in open(evpath, encoding='utf-8') if l.strip()]
    need = {c for r in rows for d in r['docs'] for c in d['chunks']}
    for line in open(os.path.join(os.environ['CLEAN_CORPUS'], 'chunks.jsonl'), encoding='utf-8'):
        c = json.loads(line)
        if c['chunk_id'] in need: TEXT[c['chunk_id']] = c['article'] + ' ' + c['text']
    assert need <= set(TEXT), 'Missing chunk text'
    jobs = [(str(r['qid']), qs[str(r['qid'])]['question'], r['docs']) for r in rows]
    with Pool(int(os.environ.get('CLEAN_THREADS', '8'))) as p, open(out, 'w', encoding='utf-8') as o:
        for row in p.imap(_row, jobs, chunksize=20): o.write(json.dumps(row) + '\n')
    print(f'lexalign {len(rows)} questions {round(time.time() - t0)} s', flush=True)


if __name__ == '__main__':
    main()
