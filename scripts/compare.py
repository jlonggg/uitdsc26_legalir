#!/usr/bin/env python3
"""compare.py <zip> <reference zip> — sha256 of both, and how many questions have the identical answer SET / top-1.
A bit-exact rebuild prints `IDENTICAL`; a re-run on other GPUs is judged by the set-identical share (fp16 noise flips a few %)."""
import hashlib, json, sys, zipfile

def load(z): return json.loads(zipfile.ZipFile(z).read('submission.json'))
a, b = sys.argv[1:3]
ha, hb = (hashlib.sha256(open(z, 'rb').read()).hexdigest() for z in (a, b))
A, B = load(a), load(b)
assert set(A) == set(B), f'question sets differ: {len(set(A) ^ set(B))}'
same = sum(set(A[q]['answer']) == set(B[q]['answer']) for q in A); top1 = sum(A[q]['answer'][:1] == B[q]['answer'][:1] for q in A)
print(f'{a}\n  sha256 {ha}\n{b}\n  sha256 {hb}')
print('IDENTICAL' if ha == hb else f'set-identical {same}/{len(A)} ({same/len(A):.1%}), top-1 identical {top1}/{len(A)}')
