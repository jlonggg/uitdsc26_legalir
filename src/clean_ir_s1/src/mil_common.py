"""Pure helpers of the MIL scorers (S1/S2: a LoRA cross-encoder that scores a document from its K evidence chunks).
No torch/ES imports, so they are unit-testable on any machine."""
import collections, hashlib, re, unicodedata


def normalize(q): return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFC', q).lower(), flags=re.UNICODE))
def fold(q): return int(hashlib.sha256(normalize(q).encode()).hexdigest()[:8], 16) % 5


def pick(seed_chunks, dense_order, corpus_order, k):
    """Evidence list for one (question, doc): BM25 chunks already in the pool row, then the doc's best dense chunks,
    then the doc's first chunks; de-duplicated, at most k."""
    ev = list(seed_chunks)
    for c in list(dense_order) + list(corpus_order):
        if len(ev) >= k: break
        if c not in ev: ev.append(c)
    return ev[:k]


def build_groups(train, evidence, ranking, outer, n_hard, n_rand, rng):
    """One group per (outer-train question, gold doc present in the pool): [gold] + n_hard hardest non-gold docs by
    `ranking` + n_rand random other pool docs. Negatives never contain a gold of the question or of any normalized-duplicate
    question (annotation variance, not irrelevance). Returns (groups, skipped_questions)."""
    dup_gold = collections.defaultdict(set)
    for q, r in train.items():
        if fold(r['question']) != outer: dup_gold[normalize(r['question'])].update(map(str, r['answer']))
    groups, skipped = [], 0
    for q in sorted(train):
        r = train[q]
        if fold(r['question']) == outer or q not in evidence: continue
        gold = set(map(str, r['answer'])); banned = gold | dup_gold[normalize(r['question'])]
        docs = {d['doc_id']: d['chunks'] for d in evidence[q]}
        hard = [d for d in ranking.get(q, []) if d in docs and d not in banned][:n_hard]
        rest = [d for d in sorted(docs) if d not in banned and d not in hard]
        present = sorted(gold & set(docs))
        if not present: skipped += 1; continue
        for g in present:
            rand = rng.sample(rest, min(n_rand, len(rest)))
            if len(hard) + len(rand) < 2: skipped += 1; continue
            order = [g] + hard + rand
            groups.append({'qid': q, 'question': r['question'], 'docs': order, 'chunks': [docs[d] for d in order]})
    return groups, skipped
