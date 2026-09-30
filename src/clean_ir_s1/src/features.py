"""Per-candidate selector features for one question (69 columns for the default `ce` scorer; optional channels
append columns, see selector_stage.WIDTH). Doc metadata and the reference graph are read from CLEAN_CORPUS,
per-run files (best_chunk_text.jsonl) from RB_OUT."""
import json, sys, os, re, hashlib, unicodedata, collections, pickle
import numpy as np
ROOT = os.environ['CLEAN_CORPUS']
O = os.environ['RB_OUT']  


def normalize(q):
    t = unicodedata.normalize('NFC', q).lower(); return ' '.join(re.findall(r'[^\W_]+', t, flags=re.UNICODE))
def fold(q): return int(hashlib.sha256(normalize(q).encode()).hexdigest()[:8], 16) % 5

def load_rows(paths):
    """Main pool rows; files whose name contains 'ce_x' are dense-only extras appended after the BM25 docs."""
    rows = {}; extras = {}
    for p in paths:
        b = os.path.basename(p)
        # Fail closed: the default branch below REPLACES rows[q] by assignment, so any non-pool file passed here would
        # silently overwrite (truncate) the 60-doc pool while the selector fit still runs.
        if not (b.startswith('ce_train') or b.startswith('ce_public') or 'ce2_' in b or 'ce_x' in b or 'mil' in b):
            raise SystemExit(f'load_rows: {b} is not a pool file. Score channels go through their env var (P3_TRAIN/STUDENT/...), never the positional list.')
        if 'ce2_' in os.path.basename(p): continue  # pass-2 files are merged below, never used as the pool
        target = extras if 'ce_x' in os.path.basename(p) else rows
        for line in open(p, encoding='utf-8'):
            if line.strip(): r = json.loads(line); q = str(r['qid']); target[q] = (target.get(q, []) + r['docs']) if target is extras else r['docs']  # several ce_x* files concatenate in path order
    for q, docs in extras.items():
        if q in rows:
            seen = {d['doc_id'] for d in rows[q]}
            for i, d in enumerate(docs, 1):
                if d['doc_id'] not in seen:
                    seen.add(d['doc_id']); rows[q].append(dict(d, rank=60 + i))
    # pass-2 files ('ce2_' prefix) add chunk logits to docs already present: merge by doc_id (max over all chunks)
    for p in paths:
        if 'ce2_' not in os.path.basename(p): continue
        for line in open(p, encoding='utf-8'):
            if not line.strip(): continue
            r = json.loads(line); q = str(r['qid'])
            if q not in rows: continue
            by = {d['doc_id']: d for d in rows[q]}
            for d in r['docs']:
                t = by.get(d['doc_id'])
                if t is None: continue
                known = set(t['chunks']); t.setdefault('p1', list(t['logits']))
                t['chunks'] = t['chunks'] + [c for c in d['chunks'] if c not in known]
                t['logits'] = t['logits'] + [l for c, l in zip(d['chunks'], d['logits']) if c not in known]
                t['pass2'] = 1
    return rows
def load_dense(path):
    d = {}
    if path and os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            r = json.loads(line); d[str(r['qid'])] = (r['ranking'], r['scores'])
    return d

def load_lex(path):
    """Lexical-alignment rows (lexalign.py output) -> {qid: {doc_id: [6 features]}}."""
    out = {}
    for line in open(path, encoding='utf-8'):
        if line.strip(): r = json.loads(line); out[str(r['qid'])] = {d['doc_id']: d['lex'] for d in r['docs']}
    return out

def load_student(spec):
    st = {}
    if not spec: return st
    for p in spec.split(','):
        for line in open(p, encoding='utf-8'):
            if not line.strip(): continue
            r = json.loads(line); q = str(r['qid']); st.setdefault(q, {})
            for d in r['docs']: st[q][d['doc_id']] = max(st[q].get(d['doc_id'], -1e9), max(d['logits']))
    return st

DOC_META = json.load(open(f'{ROOT}/doc_meta.json'))
COVFEAT = bool(os.environ.get('COVFEAT'))
DETRANK = bool(os.environ.get('DETRANK'))  # tie-break score ranks by doc_id instead of dict insertion order.
# Without it, srank/qrank/p3rank depend on the order the score files were listed, because a few percent of
# student scores tie exactly. The pipeline always sets DETRANK=1.
def _rank(d):
    ks = sorted(d, key=(lambda k: (-d[k], k)) if DETRANK else (lambda k: -d[k]))
    return {k: i + 1 for i, k in enumerate(ks)}
DOC_NCHUNKS = json.load(open(f'{ROOT}/doc_nchunks.json')) 
TYPES = ['luat', 'nghi-dinh', 'thong-tu', 'quyet-dinh', 'other']
def _strip(s): return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').lower()
def qmeta(question):
    toks = set(t for t in re.findall(r'[a-z0-9]+', _strip(question)) if len(t) > 1)
    nums = set(re.findall(r'(\d+)/(\d{4})', question))
    return toks, {a + '/' + b for a, b in nums}
def lex_feats(question, doc_id):
    m = DOC_META.get(doc_id); qt, qn = qmeta(question)
    if not m: return [0.0] * 9
    dt = set(m['toks']); inf = {t for t in dt if len(t) > 2}
    ov = len(qt & inf) / len(inf) if inf else 0.0
    return [ov, float(len(qt & inf)), int(bool(m['num']) and m['num'] in qn), m['amend'], m['year'] / 2030.0] + [int(m['type'] == t) for t in TYPES[:4]]

_CHUNK_TOK = None
def chunk_tokens():
    global _CHUNK_TOK
    if _CHUNK_TOK is None:
        _CHUNK_TOK = {}
        for line in open(f'{O}/best_chunk_text.jsonl', encoding='utf-8'):
            r = json.loads(line); _CHUNK_TOK[r['chunk_id']] = frozenset(t for t in re.findall(r'\w+', unicodedata.normalize('NFC', r['text']).casefold()) if len(t) > 1)
    return _CHUNK_TOK
_GRAPH = None
def graph():
    global _GRAPH
    if _GRAPH is None:
        E = collections.defaultdict(set); REL = collections.defaultdict(set)
        for line in open(f'{ROOT}/graph/legal_reference_graph.jsonl', encoding='utf-8'):
            r = json.loads(line)
            if r.get('expandable') and r.get('target_doc_id') and r['target_doc_id'] != r['source_doc_id']:
                E[r['source_doc_id']].add(r['target_doc_id']); E[r['target_doc_id']].add(r['source_doc_id'])
                REL[(r['source_doc_id'], r['target_doc_id'])].add(r['relation']); REL[(r['target_doc_id'], r['source_doc_id'])].add(r['relation'])
        _GRAPH = (E, REL)
    return _GRAPH
RELS = ['amends', 'implements', 'legal_basis', 'replaces_or_repeals', 'cross_reference']
def graph_feats(docs, order):
    E, REL = graph(); top1 = docs[order[0]]['doc_id']; top3 = {docs[i]['doc_id'] for i in order[:3]}
    return [[int(d['doc_id'] in E.get(top1, ())), sum(d['doc_id'] in E.get(t, ()) for t in top3 if t != d['doc_id']), len(E.get(d['doc_id'], ()))]
            + [int(r in REL.get((d['doc_id'], top1), ())) for r in RELS] for d in docs]
STOP = {'và', 'của', 'là', 'được', 'theo', 'tại', 'cho', 'trong', 'về', 'với', 'các', 'những', 'một', 'thì', 'đó', 'này', 'nào', 'như', 'thế', 'gì', 'có', 'không', 'bao', 'nhiêu', 'khi'}
def coverage_feats(question, docs, order):
    """Residual coverage: walk candidates in teacher order; per doc, the share of question tokens its best chunk covers,
    and how many of those were not already covered by higher-ranked docs."""
    ct = chunk_tokens(); qt = {t for t in re.findall(r'\w+', unicodedata.normalize('NFC', question).casefold()) if len(t) > 1 and t not in STOP}
    out = [None] * len(docs); covered = set()
    for i in order:
        d = docs[i]; c = d['chunks'][max(range(len(d['logits'])), key=lambda k: d['logits'][k])]
        toks = ct.get(c, frozenset()); hit = qt & toks; new = hit - covered
        out[i] = [len(hit) / max(len(qt), 1), len(new) / max(len(qt), 1), float(len(new)), (len(hit) - len(new)) / max(len(hit), 1)]
        covered |= hit
    return out

QFEAT = bool(os.environ.get('QFEAT'))
BMFEAT = bool(os.environ.get('BMFEAT'))  # QFEAT: in-query comparative blocks; BMFEAT: raw BM25 magnitudes (needs BM_POOL=<pool_*.jsonl>). Both OFF in the pipeline.
QF_SPANS = {}  # absolute column ranges of the QFEAT blocks, filled on the first features() call
def _cmp3(v):
    """One score channel seen from inside its own query: z-score, gap to the 5th best (the slot-5 cut line),
    and that gap normalised by the query's own top1-top5 spread. Only meaningful under a listwise ranking loss."""
    v = np.asarray(v, dtype=np.float64); s = np.sort(v)[::-1]; cut = s[4] if len(s) > 4 else s[-1]
    return np.stack([(v - v.mean()) / (v.std() + 1e-6), v - cut, (v - cut) / (s[0] - cut + 1e-6)], 1)

_BMP = None
def bm_pool():
    """BM25 pool scores from the BM_POOL files: {qid: {doc_id: (doc score, top chunk scores)}}. The base columns only
    use the BM25 rank; the BMFEAT block adds these magnitudes."""
    global _BMP
    if _BMP is None:
        _BMP = {}
        for path in os.environ.get('BM_POOL', '').split(','):
            if not path: continue
            for line in open(path, encoding='utf-8'):
                if not line.strip(): continue
                r = json.loads(line); _BMP[str(r['qid'])] = {d['doc_id']: (d['score'], d['chunk_scores']) for d in r['docs']}
    return _BMP

def load_extra(spec):
    """'name=path,...' -> list of dense dicts, one per extra dense channel (5 columns each in features())."""
    return [load_dense(a.split('=')[1]) for a in (spec or '').split(',') if a]

def features(docs, dense=None, top=60, student=None, question=None, extra=(), qrr=None, p3=None, qid=None, lex=None, qrr2=None):
    """Per-candidate feature rows for one query. Uses only label-blind pool/score info."""
    docs = [d for d in docs if d['rank'] <= top or d['rank'] > 60]
    p1 = [d.get('p1', d['logits']) for d in docs]
    mx = np.array([max(l) for l in p1]); order = np.array(sorted(range(len(docs)), key=lambda i: (-mx[i], docs[i]['doc_id']))); ce_rank = np.empty(len(docs)); ce_rank[order] = np.arange(1, len(docs) + 1)
    best = mx.max(); second_best = np.sort(mx)[-2] if len(mx) > 1 else best
    sig = 1 / (1 + np.exp(-mx))
    p2 = np.array([max(d['logits']) if d.get('pass2') else -20.0 for d in docs]); p2best = p2.max()
    p2order = np.array(sorted(range(len(docs)), key=lambda i: (-p2[i], docs[i]['doc_id']))); p2rank = np.empty(len(docs)); p2rank[p2order] = np.arange(1, len(docs) + 1)
    drank = {}; dscore = {}
    if dense: drank = {d: i + 1 for i, d in enumerate(dense[0])}; dscore = dict(zip(dense[0], dense[1]))
    dtop = dense[1][0] if dense else 0.0
    if student is not None:
        smax = max(student.values()) if student else 0.0
        srank = _rank(student)
    if qrr is not None:
        qmax = max(qrr.values()) if qrr else 0.0; qrank = _rank(qrr)
    if qrr2 is not None:
        q2max = max(qrr2.values()) if qrr2 else 0.0; q2rank = _rank(qrr2)
    if p3 is not None:
        # pass-3 = teacher rescoring of more chunks for the docs ranked 11-40 by the teacher CE. Kept as its own
        # channel with its own coverage flag; it is not max-merged into the student channel.
        p3max = max(p3.values()) if p3 else 0.0; p3rank = _rank(p3)
    rows = []
    cov = coverage_feats(question, docs, list(order)) if question is not None else None
    gf = graph_feats(docs, list(order))
    for i, d in enumerate(docs):
        l = sorted(p1[i], reverse=True)
        f = [l[0], l[1] if len(l) > 1 else -20, l[2] if len(l) > 2 else -20, float(np.mean(l[:3])), min(len(l), 3),
             l[0] - best, l[0] - second_best, ce_rank[i], ce_rank[i] / len(docs), sig[i], sig[i] - sig.max(),
             d['rank'], np.log(d['rank']),
             drank.get(d['doc_id'], 200), np.log(drank.get(d['doc_id'], 200)), dscore.get(d['doc_id'], 0.0),
             dscore.get(d['doc_id'], 0.0) - dtop,
             int(d['doc_id'] in drank and drank[d['doc_id']] <= 10), int(d['rank'] <= 10),
             int(bool(d.get('pass2'))), p2[i], p2[i] - l[0], p2[i] - p2best, p2rank[i],
             float(np.sum(np.array(d['logits']) > max(d['logits']) - 2.0))]
        if student is not None:
            sm = student.get(d['doc_id'], -20.0); f += [sm, sm - smax, srank.get(d['doc_id'], len(docs) + 1), sm - l[0]]
        if question is not None: f += lex_feats(question, d['doc_id']) + cov[i]
        if COVFEAT: 
            nc = len(d['chunks']); tc = DOC_NCHUNKS.get(d['doc_id'], nc)
            f += [nc, tc, nc / max(tc, 1), float(tc - nc)]  # how much of this doc the CE actually read
        f += gf[i]
        if qrr is not None:
            qm = qrr.get(d['doc_id'], -20.0); f += [qm, qm - qmax, qrank.get(d['doc_id'], len(docs) + 1), qm - l[0]]
        if p3 is not None:
            p3cov = int(d['doc_id'] in p3); pm = p3[d['doc_id']] if p3cov else -20.0  # NB: `cov` is taken (coverage_feats)
            f += [pm, pm - p3max, p3rank.get(d['doc_id'], len(docs) + 1), pm - l[0], p3cov]
        for e in extra:
            r = e['rk'].get(d['doc_id'], 200); v = e['sc'].get(d['doc_id'], 0.0)
            f += [r, np.log(r), v, v - e['top'], int(r <= 10)]
        if qrr2 is not None:  # S2 MIL scorer (Qwen3-Reranker base), same 4 cols as qrr
            qm = qrr2.get(d['doc_id'], -20.0); f += [qm, qm - q2max, q2rank.get(d['doc_id'], len(docs) + 1), qm - l[0]]
        if lex is not None: f += list(lex.get(d['doc_id'], [0.0] * 6))  # lexical alignment (lexalign.py), 6 cols at the end
        rows.append(f)
    R = np.array(rows, dtype=np.float32)
    if not QFEAT: return R
    n = len(docs); ids = [d['doc_id'] for d in docs]; z0 = np.zeros(n)
    sc = [mx, p2,
          np.array([student.get(i, -20.0) for i in ids]) if student is not None else z0,
          np.array([dscore.get(i, 0.0) for i in ids]) if dense else z0,
          np.array([p3.get(i, -20.0) for i in ids]) if p3 is not None else z0]
    sc += [np.array([e['sc'].get(i, 0.0) for i in ids]) for e in extra]
    rk = [ce_rank, p2rank,
          np.array([srank.get(i, n + 1) for i in ids]) if student is not None else z0,
          np.array([drank.get(i, 200) for i in ids]) if dense else z0,
          np.array([p3rank.get(i, n + 1) for i in ids]) if p3 is not None else z0]
    rk += [np.array([e['rk'].get(i, 200) for i in ids]) for e in extra]
    Rk = np.stack(rk, 1).astype(np.float64)
    Z = np.hstack([_cmp3(c) for c in sc])                                   # 3 cols per score channel
    Xr = np.stack([Rk.mean(1), Rk.min(1), Rk.max(1), Rk.std(1),             # this doc's rank agreement across channels
                   (Rk <= 5).sum(1), (Rk <= 1).sum(1), (1.0 / (60.0 + Rk)).sum(1)], 1)
    s = np.sort(mx)[::-1]; cut = s[4] if len(s) > 4 else s[-1]; t1 = Rk.argmin(0)
    Qc = np.tile(np.array([mx.max(), mx.mean(), mx.std(), s[0] - cut, float(n), float((p2 > -20.0).sum()),
                           float((t1 == t1[0]).sum()),                      # channels agreeing with the teacher top-1
                           float(len(set(re.findall(r'\w+', question))) if question else 0.0)], dtype=np.float64), (n, 1))
    nc = np.array([len(d['chunks']) for d in docs], dtype=np.float64)
    tc = np.array([float(DOC_NCHUNKS.get(d['doc_id'], len(d['chunks']))) for d in docs])
    rat = nc / np.maximum(tc, 1.0)
    Cc = np.stack([rat - rat.mean(), (np.argsort(np.argsort(-rat)) + 1).astype(float),
                   nc - nc.max(), tc / (np.median(tc) + 1e-6)], 1)          # evidence coverage vs the rivals of this query
    blocks = [('z', Z), ('x', Xr), ('q', Qc), ('c', Cc)]
    bp = bm_pool().get(qid or '', {}) if BMFEAT else None
    if bp is not None:
        bs = np.array([bp[i][0] if i in bp else 0.0 for i in ids]); bt = bs.max() if len(bs) else 0.0
        cs = [bp[i][1] if i in bp else [] for i in ids]
        Bc = np.stack([bs, bs - bt, (bs - bs.mean()) / (bs.std() + 1e-6),
                       np.array([c[1] if len(c) > 1 else 0.0 for c in cs]),
                       np.array([float(np.mean(c)) if c else 0.0 for c in cs]),
                       np.array([c[0] - c[-1] if c else 0.0 for c in cs]),
                       np.array([float(sum(x >= 0.9 * c[0] for x in c)) if c else 0.0 for c in cs]),
                       np.array([float(i in bp) for i in ids]),
                       np.full(n, bt), np.full(n, float(np.std(bs)) if len(bs) else 0.0)], 1)
        blocks.append(('b', Bc))
    if not QF_SPANS:
        b = R.shape[1]
        for k, m in blocks:
            QF_SPANS[k] = (b, b + m.shape[1]); b += m.shape[1]
        QF_SPANS['all'] = (R.shape[1], b)
    return np.hstack([R] + [m.astype(np.float32) for _, m in blocks])

def extra_q(extras, q):
    out = []
    for e in extras:
        rank, score = e.get(q, ([], []))
        out.append({'rk': {d: i + 1 for i, d in enumerate(rank)}, 'sc': dict(zip(rank, score)), 'top': score[0] if score else 0.0})
    return out
