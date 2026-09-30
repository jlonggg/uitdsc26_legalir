"""Fit selector a3 on TRAIN only (train labels are read here and nowhere else; no public/warmup/private rows touched).
Inputs in $FIT (assembled by `run_from_scratch.sh a3`): the ce train outputs (pools, dense, teacher, student held-fold scores),
S1 MIL held-fold scores mil/outer0-4, S2 MIL held-fold scores mil2/outer<MIL2_FOLDS>, lexalign_train.jsonl (7,000 rows).
  build        : 83-col matrix = ce+mil2 (77) + 6 lexalign cols, cached to $OUT/X83.npz; self-check first 73 cols == selector_stage 'ce+mil'.
  final:SCORER : fit on all train rows (6 seeds, same LightGBM recipe as selector_stage), S2 cols NaN on NAN_FOLDS
                 -> $OUT/model_<SCORER>/{seed*.txt,selector.json}, consumed by `selector_stage.py predict`. The shipped run uses ce+mil2.
  e1 / e2:CFG / e3:NAME / report : diagnostics (fold-swap comparisons of column sets, miss anatomy); not used by the reproduction.
Env: FIT (dir), OUT, MIL2_FOLDS, NAN_FOLDS, CLEAN_CORPUS, COVFEAT=1, DETRANK=1, CLEAN_THREADS."""
import json, os, sys, numpy as np
from pathlib import Path
sys.path.insert(0, '/app/src'); import features as F, selector_stage as S
import lightgbm as lgb
G = Path(os.environ['FIT']); OUT = Path(os.environ['OUT']); SEEDS = [2026, 7, 99, 314, 2718, 1618]
Q = json.load(open(G / 'questions.json')); GOLD = {q: set(v['answer']) for q, v in Q.items()}
COLS = {'ce': list(range(54)) + list(range(58, 73)), 'ce+mil': list(range(73)), 'ce+mil2': list(range(77)),
        'ce+mil+lex': list(range(73)) + list(range(77, 83)), 'ce+mil2+lex': list(range(83))}
S2COLS = list(range(73, 77))
MIL2F = [int(x) for x in os.environ.get('MIL2_FOLDS', '0,1').split(',')]; NANF = tuple(int(x) for x in os.environ.get('NAN_FOLDS', '2,3,4').split(','))  # run_from_scratch.sh a3: 0,1,2,3 / 4 (S2 has no held-fold scores on fold 4)


def load_scores(paths):
    s = {}
    for p in paths:
        for line in open(p):
            r = json.loads(line); d = s.setdefault(str(r['qid']), {})
            for x in r['docs']: d[x['doc_id']] = max(d.get(x['doc_id'], -1e9), max(x['logits']))
    return s


def build():
    student = [G / f'ce_ft_hard_kl0.0/outer{k}/{f}.jsonl' for k in range(5) for f in ('held_scores', 'held_scores2', 'held_xscores', 'held_xscores2')]
    milp = [G / f'mil/outer{k}/held_mil.jsonl' for k in range(5)]
    qs, X73, groups, ids = S.matrix(G, Q, student, 'ce+mil', milp)  # writes best_chunk_text.jsonl, asserts 73 cols
    # second pass: same channels + S2 (MIL2_FOLDS only; other folds -> {} = constant fill, NaN-ed later in final()) + lex
    pools = [G / 'ce_train.s0.jsonl', G / 'ce_xtrain.jsonl', G / 'ce_x2train.jsonl', G / 'ce2_train.jsonl']
    rows = F.load_rows([str(p) for p in pools]); F.O = str(G); F._CHUNK_TOK = None
    dense = F.load_dense(G / 'dense_train.jsonl'); extra = [F.load_dense(G / f'dense_{n}_train.jsonl') for n in ('viemb2', 'harrier')]
    st = F.load_student(','.join(map(str, student))); p3 = F.load_student(str(G / 'p3teach_train.jsonl')); mil = F.load_student(','.join(map(str, milp)))
    mil2 = F.load_student(','.join(str(G / f'mil2/outer{k}/held_mil.jsonl') for k in MIL2F)); lex = F.load_lex(G / 'lexalign_train.jsonl')
    assert set(lex) == set(qs), 'lex incomplete'
    vals = []
    for q in qs:
        x = F.features(rows[q], dense[q], student=st[q], question=Q[q]['question'], extra=F.extra_q(extra, q), p3=p3[q], qrr=mil[q], qid=q, lex=lex[q], qrr2=mil2.get(q, {}))
        assert x.shape[1] == 83, x.shape; vals.append(x)
    X = np.vstack(vals); assert np.array_equal(X[:, :73], X73), 'column layout drifted'
    fold = np.array([F.fold(Q[q]['question']) for q in qs]); off = np.cumsum([0] + groups)
    np.savez(OUT / 'X83.npz', X=X, groups=np.array(groups), fold=fold, qs=np.array(qs), ids=json.dumps(ids))
    print('built', X.shape, 'folds', np.bincount(fold), flush=True)


def load():
    z = np.load(OUT / 'X83.npz', allow_pickle=True); qs = list(z['qs']); groups = list(z['groups']); ids = json.loads(str(z['ids']))
    return z['X'], groups, z['fold'], qs, ids


def fit_predict(X, groups, fold, qs, ids, cols, tr_folds, ev_fold, nan_s2_folds=(), tag=''):
    off = np.cumsum([0] + groups); Xc = X[:, cols].astype(np.float32).copy()
    if nan_s2_folds:
        s2 = [cols.index(c) for c in S2COLS]
        for i, q in enumerate(qs):
            if fold[i] in nan_s2_folds: Xc[off[i]:off[i + 1], s2] = np.nan
    tri = [i for i in range(len(qs)) if fold[i] in tr_folds]; evi = [i for i in range(len(qs)) if fold[i] == ev_fold]
    rows = lambda I: np.concatenate([np.arange(off[i], off[i + 1]) for i in I])
    Xt, gt = Xc[rows(tri)], [groups[i] for i in tri]; yt = np.array([d in GOLD[qs[i]] for i in tri for d in ids[qs[i]]], dtype=np.int32)
    pred = {}
    for seed in SEEDS:
        m = lgb.LGBMRanker(objective='lambdarank', n_estimators=1200, learning_rate=.03, num_leaves=15, min_child_samples=30, reg_lambda=5.,
                           lambdarank_truncation_level=5, label_gain=[0, 1], random_state=seed, n_jobs=int(os.environ.get('CLEAN_THREADS', '8')), verbose=-1)
        m.fit(Xt, yt, group=gt)
        for i in evi:
            s = m.predict(Xc[off[i]:off[i + 1]]); s = (s - s.mean()) / (s.std() + 1e-9)
            p = pred.setdefault(qs[i], np.zeros(groups[i])); p += s
        print(tag, 'seed', seed, 'done', flush=True)
    return {q: dict(zip(ids[q], p.tolist())) for q, p in pred.items()}


def top5(s, q): return set(sorted(s[q], key=lambda d: (-s[q][d], d))[:5])
def rec(s, q): return len(top5(s, q) & GOLD[q]) / len(GOLD[q])
def paired(a, b, qs):
    d = np.array([rec(a, q) - rec(b, q) for q in qs])
    return {'n': len(qs), 'A': float(np.mean([rec(a, q) for q in qs])), 'B': float(np.mean([rec(b, q) for q in qs])),
            'delta': float(d.mean()), 'se': float(d.std(ddof=1) / np.sqrt(len(d))), 'W': int((d > 0).sum()), 'L': int((d < 0).sum())}


def save_pred(name, pred): json.dump(pred, open(OUT / f'pred_{name}.json', 'w'))
def load_pred(name): return json.load(open(OUT / f'pred_{name}.json'))


def e1():
    X, groups, fold, qs, ids = load()
    for cfg, cols in COLS.items():
        pred = {}
        for tr, ev in ((1, 0), (0, 1)): pred.update(fit_predict(X, groups, fold, qs, ids, cols, (tr,), ev, tag=f'e1 {cfg} fit{tr}'))
        save_pred('e1_' + cfg, pred)
        print('E1', cfg, {k: float(np.mean([rec(pred, q) for q in pred if fold[qs.index(q)] == k])) for k in (0, 1)}, flush=True)


def e2(cfg):
    X, groups, fold, qs, ids = load()
    base, mode = cfg.rsplit('_', 1); cols = COLS[base]
    pred = {}
    for tr, ev in ((1, 0), (0, 1)):
        nan = (2, 3, 4) if mode == 'nan' else ()
        pred.update(fit_predict(X, groups, fold, qs, ids, cols, (tr, 2, 3, 4), ev, nan_s2_folds=nan, tag=f'e2 {cfg} ev{ev}'))
    save_pred('e2_' + cfg, pred)
    print('E2', cfg, {k: float(np.mean([rec(pred, q) for q in pred if fold[qs.index(q)] == k])) for k in (0, 1)}, flush=True)


def report():
    X, groups, fold, qs, ids = load(); fo = {q: int(fold[i]) for i, q in enumerate(qs)}
    names = [p.stem[5:] for p in sorted(OUT.glob('pred_*.json'))]; P = {n: load_pred(n) for n in names}
    ev = sorted(P[names[0]]); out = {'n_eval': len(ev), 'R@5': {}, 'paired': {}}
    for n in names: out['R@5'][n] = {'all': float(np.mean([rec(P[n], q) for q in ev])), **{f'fold{k}': float(np.mean([rec(P[n], q) for q in ev if fo[q] == k])) for k in (0, 1)}}
    for ref in ('e1_ce+mil2', 'e2_ce+mil_big', 'e1_ce+mil'):
        if ref in P: out['paired'][f'vs {ref}'] = {n: paired(P[n], P[ref], ev) for n in names if n != ref}
    json.dump(out, open(OUT / 'report.json', 'w'), indent=1); print(json.dumps(out, indent=1))


def e3(name):
    """Anatomy of the misses of prediction `name` on folds 0+1: where is the gold in each channel."""
    X, groups, fold, qs, ids = load(); pred = load_pred(name); off = np.cumsum([0] + groups); qi = {q: i for i, q in enumerate(qs)}
    s1 = load_scores([G / f'mil/outer{k}/held_mil.jsonl' for k in (0, 1)]); s2 = load_scores([G / f'mil2/outer{k}/held_mil.jsonl' for k in (0, 1)])
    st = load_scores([G / f'ce_ft_hard_kl0.0/outer{k}/{f}.jsonl' for k in (0, 1) for f in ('held_scores', 'held_scores2', 'held_xscores', 'held_xscores2')])
    zm = {q: {d: (s1[q][d] - np.mean(list(s1[q].values()))) / (np.std(list(s1[q].values())) + 1e-9) + (s2[q][d] - np.mean(list(s2[q].values()))) / (np.std(list(s2[q].values())) + 1e-9) for d in s1[q]} for q in pred}
    def rk(s, q, g): return sorted(s[q], key=lambda d: (-s[q][d], d)).index(g) + 1 if g in s[q] else 999
    miss = []; nch = F.DOC_NCHUNKS
    for q in pred:
        for g in GOLD[q]:
            if g in top5(pred, q): continue
            miss.append({'q': q, 'fold': int(fold[qi[q]]), 'gold': g, 'in_pool': g in pred[q], 'multi': len(GOLD[q]) > 1, 'final': rk(pred, q, g), 's1': rk(s1, q, g), 's2': rk(s2, q, g), 'student': rk(st, q, g), 'zmean': rk(zm, q, g), 'nchunks': nch.get(g, 0)})
    def bucket(r): return 'top5' if r <= 5 else '6-10' if r <= 10 else '11-20' if r <= 20 else '21+' if r < 999 else 'absent'
    summ = {'n_missed_gold': len(miss), 'n_questions_with_miss': len({m['q'] for m in miss}), 'out_of_pool': sum(not m['in_pool'] for m in miss), 'multi_gold_share': float(np.mean([m['multi'] for m in miss]))}
    for ch in ('final', 's1', 's2', 'student', 'zmean'):
        b = {}
        for m in miss: b[bucket(m[ch])] = b.get(bucket(m[ch]), 0) + 1
        summ[f'rank_{ch}'] = b
    summ['any_scorer_top5'] = sum(min(m['s1'], m['s2'], m['student']) <= 5 for m in miss); summ['both_mil_top5'] = sum(m['s1'] <= 5 and m['s2'] <= 5 for m in miss)
    summ['union_oracle_R@5'] = float(np.mean([len((top5(pred, q) | top5(s1, q) | top5(s2, q)) & GOLD[q]) / len(GOLD[q]) for q in pred]))
    summ['final_top10_oracle'] = float(np.mean([len(set(sorted(pred[q], key=lambda d: (-pred[q][d], d))[:10]) & GOLD[q]) / len(GOLD[q]) for q in pred]))
    # evidence-size lever: S1 standalone 0-hit rate of single-gold questions by gold doc chunk count
    allq = [q for q in pred if len(GOLD[q]) == 1]; edges = [0, 5, 10, 20, 40, 80, 10 ** 9]; by = {}
    for q in allq:
        g = next(iter(GOLD[q])); n = nch.get(g, 0); k = next(i for i in range(len(edges) - 1) if edges[i] <= n < edges[i + 1])
        key = f'{edges[k]}-{edges[k+1]-1 if edges[k+1] < 10**9 else "+"}'; v = by.setdefault(key, [0, 0, 0]); v[0] += 1; v[1] += rk(s1, q, g) > 5; v[2] += rk(pred, q, g) > 5
    summ['s1_miss_by_gold_nchunks'] = {k: {'n': v[0], 's1_miss': v[1], 'final_miss': v[2]} for k, v in sorted(by.items(), key=lambda kv: int(kv[0].split('-')[0]))}
    json.dump({'summary': summ, 'misses': miss}, open(OUT / f'anatomy_{name}.json', 'w'), indent=1); print(json.dumps(summ, indent=1))


def final(scorer='ce+mil2+lex'):
    """Selector fit on ALL 7,000 train rows, S2 columns NaN on NAN_FOLDS (no S2 held-fold scores there), same lgb recipe
    as selector_stage.main; boosters + selector.json in OUT/model_<scorer> so selector_stage.py predict can consume them."""
    X, groups, fold, qs, ids = load(); cols = COLS[scorer]; off = np.cumsum([0] + groups); Xc = X[:, cols].astype(np.float32).copy()
    s2 = [cols.index(c) for c in S2COLS]
    for i, q in enumerate(qs):
        if fold[i] in NANF: Xc[off[i]:off[i + 1], s2] = np.nan
    y = np.array([d in GOLD[q] for q in qs for d in ids[q]], dtype=np.int32); M = OUT / f'model_{scorer}'; M.mkdir(exist_ok=True)
    for seed in SEEDS:
        m = lgb.LGBMRanker(objective='lambdarank', n_estimators=1200, learning_rate=.03, num_leaves=15, min_child_samples=30, reg_lambda=5.,
                           lambdarank_truncation_level=5, label_gain=[0, 1], random_state=seed, n_jobs=int(os.environ.get('CLEAN_THREADS', '8')), verbose=-1)
        m.fit(Xc, y, group=groups); m.booster_.save_model(str(M / f'seed{seed}.txt')); print('final', scorer, 'seed', seed, 'done', flush=True)
    (M / 'selector.json').write_text(json.dumps({'features': len(cols), 'scorer': scorer, 'seeds': SEEDS, 'trees': 1200, 'train_queries': len(qs),
        'recipe': 'symmetric69_plus_mil_plus_mil2_v1' if scorer == 'ce+mil2' else 'symmetric69_plus_mil_plus_mil2_plus_lexalign_v1', 'note': f'S2 (mil2) columns NaN on folds {NANF}: no S2 OOF rows there yet (CONTEXT 50 / 51.5)', 'mil2_folds': MIL2F,
        'evaluation': 'No unbiased end-to-end validation has been run by this fitting command.'}, indent=2))


if __name__ == '__main__':
    part = sys.argv[1]
    if part == 'build': build()
    elif part == 'e1': e1()
    elif part.startswith('e2:'): e2(part[3:])
    elif part == 'report': report()
    elif part.startswith('e3:'): e3(part[3:])
    elif part.startswith('final:'): final(part[6:])
