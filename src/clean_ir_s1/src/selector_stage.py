"""LightGBM lambdarank selector: `fit` trains one ranker per seed on the train features; `predict` averages the
per-query z-scored outputs of the saved rankers and keeps the top-5 docs. Same feature channels on both sides."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import lightgbm as lgb
import features as F


WIDTH = {'ce': 69, 'mil': 60, 'ce+mil': 73, 'ce+lex': 75, 'ce+mil2': 77, 'ce+mil2+lex': 83}  # scorer -> feature width; asserted against the real features() output


def matrix(directory, questions, student_paths, scorer='ce', mil_paths=(), mil2_paths=()):
    tag = 'train' if (directory/'ce_train.s0.jsonl').exists() else 'public'
    pools = [directory/('ce_'+tag+('.s0' if tag=='train' else '')+'.jsonl'),
             directory/('ce_x'+tag+'.jsonl'), directory/('ce_x2'+tag+'.jsonl'),
             directory/('ce2_'+tag+'.jsonl')]
    if scorer == 'mil':  # S1: the MIL scorer's rows ARE the pool (every doc, K evidence chunks); no teacher/student/pass columns
        pools = list(mil_paths)
    rows = F.load_rows([str(p) for p in pools])
    assert set(rows) == set(questions), 'Incomplete pool'
    # Cache text for every best chunk actually used, including pass-2 changes.
    needed = {d['chunks'][int(np.argmax(d['logits']))] for q in rows for d in rows[q]}
    text_path = directory/'best_chunk_text.jsonl'
    found = set()
    with text_path.open('w') as out, (Path(os.environ['CLEAN_CORPUS'])/'chunks.jsonl').open() as corpus:
        for line in corpus:
            c = json.loads(line)
            if c['chunk_id'] in needed:
                found.add(c['chunk_id'])
                out.write(json.dumps({'chunk_id': c['chunk_id'], 'text': c['article']+' '+c['text']})+'\n')
    assert found == needed, 'Missing best-chunk text'
    F.O = str(directory); F._CHUNK_TOK = None
    dense = F.load_dense(directory/('dense_'+tag+'.jsonl'))
    extra = [F.load_dense(directory/('dense_'+name+'_'+tag+'.jsonl')) for name in ('viemb2', 'harrier')]
    student = F.load_student(','.join(map(str, student_paths))) if scorer != 'mil' else None
    p3 = F.load_student(str(directory/('p3teach_'+tag+'.jsonl'))) if scorer != 'mil' else None
    mil = F.load_student(','.join(map(str, mil_paths))) if scorer.startswith('ce+mil') else None  # doc score = max evidence logit
    mil2 = F.load_student(','.join(map(str, mil2_paths))) if 'mil2' in scorer else None
    lex = F.load_lex(directory/('lexalign_'+tag+'.jsonl')) if 'lex' in scorer else None
    qs, values, groups, ids = sorted(questions), [], [], {}
    for channel in (dense, *extra, student, p3, mil, mil2, lex):
        assert channel is None or set(channel) == set(qs), 'Incomplete score channel'
    for q in qs:
        ds = rows[q]
        assert len({d['doc_id'] for d in ds}) == len(ds) and len(ds)>=5
        x = F.features(ds, dense[q], student=student[q] if student is not None else None, question=questions[q]['question'],
                       extra=F.extra_q(extra,q), p3=p3[q] if p3 is not None else None, qrr=mil[q] if mil is not None else None, qid=q, lex=lex[q] if lex is not None else None, qrr2=mil2[q] if mil2 is not None else None)
        assert x.shape == (len(ds), WIDTH[scorer]), x.shape
        values.append(x); groups.append(len(ds)); ids[q] = [d['doc_id'] for d in ds]
    X = np.vstack(values)
    assert np.isfinite(X).all()
    return qs, X, groups, ids


def rank(qs, ids, groups, X, models):
    acc = np.zeros(len(X)); offsets = np.cumsum([0]+groups)
    for model in models:
        score = model.predict(X, num_threads=int(os.environ.get('CLEAN_THREADS','3')))
        for i,q in enumerate(qs):
            sl = slice(offsets[i], offsets[i+1]); p=score[sl]
            acc[sl] += (p-p.mean())/(p.std()+1e-9)
    result = {}
    for i,q in enumerate(qs):
        p = acc[offsets[i]:offsets[i+1]]
        top = sorted(range(len(p)), key=lambda j: (-float(p[j]),ids[q][j]))[:5]
        result[q] = {'answer': [ids[q][j] for j in top]}
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['fit','predict'])
    ap.add_argument('directory',type=Path);ap.add_argument('questions',type=Path);ap.add_argument('model',type=Path)
    ap.add_argument('--trees',type=int,default=1200);ap.add_argument('--seeds',default='2026,7,99,314,2718,1618')
    ap.add_argument('--scorer',choices=list(WIDTH),default=None)
    a=ap.parse_args();questions=json.loads(a.questions.read_text());d=a.directory
    if a.stage=='fit':
        scorer=a.scorer or 'ce'
        active={F.fold(r['question']) for r in questions.values()}
        student=[d/('ce_ft_hard_kl0.0/outer%d/%s.jsonl'%(k,s)) for k in sorted(active)
                 for s in ('held_scores','held_scores2','held_xscores','held_xscores2')]
        mil=[d/('mil/outer%d/held_mil.jsonl'%k) for k in sorted(active)]
        mil2=[d/('mil2/outer%d/held_mil.jsonl'%k) for k in sorted(active)]
    else:
        scorer=json.loads((a.model/'selector.json').read_text()).get('scorer','ce')
        student=[d/(s+'.jsonl') for s in ('public_scores','public_scores2','public_xscores','public_xscores2')]
        mil=[d/'mil_public.jsonl']
        mil2=[d/'mil2_public.jsonl']
    qs,X,groups,ids=matrix(d,questions,student,scorer,mil,mil2)
    if a.stage=='fit':
        a.model.mkdir(exist_ok=True,parents=True)
        y=np.array([doc in set(questions[q]['answer']) for q in qs for doc in ids[q]],dtype=np.int32)
        seeds=[int(s) for s in a.seeds.split(',')]
        for seed in seeds:
            m=lgb.LGBMRanker(objective='lambdarank',n_estimators=a.trees,learning_rate=.03,
                num_leaves=15,min_child_samples=30,reg_lambda=5.,lambdarank_truncation_level=5,
                label_gain=[0,1],random_state=seed,n_jobs=int(os.environ.get('CLEAN_THREADS','3')),verbose=-1)
            m.fit(X,y,group=groups);m.booster_.save_model(str(a.model/('seed%d.txt'%seed)))
        (a.model/'selector.json').write_text(json.dumps({'features':WIDTH[scorer],'scorer':scorer,'seeds':seeds,'trees':a.trees,
            'train_queries':len(qs),'recipe':{'ce':'symmetric_teacher_student_pass2_v1','mil':'s1_mil_scorer_v1','ce+mil':'symmetric69_plus_mil_v1','ce+lex':'symmetric69_plus_lexalign_v1','ce+mil2':'symmetric69_plus_mil_plus_mil2_v1','ce+mil2+lex':'symmetric69_plus_mil_plus_mil2_plus_lexalign_v1'}[scorer],
            'evaluation':'No unbiased end-to-end validation has been run by this fitting command.'},indent=2))
    else:
        manifest=json.loads((a.model/'selector.json').read_text())
        models=[lgb.Booster(model_file=str(a.model/('seed%d.txt'%s))) for s in manifest['seeds']]
        assert all(m.num_feature()==X.shape[1] for m in models)
        (d/'prediction.json').write_text(json.dumps(rank(qs,ids,groups,X,models),ensure_ascii=False,indent=1))


if __name__=='__main__':main()
