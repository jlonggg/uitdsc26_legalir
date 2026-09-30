#!/usr/bin/env python3
"""Standalone LegalIR: raw contexts -> indexes -> fitted model -> arbitrary questions."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

ROOT=Path(__file__).resolve().parent
SRC=ROOT/'src'
MODELS=['Qwen3-Embedding-0.6B','Vietnamese_Embedding_v2','vietlegal-harrier-0.6b','Vietnamese_Reranker']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def inventory(path, patterns=None):
    path=Path(path)
    files=sorted(path.rglob('*')) if path.is_dir() else [path]
    return {str(p.relative_to(path) if path.is_dir() else p.name):sha(p) for p in files
            if p.is_file() and '__pycache__' not in p.parts and (not patterns or p.suffix in patterns)}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def write(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(path)


def questions(path, training=False):
    # Reject duplicate JSON keys before any model sees this input.
    def unique(pairs):
        out={}
        for k,v in pairs:
            if k in out:raise ValueError('Duplicate JSON key '+k)
            out[k]=v
        return out
    data=json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=unique)
    if not isinstance(data,dict) or not data:raise ValueError('Expected nonempty question mapping')
    out={}
    for q,row in data.items():
        if not isinstance(row,dict) or not isinstance(row.get('question'),str) or not row['question'].strip():
            raise ValueError('Invalid question '+q)
        out[q]={'question':row['question']}
        if training:
            a=row.get('answer')
            if not isinstance(a,list) or not a or not all(isinstance(d,str) for d in a) or len(set(a))!=len(a):
                raise ValueError('Invalid train labels '+q)
            out[q]['answer']=a
    return out  # Inference answer/metadata fields are discarded, never used.


def fold(s):
    s=' '.join(re.findall(r'[^\W_]+',unicodedata.normalize('NFC',s).lower()))
    return int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%5


@contextlib.contextmanager
def lock(work):
    work.mkdir(parents=True,exist_ok=True)
    with (work/'.pipeline.lock').open('w') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another pipeline process owns '+str(work))
        yield


class Pipeline:
    def __init__(self,a):
        self.a=a;self.work=a.work.resolve();self.corpus=self.work/'corpus';self.state=self.work/'steps'
        self.state.mkdir(exist_ok=True);self.logdir=self.work/'logs';self.logdir.mkdir(exist_ok=True)
        self.source=inventory(SRC,{'.py'});self.source['pipeline.py']=sha(__file__)
        configpath=self.work/'config.json'
        if configpath.exists():
            self.cfg=json.loads(configpath.read_text())
            if self.cfg['source']!=self.source:
                raise RuntimeError('Source changed: use a new --work directory; refusing stale artifacts')
        else:
            if a.command!='prepare' or not a.contexts or not a.models:
                raise ValueError('First command must be prepare --contexts ... --models ...')
            contexts=a.contexts.resolve();models=a.models.resolve()
            if not list(contexts.glob('context_*.json')):raise ValueError('No context_*.json')
            for name in MODELS:
                if not (models/name/'config.json').is_file():raise ValueError('Missing local model '+name)
            context_hash={p.name:sha(p) for p in sorted(contexts.glob('context_*.json'))}
            model_hash={name:inventory(models/name,{'.json','.safetensors','.bin','.txt','.model'}) for name in MODELS}
            self.cfg={'recipe':'clean_ir_symmetric69_v1','contexts':str(contexts),'models':str(models),
                'context_hash':context_hash,'model_hash':model_hash,'source':self.source,'es':a.es,
                'owner':digest({'source':self.source,'contexts':context_hash,'work':str(self.work)})[:20]}
            write(configpath,self.cfg)
        self.env={**os.environ,'PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION':'python','HF_HUB_OFFLINE':'1',
            'TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false','CLEAN_WORK':str(self.work),
            'CLEAN_CORPUS':str(self.corpus),'CLEAN_MODELS':self.cfg['models'],'CLEAN_ES':self.cfg['es'],
            'CLEAN_INDEX':'clean_ir_'+self.cfg['owner']+'_chunks','CLEAN_WSEG':'clean_ir_'+self.cfg['owner']+'_wseg',
            'CLEAN_OWNER':self.cfg['owner'],'CLEAN_CE':str(Path(self.cfg['models'])/'Vietnamese_Reranker'),
            'COVFEAT':'1','DETRANK':'1','CLEAN_THREADS':str(a.threads),'OMP_NUM_THREADS':str(a.threads),
            'OPENBLAS_NUM_THREADS':'1'}
        # Optional feature switches read by src/features.py; unset them so the caller's shell cannot
        # change the fixed 69-column selector matrix.
        for key in ('QFEAT','BMFEAT','QRR_TRAIN','QRR_PUBLIC','QF_KEEP'):
            self.env.pop(key,None)

    def check_inputs(self,models=False):
        contexts=Path(self.cfg['contexts'])
        now={p.name:sha(p) for p in sorted(contexts.glob('context_*.json'))}
        if now!=self.cfg['context_hash']:raise RuntimeError('Corpus drift: use a new work directory')
        if models:
            for name in MODELS:
                now=inventory(Path(self.cfg['models'])/name,{'.json','.safetensors','.bin','.txt','.model'})
                if now!=self.cfg['model_hash'][name]:raise RuntimeError('Model drift: '+name)

    def step(self,name,script,args,outputs,extra=None):
        outputs=[Path(p) for p in outputs];cmd=[sys.executable,str(SRC/script),*map(str,args)]
        contract=digest({'config':self.cfg,'cmd':cmd,'extra':extra or {}})
        marker=self.state/(name+'.json')
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['contract']!=contract:raise RuntimeError('Stage input drift: '+name)
            if all(p.exists() and inventory(p)==old['outputs'][str(p)] for p in outputs):
                print('verified, skip',name,flush=True);return
            raise RuntimeError('Stage output missing/modified: '+name+'; use a fresh work directory')
        print('running',name,flush=True);started=time.time()
        with (self.logdir/(name+'.log')).open('a') as log:
            subprocess.run(cmd,env=self.env,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
        if not all(p.exists() for p in outputs):raise RuntimeError('Missing output for '+name)
        write(marker,{'contract':contract,'command':cmd,'elapsed_seconds':time.time()-started,
            'outputs':{str(p):inventory(p) for p in outputs}})

    def prepare(self):
        self.check_inputs(models=self.a.stage in ('all','dense'))
        if self.a.stage in ('all','corpus'):
            self.step('corpus','corpus.py',[self.cfg['contexts'],self.corpus],
                [self.corpus/p for p in ('catalog.json','doc_meta.json','doc_nchunks.json','chunks.jsonl','articles.jsonl','statistics.json')])
            self.step('graph','reference_graph.py',['--contexts',self.cfg['contexts'],'--output-dir',self.corpus/'graph'],
                [self.corpus/'graph'])
        if self.a.stage in ('all','index'):
            self.step('index','index.py',[],[self.corpus/'index.json'])
            self.verify_index()
        if self.a.stage in ('all','dense'):
            for name in MODELS[:3]:
                self.step('encode_'+name,'dense.py',['encode',name,'--batch',self.a.batch],[self.corpus/(name+'.npy')])
            self.step('model_audit','model_audit.py',[],[self.work/'parameter_audit.json'])

    def verify_index(self):
        from elasticsearch import Elasticsearch
        es=Elasticsearch(self.cfg['es'],request_timeout=60)
        expected=json.loads((self.corpus/'statistics.json').read_text())['chunks']
        for name in (self.env['CLEAN_INDEX'],self.env['CLEAN_WSEG']):
            m=es.indices.get_mapping(index=name)[name]['mappings']
            assert m.get('_meta',{}).get('clean_ir_owner')==self.cfg['owner'], 'Index ownership mismatch'
            assert es.count(index=name)['count']==expected, 'Index incomplete'

    def retrieve(self,d,qpath,tag):
        self.env['RB_OUT']=str(d)
        prefix=d.name+'_'
        self.step(prefix+'pool','build_pool.py',[qpath,d/('pool_'+tag+'.jsonl')],[d/('pool_'+tag+'.jsonl')])
        for model,short in zip(MODELS[:3],['','viemb2_','harrier_']):
            out=d/('dense_'+short+tag+'.jsonl')
            self.step(prefix+'dense_'+model,'dense.py',['retrieve',model,'--questions',qpath,'--out',out],[out])
        # extra: Qwen3 Article top-30 docs missing from the BM25 pool. extra2: interleaved
        # Vietnamese_Embedding_v2/Harrier top-20 docs missing from both, capped at 40.
        self.step(prefix+'extra','extra_pool.py',[d/('pool_'+tag+'.jsonl'),d/('dense_'+tag+'.jsonl'),d/('extra_'+tag+'.jsonl'),30],[d/('extra_'+tag+'.jsonl')])
        union=d/('dense_union_'+tag+'.jsonl')
        def readrows(p):return {r['qid']:r for r in map(json.loads,p.read_text().splitlines())}
        v,h,x=[readrows(d/(name+tag+'.jsonl')) for name in ('dense_viemb2_','dense_harrier_','extra_')]
        with union.open('w') as out:
            for q in sorted(v):
                seen={r['doc_id'] for r in x[q]['docs']};ranking=[]
                for i in range(20):
                    for channel in (v,h):
                        if i<len(channel[q]['ranking']):
                            doc=channel[q]['ranking'][i]
                            if doc not in seen:seen.add(doc);ranking.append(doc)
                out.write(json.dumps({'qid':q,'ranking':ranking})+'\n')
        self.step(prefix+'extra2','extra_pool.py',[d/('pool_'+tag+'.jsonl'),union,d/('extra2_'+tag+'.jsonl'),40],[d/('extra2_'+tag+'.jsonl')])

    def score(self,name,qpath,pool,out,adapter=None,top=60):
        args=[self.env['CLEAN_CE'],qpath,pool,out,'--top',top,'--batch',self.a.batch]
        if adapter:args+=['--adapter',adapter]
        self.step(name,'score_ce.py',args,[out])

    def second(self,name,qpath,pool,inputs,out,adapter=None,deep=False):
        """Pass 2: rescore up to 40 chunks of each of the top-10 docs by pass-1 max logit.
        deep=True is pass 3: docs ranked 11-40, up to 20 chunks each."""
        args=[self.env['CLEAN_CE'],qpath,pool,','.join(map(str,inputs)),out,'--batch',self.a.batch,
              '--top-docs',40 if deep else 10,'--cap',20 if deep else 40]
        if deep:args+=['--from-doc',10]
        if adapter:args+=['--adapter',adapter]
        self.step(name,'second_pass.py',args,[out])

    def teacher(self,d,qpath,tag):
        self.env['RB_OUT']=str(d);p=d.name+'_teacher_'
        inputs=[]
        for kind,stem,top in [('pool','ce_',60),('extra','ce_x',40),('extra2','ce_x2',40)]:
            out=d/(stem+tag+('.s0' if tag=='train' and kind=='pool' else '')+'.jsonl')
            inputs.append(out);self.score(p+kind,qpath,d/(kind+'_'+tag+'.jsonl'),out,top=top)
        self.second(p+'pass2',qpath,d/('pool_'+tag+'.jsonl'),inputs,d/('ce2_'+tag+'.jsonl'))
        self.second(p+'pass3',qpath,d/('pool_'+tag+'.jsonl'),inputs,d/('p3teach_'+tag+'.jsonl'),deep=True)

    def train(self):
        self.check_inputs(models=True);self.verify_index()
        q=questions(self.a.train,training=True)
        catalog=json.loads((self.corpus/'catalog.json').read_text())
        assert all(set(r['answer'])<=set(catalog) for r in q.values())
        active={fold(r['question']) for r in q.values()}
        assert len(active)>=2, 'Train must cover at least two question-hash folds'
        d=self.work/'train';d.mkdir(exist_ok=True);qpath=d/'questions.json'
        contract={'questions':q,'smoke':self.a.smoke,'seeds':self.a.seeds,'trees':self.a.trees}
        cpath=d/'contract.json'
        if cpath.exists() and json.loads(cpath.read_text())!=contract:raise RuntimeError('Training drift: use a new work directory')
        write(cpath,contract);write(qpath,q);self.env['CLEAN_TRAIN']=str(qpath);self.env['RB_OUT']=str(d)
        self.retrieve(d,qpath,'train');self.teacher(d,qpath,'train')
        # Folds 0-4: fit the student without fold k, then score fold k so student features are
        # out-of-fold for the selector. Fold 5 (no held fold) is the full-data student used by predict.
        for k in range(6):
            if k<5 and k not in active:continue
            dest=d/('ce_ft_hard_kl0.0/outer%d'%k)
            args=['--outer',k,'--neg','hard','--kl','0.0']+(['--smoke'] if self.a.smoke else [])
            self.step('student_fit%d'%k,'train_ce.py',args,[dest/'adapter',dest/'held_pool.jsonl',dest/'manifest.json'],extra=contract)
            if k==5:continue
            for kind,held,score,top in [('pool','held_pool','held_scores',60),('extra','held_xpool','held_xscores',40),('extra2','held_xpool2','held_xscores2',40)]:
                if kind!='pool':
                    with (dest/(held+'.jsonl')).open('w') as out:
                        for line in (d/(kind+'_train.jsonl')).open():
                            row=json.loads(line)
                            if fold(q[row['qid']]['question'])==k:out.write(line)
                self.score('student%d_%s'%(k,score),qpath,dest/(held+'.jsonl'),dest/(score+'.jsonl'),dest/'adapter',top)
            inputs=[dest/(s+'.jsonl') for s in ('held_scores','held_xscores','held_xscores2')]
            self.second('student%d_pass2'%k,qpath,d/'pool_train.jsonl',inputs,dest/'held_scores2.jsonl',dest/'adapter')
        model=self.work/'model';model.mkdir(exist_ok=True)
        self.step('selector_fit','selector_stage.py',['fit',d,qpath,model,'--trees',self.a.trees,'--seeds',self.a.seeds],
            [model/'selector.json',*[model/('seed%s.txt'%s) for s in self.a.seeds.split(',')]],extra=contract)
        self.step('fitted_audit','model_audit.py',['--adapter',d/'ce_ft_hard_kl0.0/outer5/adapter'],[self.work/'parameter_audit_fitted.json'])
        write(model/'manifest.json',{'config_sha':digest(self.cfg),'training_contract_sha':digest(contract),
            'smoke_only':self.a.smoke,'selector':inventory(model,{'.txt'}),
            'student':inventory(d/'ce_ft_hard_kl0.0/outer5/adapter'),
            'parameter_audit_sha':sha(self.work/'parameter_audit_fitted.json'),
            'recipe':self.cfg['recipe'],'metric_claim':None})

    def predict(self):
        self.check_inputs(models=True);self.verify_index()
        model=self.work/'model';manifest=json.loads((model/'manifest.json').read_text())
        assert manifest['config_sha']==digest(self.cfg)
        assert manifest['selector']==inventory(model,{'.txt'}),'Selector changed'
        adapter=self.work/'train/ce_ft_hard_kl0.0/outer5/adapter'
        assert manifest['student']==inventory(adapter),'Student changed'
        audit_path=self.work/'parameter_audit_fitted.json'
        assert manifest['parameter_audit_sha']==sha(audit_path)
        audit=json.loads(audit_path.read_text())
        assert not audit['adapter_audit_required_after_fit'] and audit['total']<4_000_000_000
        q=questions(self.a.questions);qid=digest(q)[:20]
        d=self.work/('predict_'+qid);d.mkdir(exist_ok=True);qpath=d/'questions.json';write(qpath,q)
        self.retrieve(d,qpath,'public');self.teacher(d,qpath,'public')
        for kind,score,top in [('pool','public_scores',60),('extra','public_xscores',40),('extra2','public_xscores2',40)]:
            self.score(d.name+'_'+score,qpath,d/(kind+'_public.jsonl'),d/(score+'.jsonl'),adapter,top)
        inputs=[d/(s+'.jsonl') for s in ('public_scores','public_xscores','public_xscores2')]
        self.second(d.name+'_student_pass2',qpath,d/'pool_public.jsonl',inputs,d/'public_scores2.jsonl',adapter)
        self.step(d.name+'_selector','selector_stage.py',['predict',d,qpath,model],[d/'prediction.json'])
        self.step(d.name+'_package','package.py',[qpath,d/'prediction.json',self.corpus/'catalog.json',d/'final'],[d/'final'])
        # Manifest lives outside the checksummed final directory.
        write(d/'manifest.json',{'questions_sha':digest(q),'model_manifest_sha':sha(model/'manifest.json'),
            'submission_sha':sha(d/'final/submission.zip'),'smoke_only':manifest['smoke_only'],
            'inference_answers_used':False,'questions':len(q),'metric_claim':None})
        print('submission:',d/'final/submission.zip',flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--threads',type=int,default=3)
    ap.add_argument('--batch',type=int,default=8)
    sub=ap.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--contexts',type=Path);p.add_argument('--models',type=Path)
    p.add_argument('--es',default='http://127.0.0.1:19211');p.add_argument('--stage',choices=['all','corpus','index','dense'],default='all')
    p=sub.add_parser('train');p.add_argument('--train',type=Path,required=True);p.add_argument('--smoke',action='store_true')
    p.add_argument('--trees',type=int,default=1200);p.add_argument('--seeds',default='2026,7,99,314,2718,1618')
    p=sub.add_parser('predict');p.add_argument('--questions',type=Path,required=True)
    sub.add_parser('verify')
    a=ap.parse_args()
    if a.threads<1 or a.batch<1:ap.error('threads and batch must be positive')
    with lock(a.work):
        pipe=Pipeline(a)
        if a.command=='verify':
            pipe.check_inputs(models=True);pipe.verify_index()
            for p in sorted(pipe.state.glob('*.json')):
                for path,expected in json.loads(p.read_text())['outputs'].items():
                    assert inventory(Path(path))==expected,'Artifact drift: '+path
            print('verified input/model hashes, index ownership/count, and completed stage outputs')
        else:getattr(pipe,a.command)()


if __name__=='__main__':main()
