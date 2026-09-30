#!/usr/bin/env python3
"""Score every evidence chunk of every pool document with the (LoRA) cross-encoder. Output = pass-1 row format
{qid, docs:[{doc_id, rank, chunks, logits}]}, so the selector reads it exactly like a teacher file. Resumable.
usage: score_mil.py <model_dir> <questions.json> <evidence.jsonl> <out.jsonl> [--adapter DIR[,DIR...]] [--max-len 768]
Several adapters = bagged inference: logits are averaged per pair (train-side features come from one held-fold adapter each;
inference averages the fold adapters instead of refitting on all data, so no extra training run is needed)."""
import argparse, json, time, torch, os
from elasticsearch import Elasticsearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qwen_rr import is_qwen, QwenRR
LEGACY = os.environ['CLEAN_INDEX']


def texts_of(es, ids):
    out = {}
    for s in range(0, len(ids), 500):
        for d in es.mget(index=LEGACY, body={'ids': ids[s:s + 500]}, _source=['name', 'article', 'text'])['docs']:
            src = d.get('_source') or {}; out[d['_id']] = f"{src.get('name','')}\n{src.get('article','')}\n{src.get('text','')}"
    return out


def load(model_dir, adapter, qwen=None):
    model = qwen.model if qwen else AutoModelForSequenceClassification.from_pretrained(model_dir, torch_dtype=torch.float16, local_files_only=True, trust_remote_code=True)
    if adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, adapter)
    return model.cuda().eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('model'); ap.add_argument('questions'); ap.add_argument('evidence'); ap.add_argument('out')
    ap.add_argument('--adapter', default=''); ap.add_argument('--max-len', type=int, default=768); ap.add_argument('--batch', type=int, default=16); a = ap.parse_args()
    qs = json.load(open(a.questions, encoding='utf-8')); qwen = is_qwen(a.model)  # S2 base: one QwenRR per adapter (weights are per model)
    tok = None if qwen else AutoTokenizer.from_pretrained(a.model, local_files_only=True, trust_remote_code=True)
    adapters = [x for x in a.adapter.split(',') if x] or [None]
    es = Elasticsearch(os.environ['CLEAN_ES'], request_timeout=180, retry_on_timeout=True, max_retries=5)
    done = set()
    try:
        for line in open(a.out, encoding='utf-8'):
            if line.strip(): done.add(str(json.loads(line)['qid']))
    except FileNotFoundError: pass
    rows = [json.loads(l) for l in open(a.evidence, encoding='utf-8') if l.strip()]; rows = [r for r in rows if str(r['qid']) not in done]
    t0 = time.time(); n_pairs = 0
    with open(a.out, 'a', encoding='utf-8') as out, torch.inference_mode():
        qrs = [QwenRR(a.model) if qwen else None for _ in adapters]; models = [load(a.model, ad, qr) for ad, qr in zip(adapters, qrs)]
        for i, r in enumerate(rows, 1):
            qid = str(r['qid']); question = str(qs[qid]['question']); ids = [c for d in r['docs'] for c in d['chunks']]
            texts = texts_of(es, ids); pairs = [texts[c] for c in ids]
            order = sorted(range(len(pairs)), key=lambda k: len(pairs[k])); buf = [0.0] * len(pairs)
            for s in range(0, len(order), a.batch):
                idx = order[s:s + a.batch]
                enc = None if qwen else tok([question] * len(idx), [pairs[k] for k in idx], padding=True, truncation='only_second', max_length=a.max_len, return_tensors='pt').to('cuda')
                acc = None
                for model, qr in zip(models, qrs):
                    if qwen: lg = qr.logits(model, qr.encode(question, [pairs[k] for k in idx], a.max_len)).float()
                    else: lg = model(**enc).logits.float(); lg = lg[:, -1] if lg.ndim == 2 and lg.shape[1] > 1 else lg.reshape(-1)
                    acc = lg if acc is None else acc + lg
                for k, v in zip(idx, (acc / len(models)).tolist()): buf[k] = v
            p = 0; outdocs = []
            for d in r['docs']:
                m = len(d['chunks']); outdocs.append({'doc_id': d['doc_id'], 'rank': d['rank'], 'chunks': d['chunks'], 'logits': [round(x, 4) for x in buf[p:p + m]]}); p += m
            n_pairs += len(pairs); out.write(json.dumps({'qid': qid, 'docs': outdocs}) + '\n')
            if i % 20 == 0 or i == len(rows):
                out.flush(); el = time.time() - t0; print(f'{len(done)+i}/{len(done)+len(rows)} q {n_pairs/el:.1f} pairs/s eta {(len(rows)-i)*el/i/60:.0f} min', flush=True)


if __name__ == '__main__':
    main()
