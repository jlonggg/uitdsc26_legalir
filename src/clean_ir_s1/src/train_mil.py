#!/usr/bin/env python3
"""MIL scorer training (S1 on Vietnamese_Reranker, S2 when CLEAN_CE is a Qwen3 reranker): LoRA fine-tune of the cross-encoder with multi-chunk evidence (MIL) and per-gold groups.
Group = (question, gold doc) : positive doc + N_HARD hardest non-gold pool docs (by the ranking file) + N_RAND random pool docs.
Doc score = logsumexp over its K evidence chunk logits (gradient finds the right chunk; no teacher-chosen positive).
Loss = cross-entropy over the group's doc scores. Reads only outer-train labels; writes <out>/outer<K>/adapter.
usage: train_mil.py --outer K --evidence evidence_train.jsonl --ranking ce_train.s0.jsonl[,...] [--epochs 2] [--lr 1e-4]
       [--rank 32] [--n-hard 6] [--n-rand 6] [--max-len 768] [--micro 13] [--smoke]"""
import argparse, json, random, collections, time, math, os
import numpy as np, torch, torch.nn.functional as F
from elasticsearch import Elasticsearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
MODEL = os.environ['CLEAN_CE']; LEGACY = os.environ['CLEAN_INDEX']; SEED = 368


from mil_common import fold, build_groups
from qwen_rr import is_qwen, QwenRR, LORA_TARGETS as QWEN_TARGETS


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--outer', type=int, required=True); ap.add_argument('--evidence', required=True)
    ap.add_argument('--ranking', required=True); ap.add_argument('--out', default=None); ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--lr', type=float, default=1e-4); ap.add_argument('--rank', type=int, default=32); ap.add_argument('--n-hard', type=int, default=6)
    ap.add_argument('--n-rand', type=int, default=6); ap.add_argument('--max-len', type=int, default=768); ap.add_argument('--micro', type=int, default=13)
    ap.add_argument('--accum', type=int, default=2); ap.add_argument('--smoke', action='store_true'); a = ap.parse_args()
    out = a.out or f"{os.environ['RB_OUT']}/mil/outer{a.outer}"; os.makedirs(out, exist_ok=True)
    train = json.load(open(os.environ['CLEAN_TRAIN'], encoding='utf-8'))
    evidence = {}
    for line in open(a.evidence, encoding='utf-8'):
        r = json.loads(line); evidence[str(r['qid'])] = r['docs']
    best = collections.defaultdict(dict)  # hard-negative order: max logit per doc over the ranking files (teacher pass-1 by default)
    for p in a.ranking.split(','):
        for line in open(p, encoding='utf-8'):
            if not line.strip(): continue
            r = json.loads(line); q = str(r['qid'])
            for d in r['docs']: best[q][d['doc_id']] = max(best[q].get(d['doc_id'], -1e9), max(d['logits']))
    ranking = {q: sorted(v, key=lambda d: (-v[d], d)) for q, v in best.items()}
    rng = random.Random(SEED + a.outer)
    groups, skipped = build_groups(train, evidence, ranking, a.outer, a.n_hard, a.n_rand, rng)
    held = [q for q in sorted(train) if fold(train[q]['question']) == a.outer]
    print(f'outer={a.outer} groups={len(groups)} skipped={skipped} held={len(held)}', flush=True)
    if a.smoke: groups = groups[:4]
    es = Elasticsearch(os.environ['CLEAN_ES'], request_timeout=180, retry_on_timeout=True, max_retries=5)
    ids = sorted({c for g in groups for cs in g['chunks'] for c in cs}); text = {}
    for s in range(0, len(ids), 500):
        for d in es.mget(index=LEGACY, body={'ids': ids[s:s + 500]}, _source=['name', 'article', 'text'])['docs']:
            src = d.get('_source') or {}; text[d['_id']] = f"{src.get('name','')}\n{src.get('article','')}\n{src.get('text','')}"
    assert groups, 'No training groups'
    torch.manual_seed(SEED + a.outer); np.random.seed(SEED + a.outer)
    qwen = QwenRR(MODEL) if is_qwen(MODEL) else None  # S2: Qwen3-Reranker base, same recipe; fp16 base, LoRA fp32
    tok = qwen.tok if qwen else AutoTokenizer.from_pretrained(MODEL, local_files_only=True, trust_remote_code=True)
    model = qwen.model if qwen else AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.float32, local_files_only=True, trust_remote_code=True)
    model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.05, bias='none', task_type='CAUSAL_LM' if qwen else 'SEQ_CLS',
                                             target_modules=QWEN_TARGETS if qwen else os.environ.get('LORA_TARGETS', r'.*\.(query|key|value|attention\.output\.dense)$')))
    model.cuda(); model.train(); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr, weight_decay=0.01)
    total = math.ceil(len(groups) * a.epochs / a.accum); sch = get_linear_schedule_with_warmup(opt, max(1, total // 10), max(1, total))
    scaler = torch.cuda.amp.GradScaler(); order_rng = random.Random(SEED + a.outer + 7); losses = []; step = 0; t0 = time.time()
    ck = f'{out}/ckpt.pt'; start = 0; EVERY = int(os.environ.get('CKPT_EVERY', 500))  # resume: a fold fit takes many GPU-hours and must not restart from step 0 if the process dies
    if os.path.exists(ck):
        s = torch.load(ck, map_location='cpu'); model.load_state_dict(s['lora'], strict=False); opt.load_state_dict(s['opt']); sch.load_state_dict(s['sch'])
        scaler.load_state_dict(s['scaler']); start, losses = s['step'], s['losses']; torch.set_rng_state(s['rng']); torch.cuda.set_rng_state(s['cuda_rng'])
        print(f'resume from step {start}', flush=True)
    for ep in range(a.epochs):
        order = list(range(len(groups))); order_rng.shuffle(order)
        for gi in order:
            if step < start: step += 1; continue  # replay the epoch shuffles deterministically, skip what the ckpt already covers
            g = groups[gi]; pairs = [text[c] for cs in g['chunks'] for c in cs]; owner = torch.tensor([i for i, cs in enumerate(g['chunks']) for _ in cs], device='cuda')
            logits = []
            for s in range(0, len(pairs), a.micro):  # micro-batches keep the graph: one softmax over all docs of the group
                enc = qwen.encode(g['question'], pairs[s:s + a.micro], a.max_len) if qwen else tok([g['question']] * len(pairs[s:s + a.micro]), pairs[s:s + a.micro], padding=True, truncation='only_second', max_length=a.max_len, return_tensors='pt').to('cuda')
                with torch.autocast('cuda', dtype=torch.float16):
                    if qwen: logits.append(qwen.logits(model, enc).float())
                    else: lg = model(**enc).logits.float(); logits.append(lg[:, -1] if lg.ndim == 2 and lg.shape[1] > 1 else lg.reshape(-1))
            lg = torch.cat(logits); ndoc = len(g['docs'])
            doc = torch.full((ndoc,), -1e4, device='cuda').scatter_reduce(0, owner, lg, reduce='amax')  # stable LSE: max + log sum exp(x - max)
            doc = doc + torch.zeros(ndoc, device='cuda').scatter_add(0, owner, torch.exp(lg - doc[owner])).log()
            loss = F.cross_entropy(doc[None], torch.zeros(1, dtype=torch.long, device='cuda'))  # positive doc is index 0
            assert torch.isfinite(loss), 'Non-finite training loss'
            scaler.scale(loss / a.accum).backward(); losses.append(float(loss)); step += 1
            if step % a.accum == 0 or step == len(groups) * a.epochs:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
                if step % EVERY == 0:  # on an optimizer boundary, so no partial accumulation is lost; atomic replace
                    torch.save({'lora': {k: v for k, v in model.state_dict().items() if 'lora_' in k}, 'opt': opt.state_dict(), 'sch': sch.state_dict(), 'scaler': scaler.state_dict(),
                                'step': step, 'losses': losses, 'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state()}, ck + '.tmp'); os.replace(ck + '.tmp', ck)
            if step % 200 == 0: print(f'step {step}/{len(groups)*a.epochs} loss={np.mean(losses[-200:]):.4f} {(step-start)/(time.time()-t0)*60:.0f} grp/min', flush=True)
    model.save_pretrained(f'{out}/adapter'); tok.save_pretrained(f'{out}/adapter')
    json.dump({'outer': a.outer, 'groups': len(groups), 'skipped': skipped, 'epochs': a.epochs, 'lr': a.lr, 'rank': a.rank, 'n_hard': a.n_hard,
               'n_rand': a.n_rand, 'max_len': a.max_len, 'mean_loss': float(np.mean(losses)), 'last200': float(np.mean(losses[-200:])),
               'smoke': a.smoke, 'held_labels_read': False, 'held': len(held), 'base': os.path.basename(MODEL.rstrip('/')),
               'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)}, open(f'{out}/manifest.json', 'w'), indent=1)
    print('saved', out, flush=True)


if __name__ == '__main__':
    main()
