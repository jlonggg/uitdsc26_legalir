import os
"""LoRA fine-tune of the cross-encoder on outer-train folds with controlled negative sampling.
Negatives are chosen by --neg (the pipeline uses hard: the 7 top teacher-scored non-gold docs); --kl adds optional
distillation towards teacher logits. --outer 5 has no held fold, i.e. the student is fit on all training questions.
usage: train_ce.py --outer K --neg {hard,denoised,random} [--kl 0.0] [--epochs 1] [--lr 1e-4] [--max-len 768] [--smoke]
Reads only outer-train labels. Writes $RB_OUT/ce_ft_<neg>_kl<kl>/outer<K>/{adapter,held_pool.jsonl,manifest.json}."""
import argparse, json, random, re, hashlib, unicodedata, collections, time, math, os
import numpy as np, torch, torch.nn.functional as F
from elasticsearch import Elasticsearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, get_peft_model
ROOT = os.environ["CLEAN_WORK"]; MODEL = os.environ["CLEAN_CE"]; LEGACY = os.environ["CLEAN_INDEX"]
N_NEG = 7; SEED = 368

def normalize(q):
    t = unicodedata.normalize("NFC", q).lower(); return " ".join(re.findall(r"[^\W_]+", t, flags=re.UNICODE))
def fold(q): return int(hashlib.sha256(normalize(q).encode()).hexdigest()[:8], 16) % 5

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--outer", type=int, required=True); ap.add_argument("--neg", required=True)
    ap.add_argument("--kl", type=float, default=0.0); ap.add_argument("--epochs", type=int, default=1); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=768); ap.add_argument("--smoke", action="store_true"); ap.add_argument("--accum", type=int, default=4)
    a = ap.parse_args()
    out = f"{os.environ.get('RB_OUT', ROOT + '/rebuild/out')}/ce_ft_{a.neg}_kl{a.kl}/outer{a.outer}"; os.makedirs(out, exist_ok=True)
    train = json.load(open(os.environ["CLEAN_TRAIN"], encoding="utf-8"))
    pool = {}; 
    for line in open(f"{os.environ.get('RB_OUT', ROOT + '/rebuild/out')}/pool_train.jsonl", encoding="utf-8"):
        r = json.loads(line); pool[str(r["qid"])] = r["docs"]
    teacher = {}
    for s in range(3):
        path = f"{os.environ.get('RB_OUT', ROOT + '/rebuild/out')}/ce_train.s{s}.jsonl"
        if not os.path.exists(path): continue
        for line in open(path, encoding="utf-8"):
            if line.strip(): r = json.loads(line); teacher[str(r["qid"])] = r["docs"]
    qids = sorted(train); folds = {q: fold(train[q]["question"]) for q in qids}
    held = [q for q in qids if folds[q] == a.outer]; tr = [q for q in qids if folds[q] != a.outer and q in teacher]
    with open(f"{out}/held_pool.jsonl", "w", encoding="utf-8") as h:
        for q in held: h.write(json.dumps({"qid": q, "docs": pool[q]}, ensure_ascii=False) + "\n")
    # duplicate-question golds (outer-train only) are excluded from negatives: annotation variance, not irrelevance
    dup_gold = collections.defaultdict(set)
    for q in tr:
        dup_gold[normalize(train[q]["question"])].update(map(str, train[q]["answer"]))
    rng = random.Random(SEED + a.outer)
    groups = []  # (question, positive_chunk_id, [negative_chunk_ids], teacher_logits)
    skipped = 0
    for q in tr:
        gold = set(map(str, train[q]["answer"])); docs = teacher[q]
        gdocs = [d for d in docs if d["doc_id"] in gold]
        if not gdocs: skipped += 1; continue
        gd = max(gdocs, key=lambda d: max(d["logits"]))  # teacher-selected positive doc & chunk
        pj = int(np.argmax(gd["logits"])); pos = (gd["chunks"][pj], gd["logits"][pj])
        banned = gold | dup_gold[normalize(train[q]["question"])]
        nongold = [d for d in sorted(docs, key=lambda d: -max(d["logits"])) if d["doc_id"] not in banned]
        if a.neg == "hard": cand = nongold[:N_NEG]
        elif a.neg == "denoised": cand = rng.sample(nongold[3:40], min(N_NEG, len(nongold[3:40])))   # skip 3 hardest: likely unlabelled positives
        elif a.neg == "random": cand = rng.sample(nongold, min(N_NEG, len(nongold)))
        else: raise SystemExit(a.neg)
        negs = [(d["chunks"][int(np.argmax(d["logits"]))], max(d["logits"])) for d in cand]
        if len(negs) < 2: skipped += 1; continue
        groups.append((str(train[q]["question"]), pos, negs))
    print(f"outer={a.outer} neg={a.neg} groups={len(groups)} skipped={skipped} held={len(held)}", flush=True)
    if a.smoke: groups = groups[:8]
    es = Elasticsearch(os.environ["CLEAN_ES"], request_timeout=180, retry_on_timeout=True, max_retries=5)
    ids = sorted({c for _, p, n in groups for c in [p[0]] + [x[0] for x in n]}); text = {}
    for s in range(0, len(ids), 500):
        for d in es.mget(index=LEGACY, body={"ids": ids[s:s + 500]}, _source=["name", "article", "text"])["docs"]:
            src = d.get("_source") or {}; text[d["_id"]] = f"{src.get('name','')}\n{src.get('article','')}\n{src.get('text','')}"
    assert groups, "No training groups with a positive and two negatives"
    torch.manual_seed(SEED + a.outer); np.random.seed(SEED + a.outer)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, torch_dtype=torch.float32, local_files_only=True)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="SEQ_CLS",
                                             target_modules=["query", "key", "value"]))
    model.cuda(); model.train(); model.gradient_checkpointing_enable(); model.enable_input_require_grads()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr, weight_decay=0.01)
    total = math.ceil(len(groups) * a.epochs / a.accum); sch = get_linear_schedule_with_warmup(opt, max(1, total // 10), max(1, total))
    scaler = torch.cuda.amp.GradScaler(); torch.manual_seed(SEED + a.outer); order_rng = random.Random(SEED + a.outer + 7)
    losses = []; step = 0; t0 = time.time()
    for ep in range(a.epochs):
        order = list(range(len(groups))); order_rng.shuffle(order)
        for gi in order:
            question, pos, negs = groups[gi]; items = [pos] + negs
            enc = tok([question] * len(items), [text[c] for c, _ in items], padding=True, truncation="only_second",
                      max_length=a.max_len, return_tensors="pt").to("cuda")
            with torch.autocast("cuda", dtype=torch.float16):
                lg = model(**enc).logits.float(); lg = lg[:, -1] if lg.ndim == 2 and lg.shape[1] > 1 else lg.reshape(-1)
            loss = F.cross_entropy(lg[None], torch.zeros(1, dtype=torch.long, device="cuda"))  # positive is index 0
            if a.kl > 0:
                t = torch.tensor([l for _, l in items], device="cuda")
                loss = loss + a.kl * F.kl_div(F.log_softmax(lg, -1), F.softmax(t, -1), reduction="sum")
            assert torch.isfinite(loss), "Non-finite training loss"
            remaining = len(groups) * a.epochs % a.accum
            divisor = remaining if remaining and step >= len(groups) * a.epochs - remaining else a.accum
            scaler.scale(loss / divisor).backward(); losses.append(float(loss)); step += 1
            if step % a.accum == 0 or step == len(groups) * a.epochs:
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sch.step(); opt.zero_grad(set_to_none=True)
            if step % 200 == 0: print(f"step {step}/{len(groups)*a.epochs} loss={np.mean(losses[-200:]):.4f} {step/(time.time()-t0)*60:.0f} grp/min", flush=True)
    model.save_pretrained(f"{out}/adapter"); tok.save_pretrained(f"{out}/adapter")
    json.dump({"outer": a.outer, "neg": a.neg, "kl": a.kl, "groups": len(groups), "skipped": skipped, "epochs": a.epochs, "lr": a.lr,
               "max_len": a.max_len, "mean_loss": float(np.mean(losses)), "last200": float(np.mean(losses[-200:])), "smoke": a.smoke,
               "held_labels_read": False}, open(f"{out}/manifest.json", "w"), indent=1)
    print("saved", out, flush=True)

if __name__ == "__main__":
    main()
