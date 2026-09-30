import os
"""Score (question, chunk) pairs with a cross-encoder over a candidate pool (BM25 pool or dense extras). Label-blind.
usage: score_ce.py <model_dir> <questions.json> <pool.jsonl> <out.jsonl> [--top 60] [--max-len 1024] [--adapter DIR]
Output row: {qid, docs:[{doc_id, rank, chunks:[id..], logits:[..]}]}  (raw logits, resumable)"""
import argparse, json, time, torch
from elasticsearch import Elasticsearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
LEGACY = os.environ["CLEAN_INDEX"]

def chunk_texts(es, ids):
    r = es.mget(index=LEGACY, body={"ids": ids}, _source=["name", "article", "text"])
    out = {}
    for d in r["docs"]:
        s = d.get("_source") or {}
        out[d["_id"]] = f"{s.get('name','')}\n{s.get('article','')}\n{s.get('text','')}"
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("questions"); ap.add_argument("pool")
    ap.add_argument("out"); ap.add_argument("--top", type=int, default=60); ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--adapter")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); a = ap.parse_args()
    qs = json.load(open(a.questions, encoding="utf-8"))
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, torch_dtype=torch.float16, local_files_only=True)
    if a.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, a.adapter)
    model = model.cuda().eval()
    es = Elasticsearch(os.environ["CLEAN_ES"], request_timeout=180, retry_on_timeout=True, max_retries=5)
    done = set()
    try:
        for line in open(a.out, encoding="utf-8"):
            if line.strip(): done.add(str(json.loads(line)["qid"]))
    except FileNotFoundError: pass
    rows = [json.loads(l) for l in open(a.pool, encoding="utf-8") if l.strip()]
    rows = [r for k, r in enumerate(rows) if k % a.nshards == a.shard]
    rows = [r for r in rows if str(r["qid"]) not in done]; t0 = time.time(); n_pairs = 0
    with open(a.out, "a", encoding="utf-8") as out, torch.inference_mode():
        for i, r in enumerate(rows, 1):
            qid = str(r["qid"]); question = str(qs[qid]["question"]); docs = r["docs"][:a.top]
            ids = [c for d in docs for c in d["chunks"]]
            if not ids:
                out.write(json.dumps({"qid": qid, "docs": []}) + "\n"); continue
            texts = chunk_texts(es, ids)
            pairs = [texts[c] for c in ids]; logits = []
            # sort by length for padding efficiency, then restore order
            order = sorted(range(len(pairs)), key=lambda k: len(pairs[k])); buf = [0.0] * len(pairs)
            for s in range(0, len(order), a.batch):
                idx = order[s:s + a.batch]
                enc = tok([question] * len(idx), [pairs[k] for k in idx], padding=True, truncation="only_second",
                          max_length=a.max_len, return_tensors="pt").to("cuda")
                lg = model(**enc).logits.float()
                lg = lg[:, -1] if lg.ndim == 2 and lg.shape[1] > 1 else lg.reshape(-1)
                for k, v in zip(idx, lg.tolist()): buf[k] = v
            p = 0; outdocs = []
            for rank, d in enumerate(docs, 1):
                m = len(d["chunks"]); outdocs.append({"doc_id": d["doc_id"], "rank": rank, "chunks": d["chunks"],
                                                     "logits": [round(x, 4) for x in buf[p:p + m]]}); p += m
            n_pairs += len(pairs)
            out.write(json.dumps({"qid": qid, "docs": outdocs}) + "\n")
            if i % 20 == 0 or i == len(rows):
                out.flush(); el = time.time() - t0
                print(f"{len(done)+i}/{len(done)+len(rows)} q  {n_pairs/el:.1f} pairs/s  eta {(len(rows)-i)*el/i/60:.0f} min", flush=True)

if __name__ == "__main__":
    main()
