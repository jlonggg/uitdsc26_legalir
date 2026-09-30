import os
"""Pass 2: for the top-K docs by pass-1 max logit (BM25 pool ∪ dense extras), score up to `cap` chunks per doc
(ranked by wseg BM25 inside the doc). Output format = pass-1 rows; the selector merges chunk logits per doc.
usage: second_pass.py <model> <questions.json> <pool.jsonl> <ce_pass1.jsonl[,more]> <out.jsonl> [--top-docs 10] [--cap 40] [--shard i --nshards n]"""
import argparse, json, time, torch
from elasticsearch import Elasticsearch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
LEGACY = os.environ["CLEAN_INDEX"]; WSEG = os.environ["CLEAN_WSEG"]

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("questions"); ap.add_argument("pool"); ap.add_argument("pass1")
    ap.add_argument("out"); ap.add_argument("--top-docs", type=int, default=10); ap.add_argument("--cap", type=int, default=40)
    ap.add_argument("--max-len", type=int, default=1024); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshards", type=int, default=1); ap.add_argument("--adapter"); ap.add_argument("--from-doc", type=int, default=0); a = ap.parse_args()
    qs = json.load(open(a.questions, encoding="utf-8"))
    seg = {}
    for line in open(a.pool, encoding="utf-8"):
        r = json.loads(line); seg[str(r["qid"])] = r["seg"]
    best = {}
    for p in a.pass1.split(","):
        for line in open(p, encoding="utf-8"):
            if not line.strip(): continue
            r = json.loads(line); q = str(r["qid"]); best.setdefault(q, {})
            for d in r["docs"]: best[q][d["doc_id"]] = max(best[q].get(d["doc_id"], -1e9), max(d["logits"]))
    done = set()
    try:
        for line in open(a.out, encoding="utf-8"):
            if line.strip(): done.add(str(json.loads(line)["qid"]))
    except FileNotFoundError: pass
    qids = [q for k, q in enumerate(sorted(best)) if k % a.nshards == a.shard and q not in done]
    tok = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(a.model, torch_dtype=torch.float16, local_files_only=True)
    if a.adapter:
        from peft import PeftModel; model = PeftModel.from_pretrained(model, a.adapter)
    model = model.cuda().eval(); es = Elasticsearch(os.environ["CLEAN_ES"], request_timeout=180, retry_on_timeout=True, max_retries=5)
    t0 = time.time(); n_pairs = 0
    with open(a.out, "a", encoding="utf-8") as out, torch.inference_mode():
        for i, q in enumerate(qids, 1):
            question = str(qs[q]["question"]); top = sorted(best[q], key=lambda d: -best[q][d])[a.from_doc:a.top_docs]  # --from-doc 10 (pass 3): skip the top-10 docs already rescored by pass 2
            outdocs = []; pairs = []; owner = []
            for rank, doc in enumerate(top, 1):
                r = es.search(index=WSEG, body={"size": a.cap, "_source": False, "query": {"bool": {
                    "filter": [{"term": {"doc_id": doc}}], "should": [{"match": {"wseg": {"query": seg[q]}}}]}}})
                ids = [h["_id"] for h in r["hits"]["hits"]]
                if not ids: continue
                src = {d["_id"]: d.get("_source") or {} for d in es.mget(index=LEGACY, body={"ids": ids}, _source=["name", "article", "text"])["docs"]}
                outdocs.append({"doc_id": doc, "rank": rank, "chunks": ids, "logits": [0.0] * len(ids)})
                for j, c in enumerate(ids):
                    s = src[c]; pairs.append(f"{s.get('name','')}\n{s.get('article','')}\n{s.get('text','')}"); owner.append((len(outdocs) - 1, j))
            order = sorted(range(len(pairs)), key=lambda k: len(pairs[k]))
            for s in range(0, len(order), a.batch):
                idx = order[s:s + a.batch]
                enc = tok([question] * len(idx), [pairs[k] for k in idx], padding=True, truncation="only_second", max_length=a.max_len, return_tensors="pt").to("cuda")
                lg = model(**enc).logits.float(); lg = lg[:, -1] if lg.ndim == 2 and lg.shape[1] > 1 else lg.reshape(-1)
                for k, v in zip(idx, lg.tolist()): di, j = owner[k]; outdocs[di]["logits"][j] = round(v, 4)
            n_pairs += len(pairs); out.write(json.dumps({"qid": q, "docs": outdocs}) + "\n")
            if i % 20 == 0 or i == len(qids):
                out.flush(); el = time.time() - t0; print(f"{i}/{len(qids)} q {n_pairs/el:.1f} pairs/s {n_pairs/i:.0f} pairs/q eta {(len(qids)-i)*el/i/60:.0f} min", flush=True)

if __name__ == "__main__":
    main()
