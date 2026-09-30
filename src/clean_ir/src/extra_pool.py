import os
"""Pool rows for documents in dense top-K that the BM25 pool (top-60) missed; chunks = best-3 wseg chunks
inside that document for the same segmented query. usage: extra_pool.py <pool.jsonl> <dense.jsonl> <out.jsonl> [K=30]"""
import json, sys, time
from elasticsearch import Elasticsearch
WSEG = os.environ["CLEAN_WSEG"]
def main(pool_path, dense_path, out_path, K=30):
    pool = {}; seg = {}
    for line in open(pool_path, encoding="utf-8"):
        r = json.loads(line); pool[str(r["qid"])] = {d["doc_id"] for d in r["docs"][:60]}; seg[str(r["qid"])] = r["seg"]
    dense = {}
    for line in open(dense_path, encoding="utf-8"):
        r = json.loads(line); dense[str(r["qid"])] = r["ranking"][:K]
    es = Elasticsearch(os.environ["CLEAN_ES"], request_timeout=180, retry_on_timeout=True, max_retries=5)
    t0 = time.time(); n = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, (qid, ranking) in enumerate(dense.items(), 1):
            extra = [d for d in ranking if d not in pool[qid]]; docs = []
            for rank, doc in enumerate(extra, 1):
                r = es.search(index=WSEG, body={"size": 3, "_source": False, "query": {"bool": {
                    "filter": [{"term": {"doc_id": doc}}], "should": [{"match": {"wseg": {"query": seg[qid]}}}]}}})
                hits = r["hits"]["hits"]
                if not hits:  # empty-passage document: nothing to score
                    continue
                docs.append({"doc_id": doc, "score": 0.0, "rank": 60 + rank, "chunks": [h["_id"] for h in hits],
                             "chunk_scores": [h["_score"] for h in hits]})
            n += len(docs)
            out.write(json.dumps({"qid": qid, "seg": seg[qid], "docs": docs}, ensure_ascii=False) + "\n")
            if i % 500 == 0: print(f"{i}/{len(dense)} extra docs so far {n} {i/(time.time()-t0)*60:.0f}/min", flush=True)
    print("done", n)
if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 30)
