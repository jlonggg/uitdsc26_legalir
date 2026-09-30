import os
"""Label-blind candidate pool: wordseg BM25 top-100 docs with their top-3 chunk ids (inner_hits).
usage: build_pool.py <questions.json> <out.jsonl>  (resumable)"""
import json, sys, time
from elasticsearch import Elasticsearch

WSEG = os.environ["CLEAN_WSEG"]
STOPWORDS = {"và", "của", "là", "được", "theo", "tại", "cho", "trong", "về", "với",
             "các", "những", "một", "thì", "đó", "này", "nào", "như_thế_nào"}
from segmentation import segment

def main(qpath, opath, size=100, chunks=3):
    qs = json.load(open(qpath, encoding="utf-8")); done = set()
    try:
        for line in open(opath, encoding="utf-8"):
            if line.strip(): done.add(str(json.loads(line)["qid"]))
    except FileNotFoundError: pass
    es = Elasticsearch(os.environ["CLEAN_ES"], request_timeout=180, retry_on_timeout=True, max_retries=5)
    todo = [q for q in qs if str(q) not in done]; t0 = time.time()
    with open(opath, "a", encoding="utf-8") as out:
        for i, qid in enumerate(todo, 1):
            seg = segment(str(qs[qid]["question"]))
            stopped = " ".join(t for t in seg.split() if t.casefold() not in STOPWORDS)
            r = es.search(index=WSEG, body={
                "size": size, "_source": ["doc_id"], "query": {"match": {"wseg": {"query": stopped}}},
                "collapse": {"field": "doc_id", "inner_hits": {"name": "top", "size": chunks, "_source": False}}})
            docs = []
            for h in r["hits"]["hits"]:
                inner = h["inner_hits"]["top"]["hits"]["hits"]
                docs.append({"doc_id": str(h["_source"]["doc_id"]), "score": h["_score"],
                             "chunks": [x["_id"] for x in inner], "chunk_scores": [x["_score"] for x in inner]})
            out.write(json.dumps({"qid": str(qid), "seg": stopped, "docs": docs}, ensure_ascii=False) + "\n")
            if i % 100 == 0 or i == len(todo):
                out.flush(); print(f"{len(done)+i}/{len(qs)} {i/(time.time()-t0)*60:.0f}/min", flush=True)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
