"""Local exact retrieval; vectors are built here, never imported from Qdrant."""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from common import LegalChunk, dense_text, normalize_text


class Encoder:
    def __init__(self, name, device):
        self.name, self.device = name, device
        path = str(Path(os.environ['CLEAN_MODELS'])/name)
        if name == 'Qwen3-Embedding-0.6B':
            from transformers import AutoModel, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(path, local_files_only=True, padding_side='left')
            self.tok.pad_token = self.tok.pad_token or self.tok.eos_token
            self.model = AutoModel.from_pretrained(path, local_files_only=True,
                torch_dtype=torch.float16 if device.startswith('cuda') else torch.float32).to(device).eval()
        else:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(path, device=device, local_files_only=True, trust_remote_code=False)

    @torch.inference_mode()
    def encode(self, texts, query=False):
        if self.name != 'Qwen3-Embedding-0.6B':
            return self.model.encode([normalize_text(t) for t in texts], batch_size=len(texts),
                normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        if query:
            texts = ['Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:'+t for t in texts]
        batch = self.tok(texts, padding=True, truncation=True, max_length=192 if query else 320, return_tensors='pt').to(self.device)
        return torch.nn.functional.normalize(self.model(**batch).last_hidden_state[:, -1].float(), dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['encode', 'retrieve'])
    ap.add_argument('name', choices=['Qwen3-Embedding-0.6B', 'Vietnamese_Embedding_v2', 'vietlegal-harrier-0.6b'])
    ap.add_argument('--questions'); ap.add_argument('--out')
    ap.add_argument('--batch', type=int, default=16); ap.add_argument('--device', default='cuda')
    a = ap.parse_args()
    corpus = Path(os.environ['CLEAN_CORPUS'])
    article = a.name == 'Qwen3-Embedding-0.6B'
    rows = [json.loads(s) for s in (corpus/('articles.jsonl' if article else 'chunks.jsonl')).open()]
    vecpath = corpus/(a.name+'.npy')
    model = Encoder(a.name, a.device)
    if a.stage == 'encode':
        vector = None
        for start in range(0, len(rows), a.batch):
            block = rows[start:start+a.batch]
            texts = [(r['header']+'\n' if r['header'] else '')+r['body'] for r in block] if article else [dense_text(LegalChunk.from_dict(r)) for r in block]
            v = model.encode(texts)
            if vector is None:
                vector = np.lib.format.open_memmap(str(vecpath)+'.partial', mode='w+', dtype=np.float32, shape=(len(rows), v.shape[1]))
            vector[start:start+len(block)] = v
            if start % (a.batch*100) == 0:
                print(a.name, start, '/', len(rows), flush=True)
        vector.flush(); del vector
        os.replace(str(vecpath)+'.partial', vecpath)
        return
    vec = np.load(vecpath, mmap_mode='r')
    docs = sorted({r['doc_id'] for r in rows})
    number = {d: i for i, d in enumerate(docs)}
    owner = np.array([number[r['doc_id']] for r in rows])
    qs = json.loads(Path(a.questions).read_text())
    with open(a.out, 'w') as out:
        for q in sorted(qs):
            v = model.encode([qs[q]['question']], query=True)[0]
            scores = np.empty(len(rows), np.float32)
            for start in range(0, len(rows), 16000):
                scores[start:start+16000] = vec[start:start+16000] @ v
            best = np.full(len(docs), -np.inf, np.float32)
            if article:
                np.maximum.at(best, owner, scores)
            else:
                # Doc score = max over the top-400 chunks (MaxP); exact search, ties broken by row index.
                ix = np.lexsort((np.arange(len(rows)), -scores))[:400]
                np.maximum.at(best, owner[ix], scores[ix])
            order = np.lexsort((np.array(docs), -best))
            order = [i for i in order if np.isfinite(best[i])][:120]
            out.write(json.dumps({'qid': q, 'ranking': [docs[i] for i in order], 'scores': [float(best[i]) for i in order]})+'\n')
    print('retrieved', a.name, len(qs), flush=True)


if __name__ == '__main__':
    main()
